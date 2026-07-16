#!/usr/bin/env python3
#
# Copyright (c) 2026, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
#
# All rights reserved.
#
# This software is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import os

from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from py_trees.blackboard import Blackboard
from py_trees.common import Status
from py_trees.composites import Sequence
from py_trees.parsers.behaviour_tree_xml import parse_behaviour_tree_xml
from py_trees_ros.trees import BehaviourTree

from imetro_behavior_msgs.action import ExecuteBehavior


class BehaviorTreeExecutor:
    """
    Behavior tree executor with a ROS action server for starting and stopping behaviors.
    """

    def __init__(self, node: Node, dt: float = 0.1, search_paths: list[str] = None):
        """
        Initializes a behavior tree executor.

        Args:
            node: The ROS node associated with this behavior tree executor.
            dt: The time step at which to tick the active behavior tree.
            search_paths: A list of search paths to find behavior tree XML files and subtrees.
        """
        self._node = node
        self._logger = node.get_logger()
        self._clock = node.get_clock()
        self._dt = dt
        self._search_paths = search_paths or []

        # Create a global TF buffer to share across behaviors.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        Blackboard.set("ros/tf_buffer", self._tf_buffer)

        # Initialize state.
        self._tree = None
        self._current_behavior = None
        self._rate = self._node.create_rate(1.0 / self._dt)
        self._max_duration = Duration(seconds=self._dt)

        # Set up ROS action client
        self._action_server = ActionServer(
            self._node,
            ExecuteBehavior,
            "execute_behavior",
            goal_callback=self.goal_cb,
            execute_callback=self.run_tree,
            cancel_callback=self.cancel_tree_cb,
            callback_group=ReentrantCallbackGroup(),
        )

        self._logger.info("Behavior tree executor ready!")

    def run_tree(self, goal_handle: ServerGoalHandle) -> ExecuteBehavior.Result:
        """
        Starts a new behavior and runs it to completion, or until it's canceled..
        """
        result = ExecuteBehavior.Result()

        tree_file_name = goal_handle.request.tree_file_name
        if not tree_file_name:
            result.message = "Behavior tree file name cannot be empty!"
            self._logger.error(result.message)
            goal_handle.abort()
            return result

        # First get the path using the search paths.
        xml_name = tree_file_name if tree_file_name.endswith(".xml") else f"{tree_file_name}.xml"
        xml_path = None
        if os.path.exists(xml_name):
            xml_path = xml_name
        else:
            for path in self._search_paths:
                candidate_path = os.path.join(path, xml_name)
                if os.path.exists(candidate_path):
                    xml_path = candidate_path
        if xml_path is None:
            result.message = f"Could not find tree: {xml_name}"
            self._logger.error(result.message)
            goal_handle.abort()
            return result

        self._logger.info(f"Running behavior: {tree_file_name}")
        try:
            root = parse_behaviour_tree_xml(xml_path, search_paths=self._search_paths)
        except Exception as e:
            result.message = f"Failed to parse XML file: {e}"
            goal_handle.abort()
            return result

        # This is ugly, but it's how to get the PyTrees viewer to behave well.
        # We create a root Sequence node that never gets replaced, and all other operations will swap
        # the (single) child node of this root.
        # If this is the first behavior we run, we set it up as normal; else, we use `replace_subtree`.
        if self._tree is None:
            self._root_sequence = Sequence(name="root", memory=True)
            self._root_sequence.add_child(root)
            self._tree = BehaviourTree(root=self._root_sequence)
        else:
            self._tree.replace_subtree(self._tree.root.children[0].id, root)

        self._tree.setup(node=self._node)

        # Once the tree is set up, tick it periodically, checking for completion or cancellation.
        while self._tree.root.status not in (Status.SUCCESS, Status.FAILURE):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.message = "Behavior canceled."
                self._logger.info(result.message)
                self.stop_tree()
                return result

            t_start = self._clock.now()
            self._tree.tick()
            t_elapsed = self._clock.now() - t_start

            if t_elapsed > self._max_duration:
                self._logger.warning(f"Overrunning behavior tree tick rate of {self._dt} s")

            self._rate.sleep()

        # If the tree has completed, return whether it succeeded.
        final_status = self._tree.root.status
        self._logger.info(f"Behavior completed with status: {final_status}")
        self.stop_tree()
        goal_handle.succeed()
        result.success = final_status == Status.SUCCESS
        return result

    def goal_cb(self, goal_request: ExecuteBehavior.Goal) -> GoalResponse:
        if self._current_behavior is not None:
            self._logger.error(f"Already running behavior: {self._current_behavior}")
            return GoalResponse.REJECT

        self._current_behavior = goal_request.tree_file_name
        return GoalResponse.ACCEPT

    def cancel_tree_cb(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        """Cancellation callback for ROS action server."""
        return CancelResponse.ACCEPT

    def stop_tree(self):
        """Stops the currently running behavior tree and resets its state."""
        if self._current_behavior is not None:
            self._logger.info("Stopping current behavior...")
            self._current_behavior = None
            self._tree.root.status = Status.INVALID
            self._tree.shutdown(destroy_node=False)
