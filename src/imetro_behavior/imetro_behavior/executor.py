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
import threading
from importlib import import_module
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_path
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from py_trees.composites import Sequence
from py_trees.parsers.behaviour_tree_xml import parse_behaviour_tree_xml
from py_trees.ports import NoDataAvailable
from py_trees_ros.trees import BehaviourTree
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from imetro_behavior_msgs.action import ExecuteBehavior

DEFAULT_CONFIG = get_package_share_path("imetro_behavior") / "config" / "default_config.yaml"


class BehaviorTreeExecutor:
    """
    Loads, runs, and stops behavior trees defined in XML files.

    This class has no ROS action interfaces, so it can be used (and tested) directly.
    """

    def __init__(self, node: Node, config: dict | Path | str | None = None):
        """
        Initializes a behavior tree executor.

        Args:
            node: The ROS node associated with this behavior tree executor.
            config: Optional configuration, either as a dictionary or a path to a YAML file.
                If None, the path is read from the node's `behavior_config` parameter,
                which itself falls back to this package's default configuration.
        """
        self._node = node
        self._logger = node.get_logger()
        self._clock = node.get_clock()

        if config is None:
            self._node.declare_parameter("behavior_config", DEFAULT_CONFIG.as_posix())
            config = Path(self._node.get_parameter("behavior_config").get_parameter_value().string_value)
        self._load_config(config)

        # Create a global TF buffer to share across behaviors.
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._init_global_blackboard()

        # Initialize state.
        self._tree = None
        self._current_behavior = None
        self._tick_period = Duration(seconds=self._dt)
        self._cancel_requested = threading.Event()

    def _init_global_blackboard(self):
        """Initializes the blackboard and places shared resources on it as necessary."""
        Blackboard.clear()
        Blackboard.set("/ros/tf_buffer", self._tf_buffer)

    def _load_config(self, config: dict | Path | str):
        """Loads parameters, imports, and tree search paths from a dictionary or YAML configuration file."""
        if not isinstance(config, dict):
            with open(config) as file:
                config = yaml.safe_load(file)

        # Load general parameters
        params_config = config.get("params", {})
        self._dt = params_config.get("dt", 0.1)

        # Import any modules specified
        for module_to_import in config.get("imports", []):
            import_module(module_to_import)

        # Set the search paths, including all the subfolders below each specified path.
        self._search_paths = []
        for path_config in config.get("tree_search_paths", []):
            package = path_config.get("package")
            path = path_config.get("path")
            if package is None or path is None:
                raise RuntimeError("Search paths must specify 'package' and a 'path' fields.")

            full_path = get_package_share_path(package) / path
            self._search_paths.append(full_path.as_posix())
            self._search_paths.extend([p.as_posix() for p in full_path.rglob("*") if p.is_dir()])

    @property
    def current_behavior(self) -> str | None:
        """The name of the currently loaded behavior tree, if any."""
        return self._current_behavior

    @property
    def status(self) -> Status:
        """The status of the current behavior tree, or INVALID if no tree is loaded."""
        return self._tree.root.status if self._tree is not None else Status.INVALID

    def find_tree(self, tree_file_name: str) -> str | None:
        """Finds a tree XML file, either as a direct path or through the configured search paths."""
        xml_name = tree_file_name if tree_file_name.endswith(".xml") else f"{tree_file_name}.xml"
        if os.path.exists(xml_name):
            return xml_name
        for path in self._search_paths:
            candidate_path = os.path.join(path, xml_name)
            if os.path.exists(candidate_path):
                return candidate_path
        return None

    def load_tree(self, tree_file_name: str):
        """
        Loads and sets up a behavior tree, making it the current behavior.

        Raises:
            RuntimeError: If the tree cannot be found or parsed.
        """
        if not tree_file_name:
            raise RuntimeError("Behavior tree file name cannot be empty!")

        xml_path = self.find_tree(tree_file_name)
        if xml_path is None:
            raise RuntimeError(f"Could not find tree: {tree_file_name}")

        try:
            root = parse_behaviour_tree_xml(xml_path, search_paths=self._search_paths)
        except Exception as e:
            raise RuntimeError(f"Failed to parse XML file: {e}") from e

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
        self._current_behavior = tree_file_name

    def tick(self) -> Status:
        """Ticks the current behavior tree once and returns its resulting status."""
        if self._tree is None:
            raise RuntimeError("No behavior tree loaded.")

        t_start = self._clock.now()
        self._tree.tick()
        t_elapsed = self._clock.now() - t_start

        if t_elapsed > self._tick_period:
            self._logger.warning(f"Overrunning behavior tree tick rate of {self._dt} s")
        return self._tree.root.status

    def request_cancel(self):
        """
        Requests cancellation of the current behavior tree run.

        This may be called from any thread; the run loop acknowledges it on its next tick.
        """
        self._cancel_requested.set()

    def run_tree(self, tree_file_name: str) -> Status:
        """
        Loads a behavior tree and runs it to completion or cancellation.

        Args:
            tree_file_name: The name (resolved through the search paths) or path of the tree XML file.

        Returns:
            The final tree status: SUCCESS or FAILURE, or INVALID if the run was canceled.

        Raises:
            RuntimeError: If the tree cannot be loaded.
        """
        self.load_tree(tree_file_name)
        self._logger.info(f"Running behavior: {tree_file_name}")

        # Drop any cancellation that was requested before this run started.
        self._cancel_requested.clear()

        while True:
            if self._cancel_requested.is_set():
                self._cancel_requested.clear()
                self._logger.info("Behavior canceled.")
                self.stop_tree()
                return Status.INVALID

            if self.tick() in (Status.SUCCESS, Status.FAILURE):
                break
            self._clock.sleep_for(self._tick_period)

        final_status = self.status
        self._logger.info(f"Behavior completed with status: {final_status}")
        self.stop_tree()
        return final_status

    def stop_tree(self):
        """Stops the currently running behavior tree and resets its state."""
        if self._current_behavior is not None:
            self._logger.info("Stopping current behavior...")
            self._current_behavior = None

        if self._tree is not None:
            self._tree.root.status = Status.INVALID
            self._tree.shutdown(destroy_node=False)

        # Wipe per-run state, otherwise sequential runs might pick up blackboard items
        # from the previous tree.
        self._init_global_blackboard()


class BehaviorTreeServer:
    """
    Wraps a behavior tree executor with a ROS action server for starting and stopping behaviors.
    """

    def __init__(self, node: Node, executor: BehaviorTreeExecutor | None = None):
        """
        Initializes the action server.

        Args:
            node: The ROS node associated with this action server.
            executor: Optional behavior tree executor to wrap. If None, one is created.
        """
        self._node = node
        self._logger = node.get_logger()
        self._executor = executor if executor is not None else BehaviorTreeExecutor(node)
        self._requested_behavior = None

        self._action_server = ActionServer(
            node,
            ExecuteBehavior,
            "execute_behavior",
            goal_callback=self.goal_cb,
            execute_callback=self.execute_cb,
            cancel_callback=self.cancel_cb,
            callback_group=ReentrantCallbackGroup(),
        )
        self._logger.info("Behavior tree executor ready!")

    def goal_cb(self, goal_request: ExecuteBehavior.Goal) -> GoalResponse:
        """Accepts a new goal only if no other behavior has been requested."""
        if self._requested_behavior is not None:
            self._logger.error(f"Already running behavior: {self._requested_behavior}")
            return GoalResponse.REJECT

        self._requested_behavior = goal_request.tree_file_name
        return GoalResponse.ACCEPT

    def cancel_cb(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        """Cancellation callback for ROS action server."""
        self._executor.request_cancel()
        return CancelResponse.ACCEPT

    def execute_cb(self, goal_handle: ServerGoalHandle) -> ExecuteBehavior.Result:
        """Runs the requested behavior tree to completion, or until it is canceled."""
        result = ExecuteBehavior.Result()
        try:
            final_status = self._executor.run_tree(goal_handle.request.tree_file_name)
        except (NoDataAvailable, RuntimeError) as e:
            result.message = str(e)
            self._logger.error(result.message)
            goal_handle.abort()
            return result
        finally:
            self._requested_behavior = None

        if final_status == Status.INVALID:
            result.message = "Behavior canceled."
            goal_handle.canceled()
        else:
            result.success = final_status == Status.SUCCESS
            goal_handle.succeed()
        return result
