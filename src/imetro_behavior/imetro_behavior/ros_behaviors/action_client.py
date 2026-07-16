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

import time
from typing import Any, Type

from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.task import Future

from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts


class RosActionClientBase(BehaviourWithPorts):
    """
    Base class for behaviors that rely on ROS action clients.

    Modified from https://github.com/splintered-reality/py_trees_ros/blob/devel/py_trees_ros/action_clients.py
    """

    def __init__(
        self,
        name: str,
        action_type: Type,
        *,
        action_name: str,
        action_server_timeout: float = 3.0,
        action_timeout: float | None = None,
        **kwargs: Any,
    ):
        """
        Constructs a ROS Action Client base behavior.

        Args:
            name: The name of the behavior (required by PyTrees)
            action_type: The ROS interface type of the action.
            action_name: The name of the ROS action to send a goal to.
            action_server_timeout: Timeout, in seconds, to wait for the action server to be available.
                If None, waits indefinitely.
            action_timeout: Timeout, in seconds, to wait for the action to complete.
                If None, waits indefinitely.
            kwargs: Additional keyword arguments to pass through to ports.
        """
        self.action_type = action_type
        self.action_name = action_name
        self.action_server_timeout = Duration(seconds=action_server_timeout) if action_server_timeout else None
        self.action_timeout = Duration(seconds=action_timeout) if action_timeout else None
        super().__init__(name, **kwargs)

    def create_goal(self) -> Any | None:
        """
        Abstract method for creating and returning a ROS action goal.

        You can return None to consider this a failure case.
        """
        raise NotImplementedError("Must implement create_goal() method.")

    def process_result(self, result: Any) -> Status:
        """
        Abstract method for processing a ROS action result.
        """
        raise NotImplementedError("Must implement process_result() method.")

    def setup(self, **kwargs):
        """
        Sets up the action client.
        """
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

        self.action_client = ActionClient(
            node=self.node,
            action_type=self.action_type,
            action_name=self.action_name,
        )
        self.client_ready = False

        self.goal_handle = None
        self.get_result_future = None
        self.action_start_time = None
        self.client_start_time = None

    def initialise(self):
        """
        Reset the internal variables.
        """
        self.goal_handle = None
        self.send_goal_future = None
        self.action_start_time = None
        self.client_start_time = self.node.get_clock().now()
        self.client_ready = self.action_client.server_is_ready()

    def update(self):
        """
        Kick off a new goal request and then check whether the action has completed or timed out.
        """
        if not self.client_ready:
            # Wait for the action server to be available until there is a timeout.
            if (
                self.action_server_timeout is not None
                and self.node.get_clock().now() - self.client_start_time > self.action_server_timeout
            ):
                self.node.get_logger().error(f"Timed out waiting for action server {self.action_name}.")
                return Status.FAILURE
            else:
                self.client_ready = self.action_client.server_is_ready()
                return Status.RUNNING

        elif self.send_goal_future is None:
            # Send a goal request if one hasn't yet been sent.
            try:
                goal = self.create_goal()  # Must be implemented
            except Exception as e:
                self.node.get_logger().error(f"Failed to create action goal: {e}")
                return Status.FAILURE

            self.node.get_logger().debug("Sending action goal...")
            self.send_goal_future = self.action_client.send_goal_async(goal)
            self.send_goal_future.add_done_callback(self.goal_response_callback)
            self.action_start_time = self.node.get_clock().now()
            return Status.RUNNING

        elif self.goal_handle is not None and not self.goal_handle.accepted:
            # Fail if the goal was not accepted.
            self.node.get_logger().error("Goal rejected.")
            self.goal_handle = None
            return Status.FAILURE

        elif self.get_result_future is not None and self.get_result_future.done():
            # If the action completed, process the result and return the corresponding status.
            try:
                # Must be implemented
                self.goal_handle = None
                return self.process_result(self.get_result_future.result())
            except Exception as e:
                self.node.get_logger().error(f"Failed to process action result: {e}")
                return Status.FAILURE

        elif (
            self.action_timeout is not None
            and self.node.get_clock().now() - self.action_start_time > self.action_timeout
        ):
            # If we made it here, the action is in progress. Check for timeouts.
            self.node.get_logger().error("Action timed out.")
            return Status.FAILURE

        else:
            # Action is in progress but not yet timed out.
            return Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        """
        If running and the current goal has not already completed, cancel it.
        """
        if self.status == Status.RUNNING and new_status == Status.INVALID:
            self.send_cancel_request()

    def shutdown(self):
        """
        Clean up the action client when shutting down.
        """
        self.send_cancel_request()  # For good measure
        if self.goal_handle is not None:
            # Give time for the cancel request to clear before destroying the client.
            time.sleep(0.1)
        self.goal_handle = None
        self.action_client.destroy()

    def goal_response_callback(self, future: Future) -> None:
        """
        Handle goal response and proceed to listen for the result if accepted.
        """
        if future.result() is None:
            self.node.get_logger().error("Goal request failed.")
            return
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.node.get_logger().error("Goal request rejected")
            return
        else:
            self.node.get_logger().debug("Goal request accepted.")
            pass

        self.get_result_future = self.goal_handle.get_result_async()

    def send_cancel_request(self):
        """
        Send a cancel request to the server.
        """
        if self.goal_handle is not None:
            self.node.get_logger().debug("Canceling goal.")
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_response_callback)

    def cancel_response_callback(self, future: Future) -> None:
        """
        Unsets the goal handle when the cancel response is processed.
        """
        self.goal_handle = None
