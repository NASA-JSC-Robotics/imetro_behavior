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

from typing import Any

from py_trees.common import Status
from py_trees.ports import PortInformation

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from imetro_behavior.ros_behaviors.action_client import RosActionClientBase


class NavigateToPoseBehavior(RosActionClientBase):
    """Sends an action goal to Nav2 to navigate to a pose."""

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, action_type=NavigateToPose, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {"goal_pose": PortInformation(data_type=PoseStamped, required=True)}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {}

    def create_goal(self) -> NavigateToPose.Goal:
        """Create a navigation goal."""
        goal_pose = self.get_input("goal_pose")
        goal_pose.header.stamp = self.node.get_clock().now().to_msg()
        return NavigateToPose.Goal(pose=goal_pose)

    def process_result(self, result: NavigateToPose.Result) -> Status:
        """Process the navigation action result."""
        if result.result.error_code == NavigateToPose.Result.NONE:
            self.node.get_logger().info("Navigation action succeeded!")
            return Status.SUCCESS
        else:
            self.node.get_logger().error(f"Navigation action failed with error: {result.result.error_msg}")
            return Status.FAILURE
