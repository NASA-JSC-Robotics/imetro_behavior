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

from rclpy.duration import Duration
from rclpy.node import Node

from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts


class WaitForDuration(BehaviourWithPorts):
    """Waits for a specified duration using the ROS clock."""

    def __init__(self, name: str, duration_sec: float, **kwargs):
        """
        Constructs behavior to wait for a duration.

        Args:
            name: The name of the behavior (required by PyTrees)
            duration_sec: Duration, in seconds, to wait for before completion.
            kwargs: Additional keyword arguments to pass through to ports.
        """
        self.duration = Duration(seconds=duration_sec)
        super().__init__(name, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {}

    def setup(self, **kwargs):
        """Get access to the ROS node for its clock."""
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

    def initialise(self):
        """Initializes the start time against which to compare the duration."""
        self.start_time = self.node.get_clock().now()

    def update(self) -> Status:
        """Return running until the duration is complete."""
        if self.node.get_clock().now() - self.start_time >= self.duration:
            return Status.SUCCESS
        return Status.RUNNING
