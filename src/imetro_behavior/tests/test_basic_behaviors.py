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

from imetro_behavior.basic_behaviors import WaitForDuration


def test_wait_for_duration_zero_duration(ros_node: Node) -> None:
    behavior = WaitForDuration(name="wait", duration_sec=0.0)
    behavior.setup(node=ros_node)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS


def test_wait_for_duration(ros_node: Node) -> None:
    behavior = WaitForDuration(name="wait", duration_sec=0.1)
    behavior.setup(node=ros_node)

    behavior.tick_once()
    assert behavior.status == Status.RUNNING

    # Sleep on the same ROS clock the behavior uses, rather than wall time.
    ros_node.get_clock().sleep_for(Duration(seconds=0.11))

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
