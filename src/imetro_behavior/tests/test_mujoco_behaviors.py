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

from rclpy.node import Node
from py_trees.blackboard import Blackboard
from py_trees.common import Status

from mujoco_ros2_control_msgs.srv import ResetWorld
from imetro_behavior.mujoco_behaviors import ResetMujocoWorld


def test_reset_mujoco_world_default_keyframe(ros_node: Node) -> None:
    """An empty keyframe should produce a request with an empty string."""
    behavior = ResetMujocoWorld(name="reset", service_name="/reset_world")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    request = behavior.create_request()
    assert isinstance(request, ResetWorld.Request)
    assert request.keyframe == ""


def test_reset_mujoco_world_named_keyframe(ros_node: Node) -> None:
    """A named keyframe should be forwarded to the request."""
    behavior = ResetMujocoWorld(name="reset", service_name="/reset_world")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    keyframe_key = behavior._get_blackboard_key("keyframe")
    Blackboard.set(keyframe_key, "bench_open")

    request = behavior.create_request()
    assert request.keyframe == "bench_open"
