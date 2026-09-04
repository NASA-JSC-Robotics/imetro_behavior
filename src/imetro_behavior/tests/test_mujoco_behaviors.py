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


import pytest
from py_trees.blackboard import Blackboard
from rclpy.node import Node


@pytest.fixture()
def mujoco_behaviors():
    """Import the mujoco behaviors module, or a test skip if its dependencies are unavailable."""
    return pytest.importorskip(
        "imetro_behavior.mujoco_behaviors",
        reason="requires mujoco_ros2_control_msgs",
    )


@pytest.fixture()
def reset_behavior(mujoco_behaviors, ros_node: Node):
    behavior = mujoco_behaviors.ResetMujocoWorld(name="reset", service_name="/reset_world")
    behavior.setup(node=ros_node)
    behavior.setup_ports()
    return behavior


def test_reset_mujoco_world_default_keyframe(mujoco_behaviors, reset_behavior) -> None:
    """An empty keyframe should produce a request with an empty string."""
    request = reset_behavior.create_request()
    assert isinstance(request, mujoco_behaviors.ResetWorld.Request)
    assert request.keyframe == ""


def test_reset_mujoco_world_named_keyframe(reset_behavior) -> None:
    """A named keyframe should be forwarded to the request."""
    keyframe_key = reset_behavior._get_blackboard_key("keyframe")
    Blackboard.set(keyframe_key, "bench_open")

    request = reset_behavior.create_request()
    assert request.keyframe == "bench_open"
