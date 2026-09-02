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
from mockup_msgs.srv import SetJointState
from py_trees.blackboard import Blackboard
from py_trees.common import Status

from imetro_behavior.mockup_behaviors import UpdateMockupStates


def test_update_mockup_states_create_request(ros_node: Node) -> None:
    behavior = UpdateMockupStates(name="update_mockups", service_name="/mockup_manager/set_joint_state")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    Blackboard.set(behavior._get_blackboard_key("mockup_joint_names"), ["door"])
    Blackboard.set(behavior._get_blackboard_key("mockup_joint_positions"), [0.78])

    request = behavior.create_request()
    assert request.joint_state.name == ["door"]
    assert list(request.joint_state.position) == [0.78]
    assert list(request.joint_state.velocity) == []
    assert list(request.joint_state.effort) == []

    Blackboard.set(behavior._get_blackboard_key("mockup_joint_names"), ["bench1", "bench2"])
    Blackboard.set(behavior._get_blackboard_key("mockup_joint_positions"), [0.0, 1.57])
    Blackboard.set(behavior._get_blackboard_key("mockup_joint_velocities"), [0.0, 0.0])
    Blackboard.set(behavior._get_blackboard_key("mockup_joint_efforts"), [0.0, 0.0])

    request = behavior.create_request()
    assert request.joint_state.name == ["bench1", "bench2"]
    assert list(request.joint_state.position) == [0.0, 1.57]
    assert list(request.joint_state.velocity) == [0.0, 0.0]
    assert list(request.joint_state.effort) == [0.0, 0.0]


def test_update_mockup_states_process_response(ros_node: Node) -> None:
    behavior = UpdateMockupStates(name="update_mockups", service_name="/mockup_manager/set_joint_state")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    response = SetJointState.Response()
    response.success = True
    assert behavior.process_response(response) == Status.SUCCESS

    response.success = False
    assert behavior.process_response(response) == Status.FAILURE
