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

from mockup_msgs.srv import SetJointState
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rclpy.node import Node

from imetro_behavior.mockup_behaviors import UpdateMockupStates


def test_update_mockup_states_create_request_lists(ros_node: Node) -> None:
    behavior = UpdateMockupStates(name="update_mockup_states", service_name="/foo")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    Blackboard.set(behavior._get_blackboard_key("joint_names"), ["joint1", "joint2"])
    Blackboard.set(behavior._get_blackboard_key("joint_positions"), [1.0, 2.0])
    Blackboard.set(behavior._get_blackboard_key("joint_velocities"), [3.0, 4.0])
    Blackboard.set(behavior._get_blackboard_key("joint_efforts"), [5.0, 6.0])

    request = behavior.create_request()
    assert request.joint_state.name == ["joint1", "joint2"]
    assert request.joint_state.position.tolist() == [1.0, 2.0]
    assert request.joint_state.velocity.tolist() == [3.0, 4.0]
    assert request.joint_state.effort.tolist() == [5.0, 6.0]


def test_update_mockup_states_create_request_scalars(ros_node: Node) -> None:
    behavior = UpdateMockupStates(name="update_mockup_states", service_name="/foo")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    # Single-joint calls may provide scalars instead of single-element lists.
    Blackboard.set(behavior._get_blackboard_key("joint_names"), "joint1")
    Blackboard.set(behavior._get_blackboard_key("joint_positions"), 1.0)
    Blackboard.set(behavior._get_blackboard_key("joint_velocities"), 3.0)
    Blackboard.set(behavior._get_blackboard_key("joint_efforts"), 5.0)

    request = behavior.create_request()
    assert request.joint_state.name == ["joint1"]
    assert request.joint_state.position.tolist() == [1.0]
    assert request.joint_state.velocity.tolist() == [3.0]
    assert request.joint_state.effort.tolist() == [5.0]


def test_update_mockup_states_create_request_defaults(ros_node: Node) -> None:
    behavior = UpdateMockupStates(name="update_mockup_states", service_name="/foo")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    Blackboard.set(behavior._get_blackboard_key("joint_names"), ["joint1"])

    request = behavior.create_request()
    assert request.joint_state.name == ["joint1"]
    assert request.joint_state.position.tolist() == []
    assert request.joint_state.velocity.tolist() == []
    assert request.joint_state.effort.tolist() == []


def test_update_mockup_states_process_response(ros_node: Node) -> None:
    behavior = UpdateMockupStates(name="update_mockup_states", service_name="/foo")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    assert behavior.process_response(SetJointState.Response(success=True)) == Status.SUCCESS
    assert behavior.process_response(SetJointState.Response(success=False, message="oops")) == Status.FAILURE
