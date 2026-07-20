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

import pytest

import rclpy
from rclpy.node import Node
from controller_manager_msgs.msg import ControllerState
from py_trees.blackboard import Blackboard

from imetro_behavior.control_behaviors import SwitchRosControllers


@pytest.fixture()
def ros_node():
    # Setup
    rclpy.init()
    node = Node("test_control_behaviors_node")

    yield node

    # Teardown
    node.destroy_node()
    rclpy.try_shutdown()


@pytest.fixture()
def sample_controller_info() -> list[ControllerState]:
    return [
        ControllerState(
            name="arm_controller",
            state="active",
            claimed_interfaces=["joint1", "joint2", "joint3"],
            required_command_interfaces=["joint1", "joint2", "joint3"],
        ),
        ControllerState(
            name="lift_controller",
            state="active",
            claimed_interfaces=["lift_joint"],
            required_command_interfaces=["lift_joint"],
        ),
        ControllerState(
            name="full_robot_controller",
            state="inactive",
            claimed_interfaces=[],
            required_command_interfaces=["lift_joint", "joint1", "joint2", "joint3"],
        ),
        ControllerState(
            name="gripper_controller",
            state="active",
            claimed_interfaces=["gripper_joint"],
            required_command_interfaces=["gripper_joint"],
        ),
    ]


def test_switch_ros_controllers_explicit() -> None:
    behavior = SwitchRosControllers(name="switch_controllers_explicit", service_name="/foo")
    behavior.setup_ports()

    activate_controllers_port = behavior._get_blackboard_key("activate_controllers")
    Blackboard.set(activate_controllers_port, ["arm_controller", "lift_controller"])

    deactivate_controllers_port = behavior._get_blackboard_key("deactivate_controllers")
    Blackboard.set(deactivate_controllers_port, ["full_robot_controller"])

    request = behavior.create_request()
    assert request.activate_controllers == ["arm_controller", "lift_controller"]
    assert request.deactivate_controllers == ["full_robot_controller"]


def test_switch_ros_controllers_using_info(ros_node: Node, sample_controller_info: list[ControllerState]) -> None:
    behavior = SwitchRosControllers(name="switch_controllers_from_info", service_name="/foo")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    controller_info_port = behavior._get_blackboard_key("controller_info")
    Blackboard.set(controller_info_port, sample_controller_info)

    activate_controllers_port = behavior._get_blackboard_key("activate_controllers")
    Blackboard.set(activate_controllers_port, ["full_robot_controller"])

    request = behavior.create_request()
    assert request.activate_controllers == ["full_robot_controller"]
    assert request.deactivate_controllers == ["arm_controller", "lift_controller"]
