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
from control_msgs.action import GripperCommand
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers, SwitchController
from imetro_behavior.control_behaviors import (
    CommandGripper,
    GetRosControllerInfo,
    SwitchRosControllers,
    UpdateAdmittanceParameters,
)
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rcl_interfaces.msg import ParameterType, SetParametersResult
from rcl_interfaces.srv import SetParametersAtomically
from rclpy.node import Node


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


def test_switch_ros_controllers_explicit(ros_node: Node) -> None:
    behavior = SwitchRosControllers(name="switch_controllers_explicit", service_name="/foo")
    behavior.setup(node=ros_node)
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


def test_switch_ros_controllers_process_response(ros_node: Node) -> None:
    behavior = SwitchRosControllers(name="switch_controllers", service_name="/foo")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    assert behavior.process_response(SwitchController.Response(ok=True)) == Status.SUCCESS
    assert behavior.process_response(SwitchController.Response(ok=False, message="oops")) == Status.FAILURE


def test_get_ros_controller_info(ros_node: Node, sample_controller_info: list[ControllerState]) -> None:
    behavior = GetRosControllerInfo(name="get_controller_info", service_name="/foo")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    # The request has no fields to fill in.
    assert isinstance(behavior.create_request(), ListControllers.Request)

    response = ListControllers.Response(controller=sample_controller_info)
    assert behavior.process_response(response) == Status.SUCCESS
    assert behavior.get_last_output("controller_info") == sample_controller_info


def test_command_gripper_create_goal(ros_node: Node) -> None:
    behavior = CommandGripper(name="command_gripper", action_name="/gripper_command")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    position_port = behavior._get_blackboard_key("position")
    Blackboard.set(position_port, 0.025)

    # Max effort should default to zero when not specified.
    goal = behavior.create_goal()
    assert goal.command.position == 0.025
    assert goal.command.max_effort == 0.0

    max_effort_port = behavior._get_blackboard_key("max_effort")
    Blackboard.set(max_effort_port, 10.0)

    goal = behavior.create_goal()
    assert goal.command.position == 0.025
    assert goal.command.max_effort == 10.0


def test_command_gripper_process_result(ros_node: Node) -> None:
    behavior = CommandGripper(name="command_gripper", action_name="/gripper_command")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    assert behavior.process_result(GripperCommand.Result(reached_goal=True)) == Status.SUCCESS
    # Stalling is considered success, since it can mean the gripper closed on an object.
    assert behavior.process_result(GripperCommand.Result(stalled=True)) == Status.SUCCESS
    assert behavior.process_result(GripperCommand.Result()) == Status.FAILURE


def test_update_admittance_parameters_create_goal_full(ros_node: Node) -> None:
    behavior = UpdateAdmittanceParameters(name="update_admittance_parameters", service_name="/foo")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    example_admittance_selected_axes = [True, True, True, False, False, False]
    example_admittance_mass = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    example_admittance_stiffness = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    example_admittance_damping_ratio = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    example_gravity_compensation_CoG_force = 9.0
    example_gravity_compensation_CoG_pos = [4.0, 5.0, 6.0]

    admittance_selected_axes_port = behavior._get_blackboard_key("admittance_selected_axes")
    Blackboard.set(admittance_selected_axes_port, example_admittance_selected_axes)
    admittance_mass_port = behavior._get_blackboard_key("admittance_mass")
    Blackboard.set(admittance_mass_port, example_admittance_mass)
    admittance_stiffness_port = behavior._get_blackboard_key("admittance_stiffness")
    Blackboard.set(admittance_stiffness_port, example_admittance_stiffness)
    admittance_damping_ratio_port = behavior._get_blackboard_key("admittance_damping_ratio")
    Blackboard.set(admittance_damping_ratio_port, example_admittance_damping_ratio)
    gravity_compensation_CoG_force_port = behavior._get_blackboard_key("gravity_compensation_CoG_force")
    Blackboard.set(gravity_compensation_CoG_force_port, example_gravity_compensation_CoG_force)
    gravity_compensation_CoG_pos_port = behavior._get_blackboard_key("gravity_compensation_CoG_pos")
    Blackboard.set(gravity_compensation_CoG_pos_port, example_gravity_compensation_CoG_pos)

    request = behavior.create_request()
    # check that the right amount of data got added
    assert len(request.parameters) == 6

    # Check the data in the request
    # These end up in the order as follows
    # 0. selected_axes
    # 1. mass
    # 2. stiffness
    # 3. damping_ratio
    # 4. cog_force
    # 5. cog_pos
    assert request.parameters[0].value.bool_array_value == example_admittance_selected_axes
    assert request.parameters[0].value.type == ParameterType.PARAMETER_BOOL_ARRAY
    assert request.parameters[1].value.double_array_value.tolist() == example_admittance_mass
    assert request.parameters[1].value.type == ParameterType.PARAMETER_DOUBLE_ARRAY
    assert request.parameters[2].value.double_array_value.tolist() == example_admittance_stiffness
    assert request.parameters[2].value.type == ParameterType.PARAMETER_DOUBLE_ARRAY
    assert request.parameters[3].value.double_array_value.tolist() == example_admittance_damping_ratio
    assert request.parameters[3].value.type == ParameterType.PARAMETER_DOUBLE_ARRAY
    assert request.parameters[4].value.double_value == example_gravity_compensation_CoG_force
    assert request.parameters[4].value.type == ParameterType.PARAMETER_DOUBLE
    assert request.parameters[5].value.double_array_value.tolist() == example_gravity_compensation_CoG_pos
    assert request.parameters[5].value.type == ParameterType.PARAMETER_DOUBLE_ARRAY


def test_update_admittance_parameters_process_result(ros_node: Node) -> None:
    behavior = UpdateAdmittanceParameters(name="update_admittance_parameters", service_name="/foo")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    assert (
        behavior.process_response(SetParametersAtomically.Response(result=SetParametersResult(successful=True)))
        == Status.SUCCESS
    )
    assert (
        behavior.process_response(
            SetParametersAtomically.Response(result=SetParametersResult(successful=False, reason="oops"))
        )
        == Status.FAILURE
    )
