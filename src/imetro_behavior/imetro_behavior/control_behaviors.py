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

from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParametersAtomically

from control_msgs.action import GripperCommand
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers, SwitchController

from imetro_behavior.ros_behaviors.action_client import RosActionClientBase
from imetro_behavior.ros_behaviors.service_client import RosServiceClientBase


class GetRosControllerInfo(RosServiceClientBase):
    """
    Requests controller information from a ros2_control controller_manager node
    and writes it to the blackboard.

    Tip: If you're not spawning new controllers during behavior execution, you can
    effectively just call this behavior once at the beginning and use its output on
    the blackboard to switch controllers.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=ListControllers, **kwargs)

    INPUT_PORTS = {}

    OUTPUT_PORTS = {"controller_info": PortInformation(data_type=list[ControllerState])}

    def create_request(self) -> ListControllers.Request:
        """Create a list controllers request (no fields necessary)."""
        return ListControllers.Request()

    def process_response(self, response: ListControllers.Response) -> Status:
        """Process the service response."""
        self._set_output("controller_info", response.controller)
        return Status.SUCCESS


class SwitchRosControllers(RosServiceClientBase):
    """
    Switches controllers using the ros2_control controller manager node's service.

    If you pass in a `controller_info` input, the behavior will automatically deactivate any
    additional controllers that conflict with your `activate_controllers` list. These will
    be added on to any `deactivate_controllers` inputs that you pass in explicitly.

    If you don't need the logic above, you can simply omit the `controller_info` input and
    pass in explicit activate/deactivate controller lists.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=SwitchController, **kwargs)

    INPUT_PORTS = {
        "controller_info": PortInformation(data_type=list[ControllerState], required=False),
        "activate_controllers": PortInformation(data_type=list[str], required=False),
        "deactivate_controllers": PortInformation(data_type=list[str], required=False),
    }

    OUTPUT_PORTS = {}

    def create_request(self) -> SwitchController.Request:
        """
        Look up which controllers must be activated and/or deactivated based on the inputs,
        and then package up a corresponding switch controller request."""
        controller_info = self.get_input("controller_info", [])
        activate_controllers = self.get_input("activate_controllers", [])
        deactivate_controllers = self.get_input("deactivate_controllers", [])

        if len(controller_info) == 0 or len(activate_controllers) == 0:
            # Simplest case: if there are no controllers to activate or no controller info,
            # we can directly use the inputs on the blackboard without any manipulation.
            pass
        else:
            # Otherwise, we may need to add more controllers to deactivate based on
            # conflicting hardware interface requirements.

            # First figure out all the necessary interfaces that must be claimed by the
            # incoming controllers to activate.
            interfaces_to_claim = set()
            for info in controller_info:
                if info.name in activate_controllers:
                    interfaces_to_claim.update(info.required_command_interfaces)

            # Next, look at all the currently active controllers. If any of their claimed
            # interfaces conflict with the list above, we must add it to the deactivate list.
            for info in controller_info:
                if info.state != "active":
                    continue

                conflicting_interfaces = set(info.claimed_interfaces).intersection(interfaces_to_claim)
                if len(conflicting_interfaces) > 0:
                    self.node.get_logger().info(
                        f"Adding controller '{info.name}' to deactivate list since it claims "
                        f"the following command interfaces: {conflicting_interfaces}."
                    )
                    deactivate_controllers.append(info.name)

        return SwitchController.Request(
            activate_controllers=activate_controllers,
            deactivate_controllers=deactivate_controllers,
        )

    def process_response(self, response: SwitchController.Response) -> Status:
        """Process the service response."""
        if response.ok:
            self.node.get_logger().debug("Successfully switched controllers!")
            return Status.SUCCESS
        else:
            self.node.get_logger().error(f"Failed to switch controllers: {response.message}")
            return Status.FAILURE


class CommandGripper(RosActionClientBase):
    """Sends an action goal to a ROS 2 gripper command controller."""

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, action_type=GripperCommand, **kwargs)

    INPUT_PORTS = {
        "position": PortInformation(data_type=float, required=True),
        "max_effort": PortInformation(data_type=float, required=False),
    }

    OUTPUT_PORTS = {}

    def create_goal(self) -> GripperCommand.Goal:
        """Create a gripper command goal."""
        goal = GripperCommand.Goal()
        goal.command.position = self.get_input("position")
        goal.command.max_effort = self.get_input("max_effort", 0.0)
        return goal

    def process_result(self, result: GripperCommand.Result) -> Status:
        """Process the gripper command action result."""
        if result.reached_goal or result.stalled:
            self.node.get_logger().debug("Successfully commanded gripper!")
            return Status.SUCCESS
        else:
            self.node.get_logger().error("Gripper command action did not reach its goal or hit a stall condition.")
            return Status.FAILURE


class UpdateAdmittanceParameters(RosServiceClientBase):
    """
    Updates admittance parameters using the SetParametersAtomically service.

    All of the parameters are optional, and only the populated ones will be applied to the controller.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=SetParametersAtomically, **kwargs)

    INPUT_PORTS = {
        "admittance_selected_axes": PortInformation(
            data_type=list[bool],
            required=False,
            default_value=[],
            description="Which axes to enable for admittance (tx, ty, tz, rx, ry, rz). Must be size 6",
        ),
        "admittance_mass": PortInformation(
            data_type=list[float],
            required=False,
            default_value=[],
            description="Mass for each axis for admittance (tx, ty, tz, rx, ry, rz). Must be size 6",
        ),
        "admittance_stiffness": PortInformation(
            data_type=list[float],
            required=False,
            default_value=[],
            description="Stiffness for each axis for admittance (tx, ty, tz, rx, ry, rz). Must be size 6",
        ),
        "admittance_damping_ratio": PortInformation(
            data_type=list[float],
            required=False,
            default_value=[],
            description="Damping ratio for each axis for admittance (tx, ty, tz, rx, ry, rz). Must be size 6",
        ),
        "gravity_compensation_CoG_force": PortInformation(
            data_type=float,
            required=False,
            default_value=None,
            description="Force of gravity to be compensated for the force torque data",
        ),
        "gravity_compensation_CoG_pos": PortInformation(
            data_type=list[float],
            required=False,
            default_value=[],
            description="Center of Gravity location w.r.t. the predefined frame (px, py, pz). Must be size 3",
        ),
    }

    OUTPUT_PORTS = {}

    def add_parameter_to_list(self, params_list, param_name, param_type, param_value) -> bool:

        # return early if param_value is None or is empty list
        if not param_value:
            return

        parameter = Parameter(name=param_name, value=ParameterValue(type=param_type))

        if param_type == ParameterType.PARAMETER_BOOL_ARRAY:
            parameter.value.bool_array_value = param_value
        elif param_type == ParameterType.PARAMETER_DOUBLE_ARRAY:
            parameter.value.double_array_value = param_value
        elif param_type == ParameterType.PARAMETER_DOUBLE:
            parameter.value.double_value = param_value
        else:
            self.node.get_logger().error(
                "Only 'PARAMETER_DOUBLE', 'PARAMETER_BOOL_ARRAY', and 'PARAMETER_DOUBLE_ARRAY' are supported for the add_parameter_to_list() method."
            )
            return False

        params_list.append(parameter)

        return True

    def create_request(self) -> SetParametersAtomically.Request:
        """
        Package up a SetParametersAtomically request for the admittance parameters we want to update."""
        admittance_selected_axes = self.get_input("admittance_selected_axes")
        admittance_mass = self.get_input("admittance_mass")
        admittance_stiffness = self.get_input("admittance_stiffness")
        admittance_damping_ratio = self.get_input("admittance_damping_ratio")
        gravity_compensation_CoG_force = self.get_input("gravity_compensation_CoG_force")
        gravity_compensation_CoG_pos = self.get_input("gravity_compensation_CoG_pos")

        request = SetParametersAtomically.Request
        params_list = []

        success = True
        success = success and self.add_parameter_to_list(
            params_list=params_list,
            param_name="admittance.selected_axes",
            param_type=ParameterType.PARAMETER_BOOL_ARRAY,
            param_value=admittance_selected_axes,
        )
        success = success and self.add_parameter_to_list(
            params_list=params_list,
            param_name="admittance.mass",
            param_type=ParameterType.PARAMETER_DOUBLE_ARRAY,
            param_value=admittance_mass,
        )
        success = success and self.add_parameter_to_list(
            params_list=params_list,
            param_name="admittance.stiffness",
            param_type=ParameterType.PARAMETER_DOUBLE_ARRAY,
            param_value=admittance_stiffness,
        )
        success = success and self.add_parameter_to_list(
            params_list=params_list,
            param_name="admittance.damping_ratio",
            param_type=ParameterType.PARAMETER_DOUBLE_ARRAY,
            param_value=admittance_damping_ratio,
        )
        success = success and self.add_parameter_to_list(
            params_list=params_list,
            param_name="gravity_compensation.CoG.force",
            param_type=ParameterType.PARAMETER_DOUBLE,
            param_value=gravity_compensation_CoG_force,
        )
        success = success and self.add_parameter_to_list(
            params_list=params_list,
            param_name="gravity_compensation.CoG.pos",
            param_type=ParameterType.PARAMETER_DOUBLE_ARRAY,
            param_value=gravity_compensation_CoG_pos,
        )
        if not success:
            raise RuntimeError("Setting parameters inside UpdateAdmittanceParameters did not work.")

        return request

    def process_response(self, response: SetParametersAtomically.Response) -> Status:
        """Process the service response."""
        if response.result.successful:
            self.node.get_logger().debug("Successfully set admittance parameters!")
            return Status.SUCCESS
        else:
            self.node.get_logger().error(f"Failed to set parameters: {response.result.reason}")
            return Status.FAILURE
