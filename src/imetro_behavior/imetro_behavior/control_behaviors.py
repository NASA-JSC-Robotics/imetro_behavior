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

from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers, SwitchController

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

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"controller_info": PortInformation(data_type=list[ControllerState])}

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

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "controller_info": PortInformation(data_type=list[ControllerState], required=False),
            "activate_controllers": PortInformation(data_type=list[str], required=False),
            "deactivate_controllers": PortInformation(data_type=list[str], required=False),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {}

    def create_request(self) -> ListControllers.Request:
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
            interfaces_to_claim = []
            for info in controller_info:
                if info.name in activate_controllers:
                    interfaces_to_claim.extend(info.required_command_interfaces)

            # Next, look at all the currently active controllers. If any of their claimed
            # interfaces conflict with the list above, we must add it to the deactivate list.
            for info in controller_info:
                if info.state == "active" and not set(info.claimed_interfaces).isdisjoint(interfaces_to_claim):
                    deactivate_controllers.append(info.name)

        return SwitchController.Request(
            activate_controllers=activate_controllers,
            deactivate_controllers=deactivate_controllers,
        )

    def process_response(self, response: ListControllers.Response) -> Status:
        """Process the service response."""
        if response.ok:
            self.node.get_logger().debug("Successfully switched controllers!")
            return Status.SUCCESS
        else:
            self.node.get_logger().error(f"Failed to switch controllers: {response.message}")
            return Status.FAILURE
