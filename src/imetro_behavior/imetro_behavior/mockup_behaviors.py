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

from mockup_msgs.srv import SetJointState

from imetro_behavior.ros_behaviors.service_client import RosServiceClientBase


class UpdateMockupStates(RosServiceClientBase):
    """Calls the service to update the mockup joint states.

    Joint names are required; positions, velocities, and efforts are not.
    Any positions, velocities, and efforts provided must be the same size as the provided joint names.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=SetJointState, **kwargs)

    INPUT_PORTS = {
        "joint_names": PortInformation(data_type=list[str], required=True),
        "joint_positions": PortInformation(data_type=list[float], required=False, default_value=[]),
        "joint_velocities": PortInformation(data_type=list[float], required=False, default_value=[]),
        "joint_efforts": PortInformation(data_type=list[float], required=False, default_value=[]),
    }
    OUTPUT_PORTS = {}

    def create_request(self) -> SetJointState.Request:
        """Create a SetJointState service request."""
        mockup_joint_names = self.get_input("joint_names")
        mockup_joint_positions = self.get_input("joint_positions")
        mockup_joint_velocities = self.get_input("joint_velocities")
        mockup_joint_efforts = self.get_input("joint_efforts")

        req = SetJointState.Request()
        req.joint_state.name = mockup_joint_names
        req.joint_state.position = mockup_joint_positions
        req.joint_state.velocity = mockup_joint_velocities
        req.joint_state.effort = mockup_joint_efforts
        return req

    def process_response(self, response: SetJointState.Response) -> Status:
        """Process the SetJointState service response."""
        if response.success:
            self.node.get_logger().info("SetJointState request succeeded!")
            return Status.SUCCESS
        else:
            self.node.get_logger().error(f"SetJointState service failed with error: {response.message}")
            return Status.FAILURE
