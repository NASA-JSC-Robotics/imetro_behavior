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

import yaml

from ament_index_python.packages import get_package_share_path
from rclpy.node import Node

from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts, PortInformation


class JointNamesAndPositionsFromYaml(BehaviourWithPorts):
    """Loads joint names and positions from a YAML file.

    Valid yaml configuration for a joint state:

    state_name:
        joint_names:
            - joint_1
            - joint_2
            - joint_3
        positions:
            - 0.1
            - 0.2
            - 0.3

    No validation is done for the respective lengths of the names and positions here,
    as these may not match up exactly in the event of multi-DOF joints.
    Downstream consumers should do their own validation based on what they support.
    """

    INPUT_PORTS = {
        "package_name": PortInformation(data_type=str, required=True),
        "yaml_file": PortInformation(data_type=str, required=True),
        "state_name": PortInformation(data_type=str, required=True),
    }

    OUTPUT_PORTS = {
        "joint_names": PortInformation(data_type=list[str], required=True),
        "joint_positions": PortInformation(data_type=list[float], required=True),
    }

    def setup(self, **kwargs):
        """Get access to the node for error statements."""
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

    def update(self) -> Status:
        """Load the YAML file and set the joint name and positions as an output port."""

        yaml_path = get_package_share_path(self.get_input("package_name")) / self.get_input("yaml_file")
        if not yaml_path.is_file():
            self.node.get_logger().error(f"File at {yaml_path} could not be found or is not a file")
            return Status.FAILURE

        with open(yaml_path) as file:
            data = yaml.safe_load(file)
        state_name = self.get_input("state_name")
        joint_dict = data.get(state_name)
        if joint_dict is None:
            self.node.get_logger().error(f"Failed to find joint configuration {state_name} in {yaml_path}")
            return Status.FAILURE

        joint_names = joint_dict.get("joint_names")
        if joint_names is None:
            self.node.get_logger().error(f"Joint configuration '{state_name}' does not have a 'joint_names' field.")
            return Status.FAILURE

        joint_positions = joint_dict.get("positions")
        if joint_positions is None:
            self.node.get_logger().error(f"Joint configuration '{state_name}' does not have a 'positions' field.")
            return Status.FAILURE

        self._set_output("joint_names", joint_names)
        self._set_output("joint_positions", joint_positions)
        return Status.SUCCESS
