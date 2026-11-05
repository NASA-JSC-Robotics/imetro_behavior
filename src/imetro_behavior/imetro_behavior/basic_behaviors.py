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

import operator

from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts, PortInformation
from rclpy.duration import Duration
from rclpy.node import Node

from imetro_behavior.helpers import set_ros_node


class WaitForDuration(BehaviourWithPorts):
    """Waits for a specified duration using the ROS clock."""

    def __init__(self, name: str, duration_sec: float, **kwargs):
        """
        Construct behavior to wait for a duration.

        Args:
            name: The name of the behavior (required by PyTrees)
            duration_sec: Duration, in seconds, to wait for before completion.
            kwargs: Additional keyword arguments to pass through to ports.
        """
        self.node: Node
        self.duration = Duration(seconds=duration_sec)
        super().__init__(name, **kwargs)

    INPUT_PORTS = {}
    OUTPUT_PORTS = {}

    def setup(self, **kwargs):
        """Get access to the ROS node for its clock."""
        set_ros_node(self, **kwargs)

    def initialise(self):
        """Initialize the start time against which to compare the duration."""
        self.start_time = self.node.get_clock().now()

    def update(self) -> Status:
        """Return running until the duration is complete."""
        if self.node.get_clock().now() - self.start_time >= self.duration:
            return Status.SUCCESS
        return Status.RUNNING


class IsSim(BehaviourWithPorts):
    """Check whether the attached ROS node's `use_sim_time` parameter is True."""

    INPUT_PORTS = {}

    OUTPUT_PORTS = {
        "is_sim": PortInformation(
            data_type=bool,
            required=True,
        )
    }

    def setup(self, **kwargs):
        """Get access to the ROS node for its parameters."""
        set_ros_node(self, **kwargs)

    def update(self) -> Status:
        """Write whether `use_sim_time` is True to the output port."""
        if self.node.has_parameter("use_sim_time"):
            use_sim_time = self.node.get_parameter("use_sim_time").get_parameter_value().bool_value
        else:
            use_sim_time = False

        self._set_output("is_sim", use_sim_time)
        return Status.SUCCESS


class BlackboardMath(BehaviourWithPorts):
    """Allow for doing simple arithmetic (+-/*^) using blackboard values."""

    INPUT_PORTS = {
        "operand1": PortInformation(
            data_type=float,
            required=True,
        ),
        "operator": PortInformation(
            data_type=str,
            required=True,
            description="Valid operators: +-/*^",
        ),
        "operand2": PortInformation(
            data_type=float,
            required=True,
        ),
    }

    OUTPUT_PORTS = {
        "result": PortInformation(
            data_type=float,
            required=True,
        )
    }

    OPERATOR_SET = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "^": operator.pow,
    }

    def setup(self, **kwargs):
        """Get access to the ROS node for logger."""
        set_ros_node(self, **kwargs)

    def update(self) -> Status:
        operand1 = self.get_input("operand1")
        op = self.get_input("operator")
        operand2 = self.get_input("operand2")

        if op == "/" and operand2 == 0.0:
            self.logger.error("Error: Division by zero encountered!")
            return Status.FAILURE

        operator_fn = self.OPERATOR_SET.get(op)

        if operator_fn is None:
            self.logger.error(f"Operator: {op} not found in valid list of operators: (+-/*)")
            return Status.FAILURE

        result = operator_fn(operand1, operand2)

        self.logger.debug(f"RESULT: {result}")
        self._set_output("result", result)
        return Status.SUCCESS


class ToList(BehaviourWithPorts):
    """Wrap a scalar blackboard value of any type into a single-element list.

    Useful for feeding a computed value (e.g. from BlackboardMath) into a port
    that expects a list, which cannot be assigned directly from a bare scalar.
    """

    INPUT_PORTS = {
        "value": PortInformation(
            data_type=object,
            required=True,
        ),
    }

    OUTPUT_PORTS = {
        "result": PortInformation(
            data_type=list[object],
            required=True,
        )
    }

    def setup(self, **kwargs):
        """Get access to the ROS node for logger."""
        set_ros_node(self, **kwargs)

    def update(self) -> Status:
        value = self.get_input("value")
        self._set_output("result", [value])
        return Status.SUCCESS


class GetListItem(BehaviourWithPorts):
    """Retrieve a single item out of a blackboard list by index.

    Generic counterpart to a pose/message-specific "get a field" behavior (e.g. GetXYZ):
    useful once a value has already been decomposed into a list (e.g. via
    DecomposePoseStamped's translation_xyz) and a single element of it is needed.
    """

    INPUT_PORTS = {
        "value": PortInformation(
            data_type=list[object],
            required=True,
            description="List to retrieve an item from.",
        ),
        "index": PortInformation(
            data_type=int,
            required=True,
            description="Zero-based index of the item to retrieve.",
        ),
    }

    OUTPUT_PORTS = {
        "result": PortInformation(
            data_type=object,
            required=True,
        )
    }

    def setup(self, **kwargs):
        """Get access to the ROS node for logger."""
        set_ros_node(self, **kwargs)

    def update(self) -> Status:
        values = self.get_input("value")
        index = self.get_input("index")

        try:
            item = values[index]
        except IndexError:
            self.logger.error(f"Index {index} out of range for list of length {len(values)}")
            return Status.FAILURE

        self._set_output("result", item)
        return Status.SUCCESS
