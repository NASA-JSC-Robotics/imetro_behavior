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

from imetro_behavior.basic_behaviors import BlackboardMath, WaitForDuration
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rclpy.duration import Duration
from rclpy.node import Node


def test_wait_for_duration_zero_duration(ros_node: Node) -> None:
    behavior = WaitForDuration(name="wait", duration_sec=0.0)
    behavior.setup(node=ros_node)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS


def test_wait_for_duration(ros_node: Node) -> None:
    behavior = WaitForDuration(name="wait", duration_sec=0.1)
    behavior.setup(node=ros_node)

    behavior.tick_once()
    assert behavior.status == Status.RUNNING

    # Sleep on the same ROS clock the behavior uses, rather than wall time.
    ros_node.get_clock().sleep_for(Duration(seconds=0.11))

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS


def test_math_add(ros_node: Node) -> None:
    """Tests simple addition"""
    behavior = BlackboardMath(name="math")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    Blackboard.set(behavior._get_blackboard_key("operand1"), 2.0)
    Blackboard.set(behavior._get_blackboard_key("operator"), "+")
    Blackboard.set(behavior._get_blackboard_key("operand2"), 2.0)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("result") == 4.0


def test_math_sub(ros_node: Node) -> None:
    """Tests simple subtraction"""
    behavior = BlackboardMath(name="math")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    Blackboard.set(behavior._get_blackboard_key("operand1"), 5.0)
    Blackboard.set(behavior._get_blackboard_key("operator"), "-")
    Blackboard.set(behavior._get_blackboard_key("operand2"), 2.0)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("result") == 3.0


def test_math_mult(ros_node: Node) -> None:
    """Tests simple multiplication"""
    behavior = BlackboardMath(name="math")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    Blackboard.set(behavior._get_blackboard_key("operand1"), 5.0)
    Blackboard.set(behavior._get_blackboard_key("operator"), "*")
    Blackboard.set(behavior._get_blackboard_key("operand2"), 2.0)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("result") == 10.0


def test_math_div(ros_node: Node) -> None:
    """Tests simple division"""
    behavior = BlackboardMath(name="math")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    Blackboard.set(behavior._get_blackboard_key("operand1"), 6.0)
    Blackboard.set(behavior._get_blackboard_key("operator"), "/")
    Blackboard.set(behavior._get_blackboard_key("operand2"), 2.0)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("result") == 3.0


def test_math_div_by_zero(ros_node: Node) -> None:
    """Tests dividing by zero"""
    behavior = BlackboardMath(name="math")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    Blackboard.set(behavior._get_blackboard_key("operand1"), 2.0)
    Blackboard.set(behavior._get_blackboard_key("operator"), "/")
    Blackboard.set(behavior._get_blackboard_key("operand2"), 0.0)

    behavior.tick_once()
    assert behavior.status == Status.FAILURE


def test_math_pow(ros_node: Node) -> None:
    """Tests simple exponent"""
    behavior = BlackboardMath(name="math")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    Blackboard.set(behavior._get_blackboard_key("operand1"), 3.0)
    Blackboard.set(behavior._get_blackboard_key("operator"), "^")
    Blackboard.set(behavior._get_blackboard_key("operand2"), 2.0)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("result") == 9.0


def test_math_wrong_operator(ros_node: Node) -> None:
    """Tests what happens when an improper operator is given"""
    behavior = BlackboardMath(name="math")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    Blackboard.set(behavior._get_blackboard_key("operand1"), 2.0)
    Blackboard.set(behavior._get_blackboard_key("operator"), "#")
    Blackboard.set(behavior._get_blackboard_key("operand2"), 2.0)

    behavior.tick_once()
    assert behavior.status == Status.FAILURE
