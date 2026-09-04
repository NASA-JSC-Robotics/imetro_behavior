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

import py_trees
from imetro_behavior.decorators import SuccessIfVariableIsFalse, SuccessIfVariableIsTrue
from py_trees.blackboard import Blackboard
from py_trees.common import Status


def test_success_if_variable_is_true_skips_child() -> None:
    child = py_trees.behaviours.Failure(name="child")
    decorator = SuccessIfVariableIsTrue(name="check_true", child=child)
    decorator.setup_ports()

    Blackboard.set(decorator._get_blackboard_key("variable"), True)

    decorator.tick_once()
    assert decorator.status == Status.SUCCESS
    # The child should never have been ticked.
    assert child.status == Status.INVALID


def test_success_if_variable_is_true_ticks_child() -> None:
    child = py_trees.behaviours.Failure(name="child")
    decorator = SuccessIfVariableIsTrue(name="check_true", child=child)
    decorator.setup_ports()

    Blackboard.set(decorator._get_blackboard_key("variable"), False)

    decorator.tick_once()
    assert decorator.status == Status.FAILURE
    assert child.status == Status.FAILURE


def test_success_if_variable_is_true_invalidates_running_child() -> None:
    child = py_trees.behaviours.Running(name="child")
    decorator = SuccessIfVariableIsTrue(name="check_true", child=child)
    decorator.setup_ports()

    Blackboard.set(decorator._get_blackboard_key("variable"), False)
    decorator.tick_once()
    assert decorator.status == Status.RUNNING
    assert child.status == Status.RUNNING

    # Once the variable becomes true, the running child should be halted.
    Blackboard.set(decorator._get_blackboard_key("variable"), True)
    decorator.tick_once()
    assert decorator.status == Status.SUCCESS
    assert child.status == Status.INVALID


def test_success_if_variable_is_false_skips_child() -> None:
    child = py_trees.behaviours.Failure(name="child")
    decorator = SuccessIfVariableIsFalse(name="check_false", child=child)
    decorator.setup_ports()

    Blackboard.set(decorator._get_blackboard_key("variable"), False)

    decorator.tick_once()
    assert decorator.status == Status.SUCCESS
    assert child.status == Status.INVALID


def test_success_if_variable_is_false_ticks_child() -> None:
    child = py_trees.behaviours.Failure(name="child")
    decorator = SuccessIfVariableIsFalse(name="check_false", child=child)
    decorator.setup_ports()

    Blackboard.set(decorator._get_blackboard_key("variable"), True)

    decorator.tick_once()
    assert decorator.status == Status.FAILURE
    assert child.status == Status.FAILURE
