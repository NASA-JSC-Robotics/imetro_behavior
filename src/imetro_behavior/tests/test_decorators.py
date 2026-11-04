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
import pytest
from py_trees.blackboard import Blackboard
from py_trees.common import Status

from imetro_behavior.decorators import (
    ForEachItem,
    OutputChildSuccess,
    SuccessIfVariableIsFalse,
    SuccessIfVariableIsTrue,
)


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


@pytest.fixture
def child() -> py_trees.behaviour.Behaviour:
    """The child of the decorator, which tests can override by parametrizing `child`."""
    return py_trees.behaviours.Success(name="child")


@pytest.fixture
def items() -> list:
    """The items to iterate over, which tests can override by parametrizing `items`."""
    return ["a", "b"]


@pytest.fixture
def for_each(child: py_trees.behaviour.Behaviour, items: list) -> ForEachItem:
    decorator = ForEachItem(name="for_each", child=child)
    decorator.setup_ports()
    Blackboard.set(decorator._get_blackboard_key("items"), items)
    return decorator


def test_for_each_item_runs_child_per_item(for_each: ForEachItem) -> None:
    for_each.tick_once()
    assert for_each.status == Status.RUNNING
    assert Blackboard.get(for_each._get_blackboard_key("item")) == "b"

    for_each.tick_once()
    assert for_each.status == Status.SUCCESS
    assert Blackboard.get(for_each._get_blackboard_key("item")) == "b"


@pytest.mark.parametrize("items", [["a"]])
def test_for_each_item_restarts_from_first_item(for_each: ForEachItem) -> None:
    for_each.tick_once()
    assert for_each.status == Status.SUCCESS

    Blackboard.set(for_each._get_blackboard_key("items"), ["c", "d"])
    for_each.tick_once()
    assert for_each.status == Status.RUNNING
    assert Blackboard.get(for_each._get_blackboard_key("item")) == "d"


@pytest.mark.parametrize("items", [[]])
@pytest.mark.parametrize("child", [py_trees.behaviours.Failure(name="child")])
def test_for_each_item_empty_list_skips_child(for_each: ForEachItem, child: py_trees.behaviour.Behaviour) -> None:
    for_each.tick_once()
    assert for_each.status == Status.SUCCESS
    assert child.status == Status.INVALID


@pytest.mark.parametrize("child", [py_trees.behaviours.Failure(name="child")])
def test_for_each_item_stops_on_child_failure(for_each: ForEachItem) -> None:
    for_each.tick_once()
    assert for_each.status == Status.FAILURE
    assert Blackboard.get(for_each._get_blackboard_key("item")) == "a"


@pytest.fixture
def output_child_success(child: py_trees.behaviour.Behaviour) -> OutputChildSuccess:
    decorator = OutputChildSuccess(name="output_child_success", child=child)
    decorator.setup_ports()
    return decorator


def test_output_child_success_when_child_succeeds(output_child_success: OutputChildSuccess) -> None:
    output_child_success.tick_once()
    assert output_child_success.status == Status.SUCCESS
    assert Blackboard.get(output_child_success._get_blackboard_key("succeeded")) is True


@pytest.mark.parametrize("child", [py_trees.behaviours.Failure(name="child")])
def test_output_child_success_when_child_fails(output_child_success: OutputChildSuccess) -> None:
    output_child_success.tick_once()
    assert output_child_success.status == Status.SUCCESS
    assert Blackboard.get(output_child_success._get_blackboard_key("succeeded")) is False


@pytest.mark.parametrize(
    "child",
    [py_trees.behaviours.StatusQueue(name="child", queue=[Status.RUNNING], eventually=Status.FAILURE)],
)
def test_output_child_success_waits_for_child(output_child_success: OutputChildSuccess) -> None:
    output_child_success.tick_once()
    assert output_child_success.status == Status.RUNNING
    assert not Blackboard.exists(output_child_success._get_blackboard_key("succeeded"))

    output_child_success.tick_once()
    assert output_child_success.status == Status.SUCCESS
    assert Blackboard.get(output_child_success._get_blackboard_key("succeeded")) is False
