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
import time

import py_trees
from py_trees.blackboard import Blackboard
from py_trees.common import Status

from imetro_behavior.decorators import SuccessIfVariableIsFalse, SuccessIfVariableIsTrue, TimeoutPort


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


def test_timeout() -> None:
    running = py_trees.behaviours.Running(name="Running")
    timeout = TimeoutPort(name="Timeout", child=running)
    timeout.setup_ports()

    Blackboard.set(timeout._get_blackboard_key("duration"), 0.2)

    print(py_trees.display.unicode_tree(timeout))
    visitor = py_trees.visitors.DebugVisitor()

    # Test that it times out and re-initialises properly
    for i in range(0, 2):
        py_trees.tests.tick_tree(timeout, 2 * i + 1, 2 * i + 1, visitors=[visitor])

        print("\n--------- Assertions ---------\n")
        print("timeout.status == py_trees.common.Status.RUNNING")
        assert timeout.status == py_trees.common.Status.RUNNING
        print("running.status == py_trees.common.Status.RUNNING")
        assert running.status == py_trees.common.Status.RUNNING

        time.sleep(0.3)
        py_trees.tests.tick_tree(timeout, 2 * i + 2, 2 * i + 2, visitors=[visitor])

        print("\n--------- Assertions ---------\n")
        print("timeout.status == py_trees.common.Status.FAILURE")
        assert timeout.status == py_trees.common.Status.FAILURE
        print("running.status == py_trees.common.Status.INVALID")  # type: ignore[unreachable]
        assert running.status == py_trees.common.Status.INVALID

    # test that it passes on success
    count = py_trees.behaviours.StatusQueue(
        name="Queue",
        queue=[py_trees.common.Status.RUNNING],
        eventually=py_trees.common.Status.SUCCESS,
    )
    timeout = TimeoutPort(name="Timeout", child=count)
    timeout.setup_ports()

    Blackboard.set(timeout._get_blackboard_key("duration"), 0.2)

    print(py_trees.display.unicode_tree(timeout))

    py_trees.tests.tick_tree(timeout, 1, 1, visitors=[visitor])

    print("\n--------- Assertions ---------\n")
    print("timeout.status == py_trees.common.Status.RUNNING")
    assert timeout.status == py_trees.common.Status.RUNNING
    print("count.status == py_trees.common.Status.RUNNING")
    assert count.status == py_trees.common.Status.RUNNING

    py_trees.tests.tick_tree(timeout, 2, 2, visitors=[visitor])

    print("\n--------- Assertions ---------\n")
    print("timeout.status == py_trees.common.Status.SUCCESS")
    assert timeout.status == py_trees.common.Status.SUCCESS
    print("count.status == py_trees.common.Status.SUCCESS")  # type: ignore[unreachable]
    assert count.status == py_trees.common.Status.SUCCESS

    # test that it passes on failure
    failure = py_trees.behaviours.Failure(name="Failure")
    timeout = TimeoutPort(name="Timeout", child=failure)
    timeout.setup_ports()

    Blackboard.set(timeout._get_blackboard_key("duration"), 0.2)

    print(py_trees.display.unicode_tree(timeout))

    py_trees.tests.tick_tree(timeout, 1, 1, visitors=[visitor])

    print("\n--------- Assertions ---------\n")
    print("timeout.status == py_trees.common.Status.FAILURE")
    assert timeout.status == py_trees.common.Status.FAILURE
    print("failure.status == py_trees.common.Status.FAILURE")
    assert failure.status == py_trees.common.Status.FAILURE

    # test that it succeeds if child succeeds on last tick
    count = py_trees.behaviours.StatusQueue(
        name="Queue",
        queue=[py_trees.common.Status.RUNNING],
        eventually=py_trees.common.Status.SUCCESS,
    )
    timeout = TimeoutPort(name="Timeout", child=count)
    timeout.setup_ports()

    Blackboard.set(timeout._get_blackboard_key("duration"), 0.1)
    print(py_trees.display.unicode_tree(timeout))

    py_trees.tests.tick_tree(timeout, 1, 1, visitors=[visitor])

    print("\n--------- Assertions ---------\n")
    print("timeout.status == py_trees.common.Status.RUNNING")
    assert timeout.status == py_trees.common.Status.RUNNING
    print("count.status == py_trees.common.Status.RUNNING")
    assert count.status == py_trees.common.Status.RUNNING

    time.sleep(0.2)  # go past the duration
    py_trees.tests.tick_tree(timeout, 2, 2, visitors=[visitor])

    print("\n--------- Assertions ---------\n")
    print("timeout.status == py_trees.common.Status.SUCCESS")
    assert timeout.status == py_trees.common.Status.SUCCESS
    print("count.status == py_trees.common.Status.SUCCESS")
    assert count.status == py_trees.common.Status.SUCCESS
