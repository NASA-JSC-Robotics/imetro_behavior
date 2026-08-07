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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts

from imetro_behavior.executor import BehaviorTreeServer, BehaviorTreeExecutor


class AlwaysSucceed(BehaviourWithPorts):
    """Test behavior that immediately succeeds. Auto-registered for use in tree XML."""

    @classmethod
    def input_ports(cls) -> dict:
        return {}

    @classmethod
    def output_ports(cls) -> dict:
        return {}

    def update(self) -> Status:
        return Status.SUCCESS


class AlwaysFail(BehaviourWithPorts):
    """Test behavior that immediately fails. Auto-registered for use in tree XML."""

    @classmethod
    def input_ports(cls) -> dict:
        return {}

    @classmethod
    def output_ports(cls) -> dict:
        return {}

    def update(self) -> Status:
        return Status.FAILURE


class RunForever(BehaviourWithPorts):
    """Test behavior that runs forever. Auto-registered for use in tree XML."""

    @classmethod
    def input_ports(cls) -> dict:
        return {}

    @classmethod
    def output_ports(cls) -> dict:
        return {}

    def update(self) -> Status:
        return Status.RUNNING


@pytest.fixture()
def executor(ros_node: Node) -> BehaviorTreeExecutor:
    """A behavior tree executor with a minimal configuration (no imports or search paths)."""
    return BehaviorTreeExecutor(ros_node, config={"params": {"dt": 0.01}})


def write_tree(tmp_path: Path, name: str, child_tag: str) -> str:
    """Writes a minimal tree XML file with a single child behavior and returns its path."""
    tree_xml = f"""
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <Sequence name="sequence">
      <{child_tag} name="child" />
    </Sequence>
  </BehaviorTree>
</root>
"""
    tree_path = tmp_path / f"{name}.xml"
    tree_path.write_text(tree_xml)
    return tree_path.as_posix()


def test_load_tree_empty_name(executor: BehaviorTreeExecutor) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        executor.load_tree("")
    assert "Behavior tree file name cannot be empty!" in str(exc_info.value)


def test_load_tree_not_found(executor: BehaviorTreeExecutor) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        executor.load_tree("nonexistent_tree")
    assert "Could not find tree: nonexistent_tree" in str(exc_info.value)


def test_load_tree_parse_failure(executor: BehaviorTreeExecutor, tmp_path: Path) -> None:
    tree_path = write_tree(tmp_path, "bad_tree", "NotARegisteredBehavior")

    with pytest.raises(RuntimeError) as exc_info:
        executor.load_tree(tree_path)
    assert "Failed to parse XML file" in str(exc_info.value)


def test_load_tick_stop(executor: BehaviorTreeExecutor, tmp_path: Path) -> None:
    tree_path = write_tree(tmp_path, "run_forever", "RunForever")

    executor.load_tree(tree_path)
    assert executor.current_behavior == tree_path
    assert executor.status == Status.INVALID  # Not yet ticked

    assert executor.tick() == Status.RUNNING
    assert executor.status == Status.RUNNING

    executor.stop_tree()
    assert executor.current_behavior is None
    assert executor.status == Status.INVALID


def test_tick_without_tree(executor: BehaviorTreeExecutor) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        executor.tick()
    assert "No behavior tree loaded." in str(exc_info.value)


def test_run_trees_to_completion(executor: BehaviorTreeExecutor, tmp_path: Path) -> None:
    success_path = write_tree(tmp_path, "success_tree", "AlwaysSucceed")
    failure_path = write_tree(tmp_path, "failure_tree", "AlwaysFail")

    assert executor.run_tree(success_path) == Status.SUCCESS
    # After completion, the tree is stopped so another behavior can be swapped in and run.
    assert executor.current_behavior is None
    assert executor.status == Status.INVALID

    assert executor.run_tree(failure_path) == Status.FAILURE
    assert executor.run_tree(success_path) == Status.SUCCESS


def test_run_tree_cancel(executor: BehaviorTreeExecutor, tmp_path: Path) -> None:
    tree_path = write_tree(tmp_path, "run_forever", "RunForever")

    future = ThreadPoolExecutor(max_workers=1).submit(executor.run_tree, tree_path)

    # Wait until the tree is actually ticking, which guarantees run_tree() has already
    # dropped stale cancellation requests, so this one cannot be lost.
    # If run_tree() fails outright instead, the future completes and result() re-raises.
    while executor.status != Status.RUNNING and not future.done():
        time.sleep(0.1)

    executor.request_cancel()
    assert future.result(timeout=5.0) == Status.INVALID
    assert executor.current_behavior is None
    assert executor.status == Status.INVALID


def test_run_tree_ignores_stale_cancel(executor: BehaviorTreeExecutor, tmp_path: Path) -> None:
    tree_path = write_tree(tmp_path, "success_tree", "AlwaysSucceed")

    # A cancellation requested before the run starts should not affect it.
    executor.request_cancel()
    assert executor.run_tree(tree_path) == Status.SUCCESS


def test_run_tree_resets_blackboard(executor: BehaviorTreeExecutor, tmp_path: Path) -> None:
    tree_path = write_tree(tmp_path, "success_tree", "AlwaysSucceed")
    Blackboard.set("/leftover_data", 42)

    executor.run_tree(tree_path)
    assert "/leftover_data" not in Blackboard.storage
    assert Blackboard.get("/ros/tf_buffer") is executor._tf_buffer


def test_load_tree_from_search_path(ros_node: Node) -> None:
    # The default configuration points at this package's installed trees.
    executor = BehaviorTreeExecutor(ros_node)
    assert executor.find_tree("switch_controllers") is not None
    assert executor.find_tree("nonexistent_tree") is None

    executor.load_tree("switch_controllers")
    assert executor.current_behavior == "switch_controllers"
    executor.stop_tree()


def test_create_server(ros_node: Node) -> None:
    ros_executor = MultiThreadedExecutor()
    ros_executor.add_node(ros_node)

    BehaviorTreeServer(ros_node)

    for _ in range(10):
        ros_executor.spin_once(timeout_sec=0.1)

    ros_executor.shutdown()
