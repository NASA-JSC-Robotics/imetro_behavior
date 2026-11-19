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

"""Tests for the behavior trial runner, ending with test runs against an executor in the same process."""

import threading
from pathlib import Path

import pytest
import rclpy
import yaml
from imetro_behavior_testing.test_runner import (
    BehaviorTrialRunner,
    expand_trials,
    load_test_file,
    resolve_input_mappings,
    set_param,
)
from py_trees.blackboard import Blackboard
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from imetro_behavior.executor import BehaviorTreeExecutor, BehaviorTreeServer


def test_set_param_nested_and_indexed() -> None:
    params = {"a": {"b": [0.0, [1.0, 2.0]]}, "c": 1}
    set_param(params, "a.b[1][0]", 5.0)
    set_param(params, "c", 2)
    assert params == {"a": {"b": [0.0, [5.0, 2.0]]}, "c": 2}


@pytest.mark.parametrize("path", ["missing", "a.missing", "a.b[2]", "c[0]", "a..b", "a[x]"])
def test_set_param_invalid(path: str) -> None:
    with pytest.raises(ValueError):
        set_param({"a": {"b": [0.0, 1.0]}, "c": 1}, path, 0.0)


def test_expand_trials() -> None:
    config = {
        "repeats": 2,
        "params": {"speed": 0.02, "offset": [0.0, 0.0]},
        "sweeps": [
            {"name": "baseline", "repeats": 1},
            {
                "name": "grid",
                "params": {"speed": 0.04},
                "vary": {"offset[0]": [-0.01, 0.01], "offset[1]": [0.5]},
            },
        ],
    }
    trials = expand_trials(config)
    assert [trial["name"] for trial in trials] == [
        "baseline_run_0",
        "grid_-0.01_0.5_run_0",
        "grid_-0.01_0.5_run_1",
        "grid_0.01_0.5_run_0",
        "grid_0.01_0.5_run_1",
    ]
    assert trials[0]["params"] == {"speed": 0.02, "offset": [0.0, 0.0]}
    assert trials[3]["params"] == {"speed": 0.04, "offset": [0.01, 0.5]}
    # The top level params are not changed by the sweeps.
    assert config["params"] == {"speed": 0.02, "offset": [0.0, 0.0]}


def test_expand_trials_without_sweeps() -> None:
    trials = expand_trials({"params": {"speed": 0.02}})
    assert trials == [{"name": "baseline_run_0", "params": {"speed": 0.02}}]


@pytest.mark.parametrize(
    "sweeps",
    [
        [{"vary": {"speed": [0.01]}}],  # No name
        [{"name": "a", "speeds": [0.01]}],  # Unknown key
        [{"name": "a", "vary": {"speeds": [0.01]}}],  # Unknown parameter
        [{"name": "a", "vary": {"speed": 0.01}}],  # Not a list
        [{"name": "a"}, {"name": "a"}],  # Duplicate names
    ],
)
def test_expand_trials_invalid(sweeps: list) -> None:
    with pytest.raises(ValueError):
        expand_trials({"params": {"speed": 0.02}, "sweeps": sweeps})


def test_resolve_input_mappings() -> None:
    config = {
        "input_mappings": {
            "params_file": "${params_file}",
            "bag_path": "${trial_dir}/bag",
            "required_depth": 0.1,
            "enabled": True,
            "xyz": [1.0, 2.0, 3.0],
        }
    }
    placeholders = {"params_file": "/tmp/t/params.yaml", "trial_dir": "/tmp/t"}
    assert resolve_input_mappings(config, placeholders) == {
        "params_file": "/tmp/t/params.yaml",
        "bag_path": "/tmp/t/bag",
        "required_depth": "0.1",
        "enabled": "true",
        "xyz": "1.0, 2.0, 3.0",
    }


def test_resolve_input_mappings_unknown_placeholder() -> None:
    with pytest.raises(ValueError, match="unknown placeholder"):
        resolve_input_mappings({"input_mappings": {"a": "${nope}"}}, {"trial_dir": "/tmp"})


def test_load_test_file(tmp_path: Path) -> None:
    test_file = tmp_path / "test.yaml"
    test_file.write_text(yaml.safe_dump({"behavior": "my_tree", "params": {"speed": 0.02}}))
    config, trials = load_test_file(test_file)
    assert config["behavior"] == "my_tree"
    assert len(trials) == 1


@pytest.mark.parametrize(
    "contents",
    [
        {"params": {"speed": 0.02}},  # No behavior
        {"behavior": "my_tree", "behaviour": "typo"},  # Unknown key
        {"behavior": "my_tree", "input_mappings": {"a": "${nope}"}},  # Unknown placeholder
        ["not", "a", "mapping"],
    ],
)
def test_load_test_file_invalid(tmp_path: Path, contents: dict | list) -> None:
    test_file = tmp_path / "test.yaml"
    test_file.write_text(yaml.safe_dump(contents))
    with pytest.raises(ValueError):
        load_test_file(test_file)


# Reads the speed parameter, fails if it is zero, and writes it back out as a result.
TRIAL_TREE = """
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree" params_file="params.yaml" results_file="results.yaml">
    <Sequence name="sequence">
      <ReadYamlFile name="read_params" file_path="{params_file}" dict="{params}" />
      <GetDictValue name="get_speed" dict="{params}" key="speed" value="{speed}" />
      <SetDictValue name="set_speed" key="speed" value="{speed}" result="{results}" />
      <SetDictValue name="set_missing" dict="{results}" key="missing" value="{never_set}" result="{results}" />
      <WriteYamlFile name="write_results" dict="{results}" file_path="{results_file}" />
      <BlackboardMath name="check_speed" operand1="1.0" operator="/" operand2="{speed}" result="{inverse}" />
    </Sequence>
  </BehaviorTree>
</root>
"""


@pytest.fixture()
def nodes():
    """A behavior tree server spinning in the background, and a node for the runner."""
    Blackboard.clear()
    rclpy.init()
    server_node = Node("test_behavior_server")
    executor = BehaviorTreeExecutor(
        server_node,
        config={
            "params": {"dt": 0.01},
            "imports": ["imetro_behavior.basic_behaviors", "imetro_behavior.dict_behaviors"],
        },
    )
    BehaviorTreeServer(server_node, executor)
    spinner = SingleThreadedExecutor()
    spinner.add_node(server_node)
    thread = threading.Thread(target=spinner.spin, daemon=True)
    thread.start()
    runner_node = Node("test_behavior_trial_runner")

    yield runner_node

    spinner.shutdown()
    runner_node.destroy_node()
    server_node.destroy_node()
    rclpy.try_shutdown()
    thread.join(timeout=5.0)


def write_test_file(tmp_path: Path, speeds: list[float], stop_on_failure: bool) -> Path:
    tree_path = tmp_path / "trial_tree.xml"
    tree_path.write_text(TRIAL_TREE)
    test_file = tmp_path / "test.yaml"
    test_file.write_text(
        yaml.safe_dump(
            {
                "behavior": str(tree_path),
                "input_mappings": {"params_file": "${params_file}", "results_file": "${results_file}"},
                "stop_on_failure": stop_on_failure,
                "params": {"speed": 0.01},
                "sweeps": [{"name": "speed", "vary": {"speed": speeds}}],
            }
        )
    )
    return test_file


def read_summaries(output_dir: Path) -> dict[str, dict]:
    (session_dir,) = output_dir.iterdir()
    assert (session_dir / "test_file.yaml").exists()
    return {
        trial_dir.name: yaml.safe_load((trial_dir / "trial.yaml").read_text())
        for trial_dir in session_dir.iterdir()
        if trial_dir.is_dir()
    }


def test_run_trials(nodes: Node, tmp_path: Path) -> None:
    runner = BehaviorTrialRunner(nodes)
    output_dir = tmp_path / "results"
    assert runner.run(write_test_file(tmp_path, [0.01, 0.04], stop_on_failure=True), output_dir)

    summaries = read_summaries(output_dir)
    assert summaries["speed_0.04_run_0"] == {
        "name": "speed_0.04_run_0",
        "status": "succeeded",
        "message": "",
        "params": {"speed": 0.04},
        "results": {"speed": 0.04, "missing": None},
    }
    assert summaries["speed_0.01_run_0"]["results"]["speed"] == 0.01


def test_run_trials_failure(nodes: Node, tmp_path: Path) -> None:
    """A zero speed fails the tree after it writes its results."""
    output_dir = tmp_path / "results"
    test_file = write_test_file(tmp_path, [0.01, 0.0, 0.04], stop_on_failure=True)
    assert not BehaviorTrialRunner(nodes).run(test_file, output_dir)
    summaries = read_summaries(output_dir)
    assert sorted(summaries) == ["speed_0.01_run_0", "speed_0_run_0"]
    assert summaries["speed_0_run_0"]["status"] == "failed"
    assert summaries["speed_0_run_0"]["results"] == {"speed": 0.0, "missing": None}

    output_dir = tmp_path / "results_continue"
    test_file = write_test_file(tmp_path, [0.01, 0.0, 0.04], stop_on_failure=False)
    assert not BehaviorTrialRunner(nodes).run(test_file, output_dir)
    assert len(read_summaries(output_dir)) == 3


def test_run_trials_aborted(nodes: Node, tmp_path: Path) -> None:
    """A tree that cannot be loaded aborts the goal, and the trial has no results."""
    test_file = tmp_path / "test.yaml"
    test_file.write_text(yaml.safe_dump({"behavior": "no_such_tree"}))
    output_dir = tmp_path / "results"
    assert not BehaviorTrialRunner(nodes).run(test_file, output_dir)
    (summary,) = read_summaries(output_dir).values()
    assert summary["status"] == "aborted"
    assert "Could not find tree" in summary["message"]
    assert summary["results"] is None
