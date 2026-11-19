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

from pathlib import Path

import pytest
import yaml

from imetro_behavior_testing.test_runner import expand_trials, load_test_file, resolve_input_mappings, set_param


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
