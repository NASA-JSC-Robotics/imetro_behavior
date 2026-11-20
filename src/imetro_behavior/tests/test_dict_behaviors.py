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
from typing import Any

import numpy as np
import pytest
import yaml
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts

from imetro_behavior.dict_behaviors import GetDictValue, ReadYamlFile, SetDictValue, WriteYamlFile


def make_behavior(cls: type[BehaviourWithPorts], **inputs: Any) -> BehaviourWithPorts:
    """Creates a behavior and sets its inputs on the blackboard."""
    behavior = cls(name=cls.__name__)
    behavior.setup_ports()
    behavior.setup()
    for port, value in inputs.items():
        Blackboard.set(behavior._get_blackboard_key(port), value)
    return behavior


def test_read_yaml_file(tmp_path: Path) -> None:
    params = {"speed": 0.02, "offset": [0.0, 1.0], "nested": {"enabled": True}}
    file_path = tmp_path / "params.yaml"
    file_path.write_text(yaml.safe_dump(params))
    behavior = make_behavior(ReadYamlFile, file_path=str(file_path))
    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("dict") == params


def test_read_yaml_file_empty(tmp_path: Path) -> None:
    file_path = tmp_path / "params.yaml"
    file_path.write_text("")
    behavior = make_behavior(ReadYamlFile, file_path=str(file_path))
    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("dict") == {}


def test_read_yaml_file_missing(tmp_path: Path) -> None:
    behavior = make_behavior(ReadYamlFile, file_path=str(tmp_path / "missing.yaml"))
    behavior.tick_once()
    assert behavior.status == Status.FAILURE


def test_read_yaml_file_not_a_mapping(tmp_path: Path) -> None:
    file_path = tmp_path / "params.yaml"
    file_path.write_text(yaml.safe_dump(["a", "list"]))
    behavior = make_behavior(ReadYamlFile, file_path=str(file_path))
    behavior.tick_once()
    assert behavior.status == Status.FAILURE


def test_read_yaml_file_malformed(tmp_path: Path) -> None:
    # Malformed YAML can only be written as raw text.
    file_path = tmp_path / "params.yaml"
    file_path.write_text("a: [unclosed\n")
    behavior = make_behavior(ReadYamlFile, file_path=str(file_path))
    behavior.tick_once()
    assert behavior.status == Status.FAILURE


def test_write_yaml_file(tmp_path: Path) -> None:
    file_path = tmp_path / "new_folder" / "results.yaml"
    values = {"depth": np.float64(0.1), "xyz": np.array([1.0, 2.0]), "ok": True, "missing": None}
    behavior = make_behavior(WriteYamlFile, dict=values, file_path=str(file_path))
    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert yaml.safe_load(file_path.read_text()) == {"depth": 0.1, "xyz": [1.0, 2.0], "ok": True, "missing": None}


def test_get_dict_value() -> None:
    values = {"a": {"b": [1.0, 2.0]}, "c": 3}
    behavior = make_behavior(GetDictValue, dict=values, key="a.b")
    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    value = behavior.get_last_output("value")
    assert value == [1.0, 2.0]

    # The output is a copy.
    value.append(3.0)
    assert values["a"]["b"] == [1.0, 2.0]


@pytest.mark.parametrize("key", ["missing", "a.missing", "c.d", "e"])
def test_get_dict_value_invalid(key: str) -> None:
    behavior = make_behavior(GetDictValue, dict={"a": {"b": 1}, "c": 3, "e": None}, key=key)
    behavior.tick_once()
    assert behavior.status == Status.FAILURE


def test_set_dict_value_new_dict() -> None:
    behavior = make_behavior(SetDictValue, key="a.b", value=1.0)
    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("result") == {"a": {"b": 1.0}}


def test_set_dict_value_existing_dict() -> None:
    values = {"a": {"b": 1.0}, "c": 2}
    behavior = make_behavior(SetDictValue, dict=values, key="a.d", value=[3.0])
    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("result") == {"a": {"b": 1.0, "d": [3.0]}, "c": 2}
    # The input is left untouched.
    assert values == {"a": {"b": 1.0}, "c": 2}


def test_set_dict_value_no_value() -> None:
    behavior = make_behavior(SetDictValue, dict={"a": 1}, key="b")
    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("result") == {"a": 1, "b": None}


def test_set_dict_value_parent_not_dict() -> None:
    behavior = make_behavior(SetDictValue, dict={"a": 1}, key="a.b", value=2)
    behavior.tick_once()
    assert behavior.status == Status.FAILURE
