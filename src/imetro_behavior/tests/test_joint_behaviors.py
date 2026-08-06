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

import pytest
from pathlib import Path

from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rclpy.node import Node

from imetro_behavior.joint_behaviors import JointNamesAndPositionsFromYaml


@pytest.fixture()
def yaml_joint_behavior(ros_node: Node, tmp_path: Path, mocker) -> JointNamesAndPositionsFromYaml:
    """A JointNamesAndPositionsFromYaml behavior whose package share directory resolves to a temporary path."""
    mocker.patch(
        "imetro_behavior.joint_behaviors.get_package_share_path",
        return_value=tmp_path,
    )
    (tmp_path / "joint_states.yaml").write_text(
        """
home:
    joint_names:
        - joint_1
        - joint_2
        - joint_3
    positions:
        - 0.1
        - 0.2
        - 0.3
"""
    )

    behavior = JointNamesAndPositionsFromYaml(name="joint_names_and_positions_from_yaml")
    behavior.setup(node=ros_node)
    behavior.setup_ports()
    Blackboard.set(behavior._get_blackboard_key("package_name"), "some_package")
    Blackboard.set(behavior._get_blackboard_key("yaml_file"), "joint_states.yaml")
    return behavior


def test_joint_names_and_positions_from_yaml(yaml_joint_behavior: JointNamesAndPositionsFromYaml) -> None:
    Blackboard.set(yaml_joint_behavior._get_blackboard_key("state_name"), "home")

    yaml_joint_behavior.tick_once()
    assert yaml_joint_behavior.status == Status.SUCCESS
    joint_names = yaml_joint_behavior.get_last_output("joint_names")
    assert joint_names == ["joint_1", "joint_2", "joint_3"]
    joint_positions = yaml_joint_behavior.get_last_output("joint_positions")
    assert joint_positions == [0.1, 0.2, 0.3]


def test_joint_names_and_positions_from_yaml_missing_state(yaml_joint_behavior: JointNamesAndPositionsFromYaml) -> None:
    Blackboard.set(yaml_joint_behavior._get_blackboard_key("state_name"), "retreat")
    yaml_joint_behavior.tick_once()
    assert yaml_joint_behavior.status == Status.FAILURE


def test_joint_names_and_positions_from_yaml_missing_file(yaml_joint_behavior: JointNamesAndPositionsFromYaml) -> None:
    Blackboard.set(yaml_joint_behavior._get_blackboard_key("yaml_file"), "nonexistent.yaml")
    Blackboard.set(yaml_joint_behavior._get_blackboard_key("state_name"), "home")
    yaml_joint_behavior.tick_once()
    assert yaml_joint_behavior.status == Status.FAILURE
