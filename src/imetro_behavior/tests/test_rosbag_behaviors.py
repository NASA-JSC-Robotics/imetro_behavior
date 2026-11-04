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

import py_trees
import pytest
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from pytest_mock import MockerFixture
from rclpy.duration import Duration
from rclpy.node import Node
from rosbag2_interfaces.srv import Record

from imetro_behavior.rosbag_behaviors import RecordRosbag


def _done_future(mocker: MockerFixture, response: Any = None) -> Any:
    future = mocker.Mock()
    future.done.return_value = True
    future.result.return_value = response
    return future


@pytest.fixture
def child() -> py_trees.behaviour.Behaviour:
    """The child of the decorator, which tests can override by parametrizing `child`."""
    return py_trees.behaviours.Success(name="child")


@pytest.fixture
def output_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "bag"


@pytest.fixture
def behavior(
    child: py_trees.behaviour.Behaviour, ros_node: Node, output_path: Path, mocker: MockerFixture
) -> RecordRosbag:
    """A behavior whose mock recorder service clients respond immediately."""
    behavior = RecordRosbag(name="record", child=child, service_timeout=0.01)
    behavior.setup_ports()
    behavior.node = ros_node
    behavior.stop_client = mocker.Mock()
    behavior.stop_client.call_async.return_value = _done_future(mocker)
    behavior.resume_client = mocker.Mock()
    behavior.resume_client.call_async.return_value = _done_future(mocker)
    behavior.record_client = mocker.Mock()
    behavior.record_client.call_async.return_value = _done_future(mocker, Record.Response(return_code=0))
    Blackboard.set(behavior._get_blackboard_key("output_path"), str(output_path))
    return behavior


@pytest.mark.parametrize(
    "child",
    [py_trees.behaviours.StatusQueue(name="child", queue=[Status.RUNNING], eventually=Status.SUCCESS)],
)
def test_starts_bag_before_child_and_stops_after(
    behavior: RecordRosbag, child: py_trees.behaviour.Behaviour, output_path: Path
) -> None:
    behavior.tick_once()
    assert behavior.status == Status.RUNNING
    assert child.status == Status.INVALID
    assert output_path.parent.is_dir()
    behavior.stop_client.call_async.assert_called_once()
    behavior.resume_client.call_async.assert_called_once()
    assert behavior.record_client.call_async.call_args.args[0].uri == str(output_path)

    behavior.tick_once()
    assert child.status == Status.RUNNING

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.stop_client.call_async.call_count == 2


def test_sends_requests_in_order(behavior: RecordRosbag, child: py_trees.behaviour.Behaviour) -> None:
    behavior.stop_client.call_async.return_value.done.return_value = False
    behavior.tick_once()
    assert behavior.status == Status.RUNNING
    behavior.resume_client.call_async.assert_not_called()

    behavior.stop_client.call_async.return_value.done.return_value = True
    behavior.resume_client.call_async.return_value.done.return_value = False
    behavior.tick_once()
    assert behavior.status == Status.RUNNING
    behavior.stop_client.call_async.assert_called_once()
    behavior.record_client.call_async.assert_not_called()

    behavior.resume_client.call_async.return_value.done.return_value = True
    behavior.record_client.call_async.return_value.done.return_value = False
    behavior.tick_once()
    assert behavior.status == Status.RUNNING
    behavior.record_client.call_async.assert_called_once()
    assert child.status == Status.INVALID


@pytest.mark.parametrize("child", [py_trees.behaviours.Failure(name="child")])
def test_stops_recording_when_child_fails(behavior: RecordRosbag) -> None:
    behavior.tick_once()
    behavior.tick_once()
    assert behavior.status == Status.FAILURE
    assert behavior.stop_client.call_async.call_count == 2


@pytest.mark.parametrize("child", [py_trees.behaviours.Running(name="child")])
def test_stops_recording_when_interrupted(behavior: RecordRosbag, child: py_trees.behaviour.Behaviour) -> None:
    behavior.tick_once()
    behavior.tick_once()
    behavior.stop(Status.INVALID)
    assert child.status == Status.INVALID
    assert behavior.stop_client.call_async.call_count == 2


def test_fails_without_ticking_child_if_record_is_rejected(
    mocker: MockerFixture, behavior: RecordRosbag, child: py_trees.behaviour.Behaviour
) -> None:
    behavior.record_client.call_async.return_value = _done_future(
        mocker, Record.Response(return_code=1, error_string="Bag directory already exists")
    )

    behavior.tick_once()
    assert behavior.status == Status.FAILURE
    assert child.status == Status.INVALID
    behavior.stop_client.call_async.assert_called_once()


def test_fails_if_recorder_is_not_running(
    behavior: RecordRosbag, child: py_trees.behaviour.Behaviour, ros_node: Node
) -> None:
    behavior.stop_client.service_is_ready.return_value = False

    behavior.tick_once()
    assert behavior.status == Status.RUNNING

    ros_node.get_clock().sleep_for(Duration(seconds=0.02))
    behavior.tick_once()
    assert behavior.status == Status.FAILURE
    assert child.status == Status.INVALID


def test_stop_before_starting_does_nothing(behavior: RecordRosbag) -> None:
    behavior.stop(Status.INVALID)
    behavior.stop_client.call_async.assert_not_called()


@pytest.mark.parametrize("child", [py_trees.behaviours.Running(name="child")])
def test_shutdown_stops_recording(behavior: RecordRosbag) -> None:
    behavior.tick_once()
    assert behavior.stop_client.call_async.call_count == 1

    behavior.shutdown()
    assert behavior.stop_client.call_async.call_count == 2


def test_shutdown_without_setup() -> None:
    RecordRosbag(name="record", child=py_trees.behaviours.Success(name="child")).shutdown()
