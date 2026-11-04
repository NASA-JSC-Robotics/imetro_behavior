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


def _done_future(mocker: MockerFixture, response: Any) -> Any:
    future = mocker.Mock()
    future.done.return_value = True
    future.result.return_value = response
    return future


@pytest.fixture
def clients(mocker: MockerFixture) -> dict[str, Any]:
    """Mock recorder service clients that respond immediately."""
    clients = {name: mocker.Mock() for name in ("stop", "resume", "record")}
    for name, client in clients.items():
        response = Record.Response(return_code=0) if name == "record" else mocker.Mock()
        client.call_async.return_value = _done_future(mocker, response)
    return clients


@pytest.fixture
def child() -> py_trees.behaviour.Behaviour:
    """The child of the decorator, which tests can override by parametrizing `child`."""
    return py_trees.behaviours.Success(name="child")


@pytest.fixture
def output_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "bag"


@pytest.fixture
def behavior(
    child: py_trees.behaviour.Behaviour, clients: dict[str, Any], ros_node: Node, output_path: Path
) -> RecordRosbag:
    behavior = RecordRosbag(name="record", child=child, service_timeout=0.01)
    behavior.setup_ports()
    behavior.node = ros_node
    behavior._clients = clients
    Blackboard.set(behavior._get_blackboard_key("output_path"), str(output_path))
    return behavior


def _send_stop_and_resume(behavior: RecordRosbag) -> None:
    """Tick through the requests before recording, after which the next tick starts the child."""
    for _ in range(2):
        behavior.tick_once()
        assert behavior.status == Status.RUNNING


@pytest.mark.parametrize(
    "child",
    [py_trees.behaviours.StatusQueue(name="child", queue=[Status.RUNNING, Status.SUCCESS], eventually=Status.SUCCESS)],
)
def test_starts_bag_before_child_and_stops_after(
    behavior: RecordRosbag, child: py_trees.behaviour.Behaviour, clients: dict[str, Any], output_path: Path
) -> None:
    _send_stop_and_resume(behavior)
    assert child.status == Status.INVALID
    assert output_path.parent.is_dir()
    clients["stop"].call_async.assert_called_once()
    clients["resume"].call_async.assert_called_once()
    clients["record"].call_async.assert_not_called()

    behavior.tick_once()
    assert clients["record"].call_async.call_args.args[0].uri == str(output_path)
    assert behavior.status == Status.RUNNING
    assert child.status == Status.RUNNING
    clients["stop"].call_async.assert_called_once()

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert clients["stop"].call_async.call_count == 2


@pytest.mark.parametrize("child", [py_trees.behaviours.Failure(name="child")])
def test_stops_recording_when_child_fails(behavior: RecordRosbag, clients: dict[str, Any]) -> None:
    _send_stop_and_resume(behavior)
    behavior.tick_once()
    assert behavior.status == Status.FAILURE
    assert clients["stop"].call_async.call_count == 2


@pytest.mark.parametrize("child", [py_trees.behaviours.Running(name="child")])
def test_stops_recording_when_interrupted(
    behavior: RecordRosbag, child: py_trees.behaviour.Behaviour, clients: dict[str, Any]
) -> None:
    _send_stop_and_resume(behavior)
    behavior.tick_once()
    behavior.stop(Status.INVALID)
    assert child.status == Status.INVALID
    assert clients["stop"].call_async.call_count == 2


def test_fails_without_ticking_child_if_record_is_rejected(
    mocker: MockerFixture, behavior: RecordRosbag, child: py_trees.behaviour.Behaviour, clients: dict[str, Any]
) -> None:
    clients["record"].call_async.return_value = _done_future(
        mocker, Record.Response(return_code=1, error_string="Bag directory already exists")
    )

    _send_stop_and_resume(behavior)
    behavior.tick_once()
    assert behavior.status == Status.FAILURE
    assert child.status == Status.INVALID
    clients["stop"].call_async.assert_called_once()


def test_fails_if_recorder_is_not_running(
    behavior: RecordRosbag, child: py_trees.behaviour.Behaviour, clients: dict[str, Any], ros_node: Node
) -> None:
    clients["stop"].service_is_ready.return_value = False

    behavior.tick_once()
    assert behavior.status == Status.RUNNING

    ros_node.get_clock().sleep_for(Duration(seconds=0.02))
    behavior.tick_once()
    assert behavior.status == Status.FAILURE
    assert child.status == Status.INVALID
