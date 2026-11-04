"""Behaviors for recording rosbags."""

from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.decorators import Decorator
from py_trees.ports import PortInformation, PortsMixin
from rclpy.duration import Duration
from rclpy.node import Node
from rosbag2_interfaces.srv import Record, Resume, Stop

from imetro_behavior.helpers import set_ros_node


class RecordRosbag(PortsMixin, Decorator):
    """
    Records a rosbag for as long as the decorated child runs.

    This controls a dormant recorder through its services, which must already be running:

        ros2 bag record --start-paused --node-name rosbag_recorder --topics ...

    The child only starts once the recorder is recording, and if that fails the child is not ticked.
    The recording always stops when this behavior ends, including if the child fails or is interrupted.
    The status of this behavior is the status of the child, regardless of the recorder.

    Constructor arguments:
        recorder_name: The name of the recorder node.
        service_timeout: Seconds to wait for each recorder service to respond.

    Input ports:
        output_path: The bag directory to create, which must not already exist.
    """

    INPUT_PORTS = {"output_path": PortInformation(data_type=str, required=True)}
    OUTPUT_PORTS = {}

    def __init__(
        self,
        name: str,
        child: Behaviour,
        *,
        recorder_name: str = "/rosbag_recorder",
        service_timeout: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(name=name, child=child, **kwargs)
        self.recorder_name = recorder_name
        self.service_timeout = Duration(seconds=service_timeout)
        self.node: Node

    def setup(self, **kwargs: Any) -> None:
        """Create the recorder service clients."""
        set_ros_node(self, **kwargs)
        self._clients = {
            "stop": self.node.create_client(Stop, f"{self.recorder_name}/stop"),
            "resume": self.node.create_client(Resume, f"{self.recorder_name}/resume"),
            "record": self.node.create_client(Record, f"{self.recorder_name}/record"),
        }

    def initialise(self) -> None:
        """Queue up the recorder services that start a new bag."""
        # A recorder that was just launched has already opened an empty bag of its own, so stop it first.
        # Resuming before recording means the bag gets latched messages, such as static transforms.
        output_path = Path(self.get_input("output_path")).expanduser().absolute()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._requests = deque(
            [
                ("stop", Stop.Request()),
                ("resume", Resume.Request()),
                ("record", Record.Request(uri=str(output_path))),
            ]
        )
        self._service_name = ""
        self._future = None
        self._start_time = None
        self._recording = False
        self._record_requested = False

    def tick(self) -> Iterator[Behaviour]:
        """Start the recorder, and tick the child once it is recording."""
        if self.status != Status.RUNNING:
            self.initialise()

        if not self._recording:
            status = self._start_recording()
            if status == Status.FAILURE:
                self.stop(status)
                yield self
                return
            if not self._recording:
                self.status = status
                yield self
                return

        yield from super().tick()

    def update(self) -> Status:
        """Reflect the child's status."""
        return self.decorated.status

    def terminate(self, new_status: Status) -> None:
        """Stop the recording. There is no need to wait, as requests to the recorder are handled in order."""
        if self._record_requested:
            self._clients["stop"].call_async(Stop.Request())
            self._record_requested = False
            self.logger.info("Stopped recording.")

    def _start_recording(self) -> Status:
        """Advance through the requests that start the recorder."""
        if self._future is None:
            self._service_name, request = self._requests[0]
            if not self._clients[self._service_name].service_is_ready():
                return self._check_timeout(
                    f"Recorder service {self.recorder_name}/{self._service_name} is not available."
                )
            self._future = self._clients[self._service_name].call_async(request)
            self._start_time = self.node.get_clock().now()
            self._record_requested = self._record_requested or self._service_name == "record"

        if not self._future.done():
            return self._check_timeout(f"Recorder service {self.recorder_name}/{self._service_name} timed out.")

        response = self._future.result()
        if self._service_name == "record" and response.return_code != 0:
            self.logger.error(f"Could not start recording: {response.error_string}")
            self._record_requested = False
            return Status.FAILURE

        self._requests.popleft()
        self._future = None
        self._start_time = None
        self._recording = not self._requests
        if self._recording:
            self.logger.info(f"Recording to {self.get_input('output_path')}.")
        return Status.RUNNING

    def _check_timeout(self, message: str) -> Status:
        """Fail if the current service has been waited on for too long."""
        if self._start_time is None:
            self._start_time = self.node.get_clock().now()
        if self.node.get_clock().now() - self._start_time > self.service_timeout:
            self.logger.error(message)
            return Status.FAILURE
        return Status.RUNNING
