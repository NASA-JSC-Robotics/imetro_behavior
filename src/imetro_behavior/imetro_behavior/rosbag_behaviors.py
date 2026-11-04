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
        service_timeout: Seconds to wait for the recorder to start recording.

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

        self.stop_client = None
        self.resume_client = None
        self.record_client = None

        self.output_path = None
        self.stop_future = None
        self.resume_future = None
        self.record_future = None
        self.start_time = None
        self.recording = False

    def setup(self, **kwargs: Any) -> None:
        """Create the recorder service clients."""
        set_ros_node(self, **kwargs)
        self.stop_client = self.node.create_client(Stop, f"{self.recorder_name}/stop")
        self.resume_client = self.node.create_client(Resume, f"{self.recorder_name}/resume")
        self.record_client = self.node.create_client(Record, f"{self.recorder_name}/record")

    def initialise(self) -> None:
        """Get ready to start a new bag."""
        self.output_path = Path(self.get_input("output_path")).expanduser().absolute()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_future = None
        self.resume_future = None
        self.record_future = None
        self.start_time = self.node.get_clock().now()
        self.recording = False

    def tick(self) -> Iterator[Behaviour]:
        """Tick the child if recording, otherwise start a new bag, sending each request after the previous one."""
        if self.status != Status.RUNNING:
            self.initialise()
        self.status = Status.RUNNING

        if self.recording:
            yield from super().tick()
            return

        if self.node.get_clock().now() - self.start_time > self.service_timeout:
            self.logger.error(f"Timed out waiting for recorder {self.recorder_name} to start recording.")
            self.stop(Status.FAILURE)
            yield self
            return

        if not self.stop_client.service_is_ready():
            yield self
            return

        # A recorder that was just launched has already opened an empty bag of its own, so stop it first.
        if self.stop_future is None:
            self.stop_future = self.stop_client.call_async(Stop.Request())
        if not self.stop_future.done():
            yield self
            return

        # Resuming before recording means the bag gets latched messages, such as static transforms.
        if self.resume_future is None:
            self.resume_future = self.resume_client.call_async(Resume.Request())
        if not self.resume_future.done():
            yield self
            return

        if self.record_future is None:
            self.record_future = self.record_client.call_async(Record.Request(uri=str(self.output_path)))
        if not self.record_future.done():
            yield self
            return

        response = self.record_future.result()
        if response.return_code != 0:
            self.logger.error(f"Could not start recording: {response.error_string}")
            self.record_future = None
            self.stop(Status.FAILURE)
            yield self
            return

        self.logger.info(f"Recording to {self.output_path}.")
        self.recording = True
        yield self

    def update(self) -> Status:
        """Reflect the child's status."""
        return self.decorated.status

    def terminate(self, new_status: Status) -> None:
        """Stop the recording. There is no need to wait, as requests to the recorder are handled in order."""
        if self.record_future is not None:
            self.stop_client.call_async(Stop.Request())
            self.record_future = None
            self.logger.info("Stopped recording.")

    def shutdown(self) -> None:
        """Stop any recording still in progress and clean up the service clients."""
        self.terminate(Status.INVALID)
        for client in (self.stop_client, self.resume_client, self.record_client):
            if client is not None:
                self.node.destroy_client(client)
