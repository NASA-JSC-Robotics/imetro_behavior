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

from typing import Any, Type

from rclpy.duration import Duration
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts, PortInformation

from std_msgs.msg import String

QOS_LATCHING = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class RosSubscriberBase(BehaviourWithPorts):
    """
    Base class for behaviors that rely on ROS service clients.

    Modified from https://github.com/splintered-reality/py_trees_ros/blob/devel/py_trees_ros/service_clients.py
    """

    def __init__(
        self,
        name: str,
        topic_type: Type,
        *,
        topic_name: str,
        subscriber_timeout: float = 3.0,
        **kwargs: Any,
    ):
        """
        Constructs a ROS Service Client base behavior.

        Args:
            name: The name of the behavior (required by PyTrees)
            service_type: The ROS interface type of the service.
            service_name: The name of the ROS service to send a request to.
            service_server_timeout: Timeout, in seconds, to wait for the service server to be available.
                If None, waits indefinitely.
            service_timeout: Timeout, in seconds, to wait for the service to complete.
                If None, waits indefinitely.
            kwargs: Additional keyword arguments to pass through to ports.
        """
        super().__init__(name, **kwargs)
        self.topic_type = topic_type
        self.topic_name = topic_name
        self.subscriber_timeout = Duration(seconds=subscriber_timeout) if subscriber_timeout else None

        self.latest_msg = None

    def setup(self, **kwargs):
        """
        Sets up the service client.
        """
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

    def initialise(self, qos: QoSProfile = QOS_LATCHING):
        """
        Reset the internal variables.
        """
        self.subscription = self.node.create_subscription(self.topic_type, self.topic_name, self.callback, qos)
        self.start_time = self.node.get_clock().now()

    def callback(self, msg) -> None:
        self.latest_msg = msg

    def update(self):
        """
        Kick off a new service request and then check whether the service has completed or timed out.
        """
        if self.latest_msg is not None:
            self.node.get_logger().info(f"[{self.qualified_name}] Got topic message!")

            self._set_output("message", self.latest_msg)
            # Clear cache so we don't process the exact same frame on the next tick.
            self.latest_msg = None

            return Status.SUCCESS

        # If no synchronized frame has arrived yet, keep waiting until timeout.
        if (
            self.subscriber_timeout is not None
            and self.node.get_clock().now() - self.start_time > self.subscriber_timeout
        ):
            self.node.get_logger().error(f"[{self.qualified_name}] Timed out waiting for topic {self.topic_name}.")
            return Status.FAILURE

        return Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        """
        If running and the current service call has not already completed, cancel it.
        """
        if self.status == Status.RUNNING and new_status == Status.INVALID:
            self.node.destroy_subscription(self.subscription)


class GetStringTopic(RosSubscriberBase):
    """
    Sends a trigger request to a ROS service server.

    This is a standard enough behavior that we keep it as part of the core library.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, topic_type=String, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {
            "message": PortInformation(data_type=String, required=True),
        }
