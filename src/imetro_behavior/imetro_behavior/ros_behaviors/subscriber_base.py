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

from rclpy.qos import QoSPresetProfiles
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
    Base class for behaviors that subscribe to a single topic.
    """

    def __init__(
        self,
        name: str,
        topic_type: Type,
        *,
        topic_name: str,
        subscriber_timeout: float = 3.0,
        qos_profile: str = "default",
        **kwargs: Any,
    ):
        """
        Constructs a ROS subscriber.

        Args:
            name: The name of the behavior (required by PyTrees)
            topic_type: The ROS interface type of the topic.
            topic_name: The name of the ROS topic to send a request to.
            subscriber_timeout: Timeout, in seconds, to wait to receive a message from the topic.
                If None, waits indefinitely.
            qos_profile: String name for the quality of service profile for the subscriber.
                The options can be found here https://github.com/ros2/rclpy/blob/jazzy/rclpy/rclpy/qos.py#L483.
            kwargs: Additional keyword arguments to pass through to ports.
        """
        super().__init__(name, **kwargs)
        self.topic_type = topic_type
        self.topic_name = topic_name
        self.subscriber_timeout = Duration(seconds=subscriber_timeout) if subscriber_timeout else None
        if qos_profile not in QoSPresetProfiles.short_keys() or qos_profile.lower() == "unknown":
            raise KeyError(f"QoSProfile [ {qos_profile} ] is not available!")
        self.qos_profile = QoSPresetProfiles.get_from_short_key(qos_profile)

        self.latest_msg = None

    def setup(self, **kwargs):
        """
        Get the node from the blackboard.
        """
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

    def initialise(self):
        """
        Reset the internal variables.
        """
        self.subscription = self.node.create_subscription(
            self.topic_type, self.topic_name, self.callback, self.qos_profile
        )
        self.start_time = self.node.get_clock().now()

    def callback(self, msg) -> None:
        self.latest_msg = msg

    def update(self):
        """
        Monitor latest_msg variable until it is set or the timer runs out.
        """
        if self.latest_msg is not None:
            self.node.get_logger().debug(f"[{self.qualified_name}] Got topic message!")
            self._set_output("message", self.latest_msg)

            self.latest_msg = None
            return Status.SUCCESS

        # If no message has arrived yet, keep waiting until timeout.
        if (
            self.subscriber_timeout is not None
            and self.node.get_clock().now() - self.start_time > self.subscriber_timeout
        ):
            self.node.get_logger().error(f"[{self.qualified_name}] Timed out waiting on topic {self.topic_name}.")
            return Status.FAILURE

        return Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        """
        Cleanup up the subscribers and the latest message if switching to INVALID status.
        """
        if self.status == Status.RUNNING and new_status == Status.INVALID:
            self.node.destroy_subscription(self.subscription)
        self.latest_msg = None


class GetStringTopic(RosSubscriberBase):
    """
    Subscribes to a string topic.
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
