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

from typing import Any

from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts, PortInformation
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import String

from imetro_behavior.helpers import set_ros_node


class RosPublisherBase(BehaviourWithPorts):
    """
    Base class for behaviors that publish a single message on a single topic.
    """

    def __init__(
        self,
        name: str,
        topic_type: type,
        *,
        topic_name: str,
        qos_profile: str = "default",
        **kwargs: Any,
    ):
        """
        Constructs a ROS publisher.

        Args:
            name: The name of the behavior (required by PyTrees)
            topic_type: The ROS interface type of the topic.
            topic_name: The name of the ROS topic to publish a message to.
            qos_profile: String name for the quality of service profile for the publisher.
            kwargs: Additional keyword arguments to pass through to ports.
        """
        super().__init__(name, **kwargs)
        self.topic_type = topic_type
        self.topic_name = topic_name
        if qos_profile not in QoSPresetProfiles.short_keys() or qos_profile.lower() == "unknown":
            raise KeyError(f"QoSProfile '{qos_profile}' is not available!")
        self.qos_profile = QoSPresetProfiles.get_from_short_key(qos_profile)

        self.node: Node

    def setup(self, **kwargs):
        """
        Get the node from the blackboard.
        """
        set_ros_node(self, **kwargs)

    def initialise(self):
        """
        Reset the internal variables.
        """
        self.publisher = self.node.create_publisher(self.topic_type, self.topic_name, self.qos_profile)

    def update(self):
        """
        Sets the message to be published and then publishes it.
        """
        try:
            message = self.create_message()
        except Exception as e:  # noqa: BLE001 -- we don't know what the user will raise
            self.logger.error(f"Failed to create message: {e}")
            return Status.FAILURE

        self.publisher.publish(message)
        return Status.SUCCESS

    def terminate(self, new_status: Status) -> None:
        """
        Clean up the publisher if switching to INVALID status.
        """
        if self.status == Status.RUNNING and new_status == Status.INVALID:
            self.node.destroy_publisher(self.publisher)

    def create_message(self) -> Any:
        """
        Abstract method for setting the message to publish.
        """
        raise NotImplementedError("Must implement set_message() method.")


class PublishString(RosPublisherBase):
    """
    Publishes a string message to a given topic.
    This is a standard enough behavior that we keep it as part of the core library.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, topic_type=String, **kwargs)

    INPUT_PORTS = {"string_data": PortInformation(data_type=str, required=True)}

    OUTPUT_PORTS = {}

    def create_message(self) -> Any:
        return String(data=self.get_input("string_data"))
