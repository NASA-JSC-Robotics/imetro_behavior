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

import message_filters
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

import py_trees
from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts, PortInformation


class GetSyncedImagePointCloudDepth(BehaviourWithPorts):
    def __init__(
        self,
        name: str,
        *,
        camera_info_topic: str,
        rgb_image_topic: str,
        depth_image_topic: str,
        point_cloud_topic: str,
        queue_size: int = 10,
        time_slop: float = 0.1,
        sync_timeout: float | None = None,
        **kwargs: Any,
    ):
        """
        Constructs a behavior to get synchronized camera info, images, and point clouds.

        Args:
            name: The name of the behavior (required by PyTrees).
            camera_info_topic: The name of the camera info topic.
            rgb_image_topic: The name of the RGB image topic.
            depth_image_topic: The name of the depth image topic.
            point_cloud_topic: The name of the point cloud topic.
            queue_size: The size of the queue in the message synchronizer.
            time_stop: The time slop, in seconds, in the message synchronizer.
            sync_timeout: Timeout, in seconds, to wait for the synchronized data.
                If None, waits indefinitely.
            kwargs: Additional keyword arguments to pass through to ports.
        """
        super().__init__(name, **kwargs)
        self.camera_info_topic = camera_info_topic
        self.rgb_image_topic = rgb_image_topic
        self.depth_image_topic = depth_image_topic
        self.point_cloud_topic = point_cloud_topic
        self.queue_size = queue_size
        self.time_slop = time_slop
        self.sync_timeout = Duration(seconds=sync_timeout) if sync_timeout else None

        self.synchronizer = None
        self.camera_info_sub = None
        self.rgb_image_sub = None
        self.depth_image_sub = None
        self.point_cloud_sub = None

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {
            "camera_info": PortInformation(data_type=CameraInfo, required=True),
            "rgb_image": PortInformation(data_type=Image, required=True),
            "depth_image": PortInformation(data_type=Image, required=True),
            "point_cloud": PortInformation(data_type=PointCloud2, required=True),
        }

    def setup(self, **kwargs):
        """
        Sets up the ROS node needed for the message filters.
        """
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

    def initialise(self) -> None:
        """
        Reset internal variables and create subscribers and synchronizers.
        """
        self.latest_data = None

        self.camera_info_sub = message_filters.Subscriber(self.node, CameraInfo, self.camera_info_topic)
        self.rgb_image_sub = message_filters.Subscriber(self.node, Image, self.rgb_image_topic)
        self.depth_image_sub = message_filters.Subscriber(self.node, Image, self.depth_image_topic)
        self.point_cloud_sub = message_filters.Subscriber(self.node, PointCloud2, self.point_cloud_topic)

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.camera_info_sub, self.rgb_image_sub, self.depth_image_sub, self.point_cloud_sub],
            queue_size=self.queue_size,
            slop=self.time_slop,
        )
        self.synchronizer.registerCallback(self._synchronize_callback)
        self.start_time = self.node.get_clock().now()

    def _synchronize_callback(
        self, camera_info_msg: CameraInfo, rgb_image_msg: Image, depth_image_msg: Image, point_cloud_msg: PointCloud2
    ) -> None:
        """Callback triggered only when all topics have synchronized headers."""
        self.latest_data = (camera_info_msg, rgb_image_msg, depth_image_msg, point_cloud_msg)

    def update(self) -> Status:
        """
        Executes every time the behavior tree ticks this node.
        """
        if self.latest_data is not None:
            self.node.get_logger().info(f"[{self.qualified_name}] Got synchronized images and point clouds!")
            camera_info_msg, rgb_image_msg, depth_image_msg, point_cloud_msg = self.latest_data
            self._set_output("camera_info", camera_info_msg)
            self._set_output("rgb_image", rgb_image_msg)
            self._set_output("depth_image", depth_image_msg)
            self._set_output("point_cloud", point_cloud_msg)

            # Clear cache so we don't process the exact same frame on the next tick.
            self.latest_data = None

            return py_trees.common.Status.SUCCESS

        # If no synchronized frame has arrived yet, keep waiting until timeout.
        if self.sync_timeout is not None and self.node.get_clock().now() - self.start_time > self.sync_timeout:
            self.node.get_logger().error(f"[{self.qualified_name}] Timed out waiting for synchronization.")
            return Status.FAILURE

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        """Cleanup if the behavior is interrupted or completes."""
        if self.synchronizer is not None:
            self.synchronizer.callbacks.clear()

        for subscription in (self.camera_info_sub, self.rgb_image_sub, self.depth_image_sub, self.point_cloud_sub):
            self.node.destroy_subscription(subscription.sub)
