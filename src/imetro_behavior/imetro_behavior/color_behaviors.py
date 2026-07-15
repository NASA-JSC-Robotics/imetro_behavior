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

import cv2
from cv_bridge import CvBridge

from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts, PortInformation

from color_blob_centroid.bindings import BlobRequest, process_blobs

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image


class DetectColorBlobs(BehaviourWithPorts):
    """Detects blob positions based on RGB and depth images."""

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "camera_info": PortInformation(data_type=CameraInfo, required=True),
            "rgb_image": PortInformation(data_type=Image, required=True),
            "depth_image": PortInformation(data_type=Image, required=True),
            "target_color": PortInformation(data_type=str, required=True),
            "min_blob_size": PortInformation(data_type=float, required=True),
            "debug_viz": PortInformation(data_type=bool, required=False),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {
            "blob_pose": PortInformation(data_type=PoseStamped, required=True),
            "masked_image": PortInformation(data_type=Image, required=True),
        }

    def setup(self, **kwargs):
        """Sets up the ROS node just to have access to a logger."""
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

    def update(self) -> Status:
        """Perform blob detection."""

        request = BlobRequest()
        request.blob_color = self.get_input("target_color")
        request.min_blob_size = self.get_input("min_blob_size")
        request.desired_blob = 0
        request.set_color_img(self.get_input("rgb_image"))
        request.set_depth_img(self.get_input("depth_image"))
        request.set_camera_info(self.get_input("camera_info"))

        result = process_blobs(request)
        if not result.success:
            self.node.get_logger().error(f"Failed to detect blobs: {result.err_msg}")
            return Status.FAILURE

        if self.get_input("debug_viz", False):
            # NOTE: There is an issue with running this outside the main thread where this window cannot
            # be displayed twice. So use the `debug_viz` option sparingly until/unless this is fixed.
            annotated_image = CvBridge().imgmsg_to_cv2(result.get_color_img(), desired_encoding="bgr8")
            cv2.imshow("Annotated Image (press any key to continue)", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self._set_output("blob_pose", result.get_centroid_pose())
        self._set_output("masked_image", result.get_mask())
        return Status.SUCCESS
