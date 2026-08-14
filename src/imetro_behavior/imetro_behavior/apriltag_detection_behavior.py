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

import time

from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts, PortInformation

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge


class DetectAprilTag(BehaviourWithPorts):
    """Detects AprilTag based on RGB images and camera info, then returns the tag pose."""

    @classmethod
    def input_ports(cls) -> dict:
        return {
            "rgb_image": PortInformation(data_type=Image, required=True),
            "camera_info": PortInformation(data_type=CameraInfo, required=True),
            "tag_id": PortInformation(data_type=int, required=True),
            "tag_size": PortInformation(data_type=float, required=True),
            "tag_family": PortInformation(data_type=str, required=True),
        }

    @classmethod
    def output_ports(cls) -> dict:
        return {
            "tag_pose": PortInformation(data_type=PoseStamped, required=True),
        }

    def setup(self, **kwargs):
        """Sets up the ROS node, detector, and bridge."""
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

        self.bridge = CvBridge()
        self.detector = None

    def update(self) -> Status:
        """Run AprilTag detection."""
        rgb_msg = self.get_input("rgb_image")
        camera_info = self.get_input("camera_info")
        target_id = self.get_input("tag_id")
        tag_size = self.get_input("tag_size")
        tag_family = self.get_input("tag_family")

        if rgb_msg is None or camera_info is None:
            return Status.FAILURE

        if self.detector is None:
            self.detector = Detector(families=tag_family, nthreads=2, quad_decimate=2.0)

        start_time = time.perf_counter()

        image_det_start = time.perf_counter()
        cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="mono8")
        fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
        camera_params = (fx, fy, cx, cy)
        image_det_time = time.perf_counter() - image_det_start

        detection_start = time.perf_counter()
        tags = self.detector.detect(cv_img, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
        detection_time = time.perf_counter() - detection_start

        target_tag = next((t for t in tags if t.tag_id == target_id), None)

        if target_tag is None:
            return Status.FAILURE

        pose_msg = PoseStamped()
        pose_msg.header = rgb_msg.header
        pose_msg.pose.position.x = target_tag.pose_t[0][0]
        pose_msg.pose.position.y = target_tag.pose_t[1][0]
        pose_msg.pose.position.z = target_tag.pose_t[2][0]

        r = R.from_matrix(target_tag.pose_R)
        q = r.as_quat()
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        self.node.get_logger().info(
            f"Found AprilTag ID {target_id}! Position -> "
            f"x: {pose_msg.pose.position.x:.3f}, "
            f"y: {pose_msg.pose.position.y:.3f}, "
            f"z: {pose_msg.pose.position.z:.3f}"
        )
        self.node.get_logger().info(f"Image conversion took {image_det_time:.4f} seconds!")
        self.node.get_logger().info(f"Apriltag detection took {detection_time:.4f} seconds!")

        total_time = time.perf_counter() - start_time
        self.node.get_logger().info(f"Apriltag detection total process took {total_time:.4f} seconds!")

        self._set_output("tag_pose", pose_msg)
        return Status.SUCCESS
