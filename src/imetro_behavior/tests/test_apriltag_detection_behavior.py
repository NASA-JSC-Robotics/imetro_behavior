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
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import pytest
import cv2

from rclpy.node import Node
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from imetro_behavior.apriltag_detection_behavior import DetectAprilTag


@pytest.fixture()
def apriltag_behavior(ros_node: Node):
    """Fixture to initialize the DetectAprilTag behavior and register its ports."""
    behavior = DetectAprilTag(name="TestAprilTagBehavior")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    # Load actual test image from test data folder and convert it to a ROS Image message
    image_path = (
        "/home/er4-user/ws/src/external/imetro_behavior/src/imetro_behavior/tests/test_data/test_apriltag_image.png"
    )
    cv_image = cv2.imread(image_path)
    img_msg = behavior.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8") if cv_image is not None else Image()

    # Provide camera calibration matching the image context so pose math succeeds
    mock_camera_info = CameraInfo()
    mock_camera_info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]

    # Populate blackboard inputs using native py_trees Blackboard API via _get_blackboard_key
    Blackboard.set(behavior._get_blackboard_key("rgb_image"), img_msg)
    Blackboard.set(behavior._get_blackboard_key("camera_info"), mock_camera_info)
    Blackboard.set(behavior._get_blackboard_key("tag_id"), 0)
    Blackboard.set(behavior._get_blackboard_key("tag_size"), 0.073)

    return behavior


def test_initial_state(apriltag_behavior):
    """Checks that the behavior initializes with the expected default states."""
    assert apriltag_behavior.status == Status.INVALID


def test_detect_apriltag_success(apriltag_behavior) -> None:
    """Runs the real detector on the actual image file and verifies successful pose extraction."""
    apriltag_behavior.tick_once()

    assert apriltag_behavior.status == Status.SUCCESS
    assert isinstance(apriltag_behavior.get_last_output("tag_pose"), PoseStamped)


def test_detect_apriltag_failure(apriltag_behavior) -> None:
    """Simulates a failed detection by looking for a non-existent tag ID in the image."""
    # Search for a tag ID that doesn't exist in the image to trigger failure
    Blackboard.set(apriltag_behavior._get_blackboard_key("tag_id"), 92)

    apriltag_behavior.tick_once()
    assert apriltag_behavior.status == Status.FAILURE
