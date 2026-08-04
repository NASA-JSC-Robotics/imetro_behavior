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

import pytest

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from py_trees.blackboard import Blackboard
from py_trees.common import Status


class FakeBlobRequest:
    """Stands in for the color_blob_centroid BlobRequest binding."""

    def __init__(self):
        self.blob_color = None
        self.min_blob_size = None
        self.desired_blob = None

    def set_color_img(self, img: Image) -> None:
        self.color_img = img

    def set_depth_img(self, img: Image) -> None:
        self.depth_img = img

    def set_camera_info(self, info: CameraInfo) -> None:
        self.camera_info = info


class FakeBlobResult:
    """Stands in for the color_blob_centroid BlobResult binding."""

    def __init__(self, success: bool):
        self.success = success
        self.err_msg = "" if success else "no blobs found"

    def get_centroid_pose(self) -> PoseStamped:
        return PoseStamped()

    def get_mask(self) -> Image:
        return Image()


@pytest.fixture()
def color_behaviors():
    """The color behaviors module, or a test skip if its dependencies are unavailable.

    The skip must happen at fixture time rather than module import time, since the
    launch_testing pytest plugin imports test modules during collection, and a
    module-level skip aborts collection of the remaining test files.
    """
    return pytest.importorskip("imetro_behavior.color_behaviors", reason="requires OpenCV and color_blob_centroid")


@pytest.fixture()
def blob_behavior(color_behaviors, ros_node: Node, mocker):
    mocker.patch.object(color_behaviors, "BlobRequest", FakeBlobRequest)

    behavior = color_behaviors.DetectColorBlobs(name="detect_blobs")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    Blackboard.set(behavior._get_blackboard_key("camera_info"), CameraInfo())
    Blackboard.set(behavior._get_blackboard_key("rgb_image"), Image())
    Blackboard.set(behavior._get_blackboard_key("depth_image"), Image())
    Blackboard.set(behavior._get_blackboard_key("target_color"), "red")
    Blackboard.set(behavior._get_blackboard_key("min_blob_size"), 30.0)
    return behavior


def test_detect_color_blobs_success(color_behaviors, blob_behavior, mocker) -> None:
    mock_process = mocker.patch.object(color_behaviors, "process_blobs", return_value=FakeBlobResult(success=True))

    blob_behavior.tick_once()
    assert blob_behavior.status == Status.SUCCESS
    assert isinstance(blob_behavior.get_last_output("blob_pose"), PoseStamped)
    assert isinstance(blob_behavior.get_last_output("masked_image"), Image)

    # The request should have been populated from the input ports.
    request = mock_process.call_args.args[0]
    assert request.blob_color == "red"
    assert request.min_blob_size == 30.0


def test_detect_color_blobs_failure(color_behaviors, blob_behavior, mocker) -> None:
    mocker.patch.object(color_behaviors, "process_blobs", return_value=FakeBlobResult(success=False))

    blob_behavior.tick_once()
    assert blob_behavior.status == Status.FAILURE
