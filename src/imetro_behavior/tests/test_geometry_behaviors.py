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

import numpy as np
import pytest
from pathlib import Path

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, TransformStamped
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rclpy.node import Node
from tf2_ros import Buffer

from imetro_behavior.geometry_behaviors import (
    AlignPoseToNearestAxis,
    CreatePoseStamped,
    LookupTransform,
    OffsetPoseStamped,
    PublishTransform,
    TransformPose,
    YamlPoseToPoseStamped,
    PoseStampedToTransformStamped,
    TransformStampedToPoseStamped,
    TwistAboutPose,
    GetRelativePoseStamped,
    GetRollPitchYaw,
    DecomposePoseStamped,
)


@pytest.fixture()
def tf_buffer() -> Buffer:
    """A TF buffer containing a single 'base' frame at x=1.0 in the 'map' frame, set on the blackboard."""
    buffer = Buffer()
    tform = TransformStamped()
    tform.header.frame_id = "map"
    tform.child_frame_id = "base"
    tform.transform.translation.x = 1.0
    tform.transform.rotation.w = 1.0
    buffer.set_transform_static(tform, "test_authority")

    Blackboard.set("/ros/tf_buffer", buffer)
    return buffer


def make_pose(position_xyz: list[float], orientation_xyzw: list[float], frame_id: str = "") -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position_xyz
    (
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z,
        pose.pose.orientation.w,
    ) = orientation_xyzw
    return pose


def get_z_axis(pose: PoseStamped) -> np.ndarray:
    q = pose.pose.orientation
    return R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 2]


def test_create_pose_stamped() -> None:
    behavior = CreatePoseStamped(name="create_pose")
    behavior.setup_ports()

    Blackboard.set(behavior._get_blackboard_key("position_xyz"), [1.0, 2.0, 3.0])
    Blackboard.set(behavior._get_blackboard_key("orientation_xyzw"), [0.0, 0.0, 0.0, 1.0])
    Blackboard.set(behavior._get_blackboard_key("frame"), "map")

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    msg = behavior.get_last_output("msg")
    assert msg.header.frame_id == "map"
    assert msg.pose.position.x == 1.0
    assert msg.pose.position.y == 2.0
    assert msg.pose.position.z == 3.0
    assert msg.pose.orientation.w == 1.0


def test_create_pose_stamped_default_frame() -> None:
    behavior = CreatePoseStamped(name="create_pose")
    behavior.setup_ports()

    Blackboard.set(behavior._get_blackboard_key("position_xyz"), [1.0, 2.0, 3.0])
    Blackboard.set(behavior._get_blackboard_key("orientation_xyzw"), [0.0, 0.0, 0.0, 1.0])

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert behavior.get_last_output("msg").header.frame_id == ""


def test_transform_pose(ros_node: Node, tf_buffer: Buffer) -> None:
    behavior = TransformPose(name="transform_pose")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    input_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], frame_id="map")
    input_pose.header.stamp.sec = 10
    input_pose.header.stamp.nanosec = 15
    Blackboard.set(behavior._get_blackboard_key("input_pose"), input_pose)
    Blackboard.set(behavior._get_blackboard_key("source_frame"), "base")

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    output_pose = behavior.get_last_output("output_pose")
    assert output_pose.header.frame_id == "base"
    assert output_pose.header.stamp.nanosec == 15
    assert output_pose.header.stamp.sec == 10
    # The base frame sits at x=1.0 in map, so the map origin is at x=-1.0 in base.
    assert output_pose.pose.position.x == pytest.approx(-1.0)


def test_transform_pose_unknown_frame(ros_node: Node, tf_buffer: Buffer) -> None:
    behavior = TransformPose(name="transform_pose")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    input_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], frame_id="map")
    Blackboard.set(behavior._get_blackboard_key("input_pose"), input_pose)
    Blackboard.set(behavior._get_blackboard_key("source_frame"), "nonexistent_frame")

    behavior.tick_once()
    assert behavior.status == Status.FAILURE


def test_align_pose_to_nearest_axis_snaps_to_z() -> None:
    behavior = AlignPoseToNearestAxis(name="align_pose")
    behavior.setup_ports()

    # Tilt 15 degrees about Y, so the local Z axis is still closest to global +Z.
    quat = R.from_euler("y", 15.0, degrees=True).as_quat()
    input_pose = make_pose([1.0, 2.0, 3.0], list(quat), frame_id="map")
    Blackboard.set(behavior._get_blackboard_key("input_pose"), input_pose)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    output_pose = behavior.get_last_output("output_pose")
    assert output_pose.header.frame_id == "map"
    assert output_pose.pose.position == input_pose.pose.position
    assert get_z_axis(output_pose) == pytest.approx([0.0, 0.0, 1.0])


def test_align_pose_to_nearest_axis_snaps_to_x() -> None:
    behavior = AlignPoseToNearestAxis(name="align_pose")
    behavior.setup_ports()

    # Tilt 80 degrees about Y, so the local Z axis is closest to global +X.
    quat = R.from_euler("y", 80.0, degrees=True).as_quat()
    input_pose = make_pose([0.0, 0.0, 0.0], list(quat))
    Blackboard.set(behavior._get_blackboard_key("input_pose"), input_pose)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    assert get_z_axis(behavior.get_last_output("output_pose")) == pytest.approx([1.0, 0.0, 0.0])


def test_align_pose_to_nearest_axis_already_aligned() -> None:
    behavior = AlignPoseToNearestAxis(name="align_pose")
    behavior.setup_ports()

    input_pose = make_pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    Blackboard.set(behavior._get_blackboard_key("input_pose"), input_pose)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    output_pose = behavior.get_last_output("output_pose")
    assert output_pose.pose.orientation == input_pose.pose.orientation


def test_offset_pose_stamped_translation() -> None:
    behavior = OffsetPoseStamped(name="offset_pose")
    behavior.setup_ports()

    input_pose = make_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], frame_id="map")
    Blackboard.set(behavior._get_blackboard_key("input_pose"), input_pose)
    Blackboard.set(behavior._get_blackboard_key("translation_xyz"), [0.5, -0.5, 1.0])

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    output_pose = behavior.get_last_output("output_pose")
    assert output_pose.pose.position.x == pytest.approx(1.5)
    assert output_pose.pose.position.y == pytest.approx(1.5)
    assert output_pose.pose.position.z == pytest.approx(4.0)
    assert output_pose.pose.orientation == input_pose.pose.orientation
    # The input pose must not be modified in place.
    assert input_pose.pose.position.x == 1.0


def test_offset_pose_stamped_orientation() -> None:
    behavior = OffsetPoseStamped(name="offset_pose")
    behavior.setup_ports()

    input_pose = make_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    Blackboard.set(behavior._get_blackboard_key("input_pose"), input_pose)
    quat_offset = R.from_euler("z", 90.0, degrees=True).as_quat()
    Blackboard.set(behavior._get_blackboard_key("orientation_xyzw"), list(quat_offset))

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    output_pose = behavior.get_last_output("output_pose")
    assert output_pose.pose.position == input_pose.pose.position
    assert output_pose.pose.orientation.z == pytest.approx(np.sin(np.pi / 4.0))
    assert output_pose.pose.orientation.w == pytest.approx(np.cos(np.pi / 4.0))


def test_offset_pose_stamped_no_offsets() -> None:
    behavior = OffsetPoseStamped(name="offset_pose")
    behavior.setup_ports()

    input_pose = make_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    Blackboard.set(behavior._get_blackboard_key("input_pose"), input_pose)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    output_pose = behavior.get_last_output("output_pose")
    assert output_pose.pose.position == input_pose.pose.position
    assert output_pose.pose.orientation == input_pose.pose.orientation


@pytest.fixture()
def yaml_pose_behavior(ros_node: Node, tmp_path: Path, mocker) -> YamlPoseToPoseStamped:
    """A YamlPoseToPoseStamped behavior whose package share directory resolves to a temporary path."""
    mocker.patch(
        "imetro_behavior.geometry_behaviors.get_package_share_path",
        return_value=tmp_path,
    )
    (tmp_path / "poses.yaml").write_text("""
grasp_pose:
    frame_id: "world"
    pose:
        position:
            x: 1.0
            y: 2.0
            z: 3.0
        orientation:
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
""")

    behavior = YamlPoseToPoseStamped(name="yaml_pose")
    behavior.setup(node=ros_node)
    behavior.setup_ports()
    Blackboard.set(behavior._get_blackboard_key("package_name"), "some_package")
    Blackboard.set(behavior._get_blackboard_key("yaml_file"), "poses.yaml")
    return behavior


def test_yaml_pose_to_pose_stamped(yaml_pose_behavior: YamlPoseToPoseStamped) -> None:
    Blackboard.set(yaml_pose_behavior._get_blackboard_key("pose_name"), "grasp_pose")

    yaml_pose_behavior.tick_once()
    assert yaml_pose_behavior.status == Status.SUCCESS
    msg = yaml_pose_behavior.get_last_output("msg")
    assert msg.header.frame_id == "world"
    assert msg.pose.position.x == 1.0
    assert msg.pose.position.y == 2.0
    assert msg.pose.position.z == 3.0
    assert msg.pose.orientation.w == 1.0


def test_yaml_pose_to_pose_stamped_missing_pose(yaml_pose_behavior: YamlPoseToPoseStamped) -> None:
    Blackboard.set(yaml_pose_behavior._get_blackboard_key("pose_name"), "nonexistent_pose")
    yaml_pose_behavior.tick_once()
    assert yaml_pose_behavior.status == Status.FAILURE


def test_yaml_pose_to_pose_stamped_missing_file(yaml_pose_behavior: YamlPoseToPoseStamped) -> None:
    Blackboard.set(yaml_pose_behavior._get_blackboard_key("yaml_file"), "nonexistent.yaml")
    Blackboard.set(yaml_pose_behavior._get_blackboard_key("pose_name"), "grasp_pose")
    yaml_pose_behavior.tick_once()
    assert yaml_pose_behavior.status == Status.FAILURE


def test_lookup_transform(ros_node: Node, tf_buffer: Buffer) -> None:
    behavior = LookupTransform(name="lookup_transform")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    Blackboard.set(behavior._get_blackboard_key("source_frame"), "map")
    Blackboard.set(behavior._get_blackboard_key("target_frame"), "base")

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    tform = behavior.get_last_output("target_T_source")
    assert tform.header.frame_id == "base"
    assert tform.child_frame_id == "map"
    assert tform.transform.translation.x == pytest.approx(-1.0)


def test_lookup_transform_unknown_frame(ros_node: Node, tf_buffer: Buffer) -> None:
    behavior = LookupTransform(name="lookup_transform")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    Blackboard.set(behavior._get_blackboard_key("source_frame"), "map")
    Blackboard.set(behavior._get_blackboard_key("target_frame"), "nonexistent_frame")

    behavior.tick_once()
    assert behavior.status == Status.FAILURE


def test_publish_transform(ros_node: Node) -> None:
    behavior = PublishTransform(name="publish_transform")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    tform = TransformStamped()
    tform.header.frame_id = "map"
    tform.child_frame_id = "object"
    tform.transform.rotation.w = 1.0
    Blackboard.set(behavior._get_blackboard_key("transform_stamped"), tform)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS


def test_transform_stamped_to_pose_stamped() -> None:
    behavior = TransformStampedToPoseStamped(name="tf_to_pose")
    behavior.setup_ports()

    t_msg = TransformStamped()
    t_msg.header.frame_id = "map"
    t_msg.child_frame_id = "base"
    t_msg.transform.translation.x = 2.0
    t_msg.transform.translation.y = 3.0
    t_msg.transform.translation.z = 4.0
    t_msg.transform.rotation.w = 1.0

    Blackboard.set(behavior._get_blackboard_key("transform_stamped"), t_msg)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS

    pose_msg = behavior.get_last_output("pose_stamped")
    assert pose_msg.header.frame_id == "map"
    assert pose_msg.pose.position.x == 2.0
    assert pose_msg.pose.position.y == 3.0
    assert pose_msg.pose.position.z == 4.0
    assert pose_msg.pose.orientation.w == 1.0

    child_frame = behavior.get_last_output("child_frame_id")
    assert child_frame == "base"


def test_pose_stamped_to_transform_stamped() -> None:
    behavior = PoseStampedToTransformStamped(name="pose_to_tf")
    behavior.setup_ports()

    p_msg = make_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], frame_id="odom")

    Blackboard.set(behavior._get_blackboard_key("pose_stamped"), p_msg)
    Blackboard.set(behavior._get_blackboard_key("child_frame_id"), "base_footprint")

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS

    t_msg = behavior.get_last_output("transform_stamped")
    assert t_msg.header.frame_id == "odom"
    assert t_msg.child_frame_id == "base_footprint"
    assert t_msg.transform.translation.x == 1.0
    assert t_msg.transform.translation.y == 2.0
    assert t_msg.transform.translation.z == 3.0
    assert t_msg.transform.rotation.w == 1.0


def test_twist_about_pose() -> None:
    """Tests basic functionality of twist by rotating about a center circle 90 degrees"""
    behavior = TwistAboutPose(name="twist_about_pose")
    behavior.setup_ports()

    target_pose = make_pose([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], frame_id="world")

    Blackboard.set(behavior._get_blackboard_key("target_pose"), target_pose)
    Blackboard.set(behavior._get_blackboard_key("rotation_axis"), [0.0, 0.0, 1.0])
    Blackboard.set(behavior._get_blackboard_key("rotation_amount"), 1.57080)
    Blackboard.set(behavior._get_blackboard_key("keep_start_orientation"), False)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    msg = behavior.get_last_output("output_pose")

    assert msg.header.frame_id == "world"
    assert msg.pose.position.x == pytest.approx(0.0, abs=1e-5)
    assert msg.pose.position.y == pytest.approx(1.0, abs=1e-5)
    assert msg.pose.position.z == pytest.approx(0.0, abs=1e-5)
    assert msg.pose.orientation.z == pytest.approx(0.7071, abs=1e-5)
    assert msg.pose.orientation.w == pytest.approx(0.7071, abs=1e-5)


def test_twist_about_pose_keep_start_orientation() -> None:
    """Tests the same as above, however with keep orientation flag set to True"""
    behavior = TwistAboutPose(name="twist_about_pose")
    behavior.setup_ports()

    target_pose = make_pose([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], frame_id="world")

    Blackboard.set(behavior._get_blackboard_key("target_pose"), target_pose)
    Blackboard.set(behavior._get_blackboard_key("rotation_axis"), [0.0, 0.0, 1.0])
    Blackboard.set(behavior._get_blackboard_key("rotation_amount"), 1.57080)
    Blackboard.set(behavior._get_blackboard_key("keep_start_orientation"), True)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    msg = behavior.get_last_output("output_pose")

    assert msg.header.frame_id == "world"
    assert msg.pose.position.x == pytest.approx(0.0, abs=1e-5)
    assert msg.pose.position.y == pytest.approx(1.0, abs=1e-5)
    assert msg.pose.position.z == pytest.approx(0.0, abs=1e-5)
    assert msg.pose.orientation.x == 0.0
    assert msg.pose.orientation.y == 0.0
    assert msg.pose.orientation.z == 0.0
    assert msg.pose.orientation.w == 1.0


def test_get_relative_pose(ros_node: Node) -> None:
    """Tests a relative pose from base to target"""
    behavior = GetRelativePoseStamped(name="get_relative_pose")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    base_pose = make_pose([-1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], frame_id="world")
    target_pose = make_pose([1.0, 0.0, 0.0], [0.0, 0.0, 0.7071068, 0.7071068], frame_id="world")

    Blackboard.set(behavior._get_blackboard_key("base_pose"), base_pose)
    Blackboard.set(behavior._get_blackboard_key("base_frame_name"), "rotation_frame")
    Blackboard.set(behavior._get_blackboard_key("target_pose"), target_pose)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    msg = behavior.get_last_output("output_pose")
    assert msg.pose.position.x == pytest.approx(2.0, abs=1e-5)
    assert msg.pose.position.y == pytest.approx(0.0, abs=1e-5)
    assert msg.pose.position.z == pytest.approx(0.0, abs=1e-5)
    assert msg.pose.orientation.x == 0.0
    assert msg.pose.orientation.y == 0.0
    assert msg.pose.orientation.z == pytest.approx(0.7071068, abs=1e-5)
    assert msg.pose.orientation.w == pytest.approx(0.7071068, abs=1e-5)


def test_get_relative_pose_mismatch_frame(ros_node: Node) -> None:
    """Tests for mismatched reference frames between two input poses"""
    behavior = GetRelativePoseStamped(name="get_relative_pose")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    base_pose = make_pose([1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], frame_id="world")
    target_pose = make_pose([0.0, 2.0, 4.0], [0.0, 0.0, 0.0, 1.0], frame_id="grasp_frame")

    Blackboard.set(behavior._get_blackboard_key("base_pose"), base_pose)
    Blackboard.set(behavior._get_blackboard_key("base_frame_name"), "rotation_frame")
    Blackboard.set(behavior._get_blackboard_key("target_pose"), target_pose)

    behavior.tick_once()
    assert behavior.status == Status.FAILURE


def test_get_rpy(ros_node: Node) -> None:
    """Tests for roll pitch yaw of with no rotation"""
    behavior = GetRollPitchYaw(name="twist_about_pose")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    input_pose = make_pose([1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], frame_id="world")

    Blackboard.set(behavior._get_blackboard_key("input_pose"), input_pose)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    roll = behavior.get_last_output("roll")
    pitch = behavior.get_last_output("pitch")
    yaw = behavior.get_last_output("yaw")

    assert roll == pytest.approx(0.0, abs=1e-5)
    assert pitch == pytest.approx(0.0, abs=1e-5)
    assert yaw == pytest.approx(0.0, abs=1e-5)


def test_get_rpy_90z(ros_node: Node) -> None:
    """Tests for roll pitch yaw of with a 90 degree rotation about z"""
    behavior = GetRollPitchYaw(name="twist_about_pose")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    input_pose = make_pose([1.0, 1.0, 1.0], [0.0, 0.0, 0.7071068, 0.7071068], frame_id="world")

    Blackboard.set(behavior._get_blackboard_key("input_pose"), input_pose)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS
    roll = behavior.get_last_output("roll")
    pitch = behavior.get_last_output("pitch")
    yaw = behavior.get_last_output("yaw")

    assert roll == pytest.approx(0.0, abs=1e-5)
    assert pitch == pytest.approx(0.0, abs=1e-5)
    assert yaw == pytest.approx(1.57079, abs=1e-5)


def test_decompose_pose_stamped() -> None:
    behavior = DecomposePoseStamped(name="decompose_pose_stamped")
    behavior.setup_ports()

    p_msg = make_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], frame_id="odom")

    Blackboard.set(behavior._get_blackboard_key("pose_stamped"), p_msg)

    behavior.tick_once()
    assert behavior.status == Status.SUCCESS

    frame_id_msg = behavior.get_last_output("frame_id")
    translation_msg = behavior.get_last_output("translation_xyz")
    orientation_msg = behavior.get_last_output("orientation_xyzw")
    assert frame_id_msg == "odom"
    assert translation_msg == [1.0, 2.0, 3.0]
    assert orientation_msg == [0.0, 0.0, 0.0, 1.0]
