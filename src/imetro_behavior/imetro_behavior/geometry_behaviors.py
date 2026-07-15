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

import copy

import numpy as np
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from py_trees.common import Access, Status
from py_trees.ports import BehaviourWithPorts, PortInformation

from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs


class CreatePoseStamped(BehaviourWithPorts):
    """Create a PoseStamped ROS message."""

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "position_xyz": PortInformation(data_type=list[float], required=True),
            "orientation_wxyz": PortInformation(data_type=list[float], required=True),
            "frame": PortInformation(data_type=str, required=False),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"msg": PortInformation(data_type=PoseStamped, required=True)}

    def update(self) -> Status:
        """Create the message and set it as an output port."""
        msg = PoseStamped()
        position_xyz = self.get_input("position_xyz")
        orientation_wxyz = self.get_input("orientation_wxyz")
        msg.pose.position.x = position_xyz[0]
        msg.pose.position.y = position_xyz[1]
        msg.pose.position.z = position_xyz[2]
        msg.pose.orientation.w = orientation_wxyz[0]
        msg.pose.orientation.x = orientation_wxyz[1]
        msg.pose.orientation.y = orientation_wxyz[2]
        msg.pose.orientation.z = orientation_wxyz[3]
        # TODO: Update this when default values are added to PyTrees ports.
        msg.header.frame_id = self.get_input("frame", "")
        self._set_output("msg", msg)
        return Status.SUCCESS


class TransformPose(BehaviourWithPorts):
    """Transforms a PoseStamped ROS message to a specified frame."""

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "input_pose": PortInformation(data_type=PoseStamped, required=True),
            "source_frame": PortInformation(data_type=str, required=True),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"output_pose": PortInformation(data_type=PoseStamped, required=True)}

    def setup(self, **kwargs):
        """Get access to the TF buffer."""
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

        self.blackboard_client.register_key(key="ros/tf_buffer", access=Access.READ)
        self.tf_buffer = self.blackboard_client.get("ros/tf_buffer")

    def update(self) -> Status:
        """Look up the transform in TF and transform the frame."""
        input_pose = self.get_input("input_pose")
        source_frame = self.get_input("source_frame")
        try:
            tform = self.tf_buffer.lookup_transform(source_frame, input_pose.header.frame_id, Time())
        except Exception as e:
            self.node.get_logger().error(f"TF lookup failed: {e}")
            return Status.FAILURE

        output_pose = PoseStamped()
        output_pose.header.frame_id = source_frame
        output_pose.pose = tf2_geometry_msgs.do_transform_pose(input_pose.pose, tform)
        self._set_output("output_pose", output_pose)
        return Status.SUCCESS


class AlignPoseToNearestAxis(BehaviourWithPorts):
    """Align a PoseStamped ROS message to the nearest axis (X Y Z)."""

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {"input_pose": PortInformation(data_type=PoseStamped, required=True)}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"output_pose": PortInformation(data_type=PoseStamped, required=True)}

    def update(self) -> Status:
        """Create the message and set it as an output port."""
        input_pose = copy.deepcopy(self.get_input("input_pose"))
        q = input_pose.pose.orientation
        r_current = R.from_quat([q.x, q.y, q.z, q.w])  # scipy uses xyzw

        # Extract the current local Z-axis vector and find which global axis it is closest to.
        current_z_axis = r_current.as_matrix()[:, 2]
        closest_axis_idx = np.argmax(np.abs(current_z_axis))
        target_axis = np.zeros(3)
        target_axis[closest_axis_idx] = np.sign(current_z_axis[closest_axis_idx])

        # Calculate the minimal rotation to snap current Z to the target axis
        axis = np.cross(current_z_axis, target_axis)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(current_z_axis, target_axis), -1.0, 1.0))
            r_align = R.from_rotvec(axis * angle)
            r_aligned = r_align * r_current
        else:
            # Already closely enough aligned to a global axis
            r_aligned = r_current

        q_aligned = r_aligned.as_quat()  # scipy uses xyzw

        aligned_pose = PoseStamped()
        aligned_pose.header = input_pose.header
        aligned_pose.pose.position = input_pose.pose.position
        aligned_pose.pose.orientation.x = q_aligned[0]
        aligned_pose.pose.orientation.y = q_aligned[1]
        aligned_pose.pose.orientation.z = q_aligned[2]
        aligned_pose.pose.orientation.w = q_aligned[3]
        self._set_output("output_pose", aligned_pose)
        return Status.SUCCESS


class OffsetPoseStamped(BehaviourWithPorts):
    """Offset a PoseStamped ROS message based on input translation and rotation offsets."""

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "input_pose": PortInformation(data_type=PoseStamped, required=True),
            "translation_xyz": PortInformation(data_type=list[float], required=False),
            "orientation_wxyz": PortInformation(data_type=list[float], required=False),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"output_pose": PortInformation(data_type=PoseStamped, required=True)}

    def update(self) -> Status:
        """Offset the pose message."""
        msg = copy.deepcopy(self.get_input("input_pose"))

        # Translation offset can be applied simply by adding it.
        translation_xyz = self.get_input("translation_xyz", [0.0, 0.0, 0.0])
        msg.pose.position.x += translation_xyz[0]
        msg.pose.position.y += translation_xyz[1]
        msg.pose.position.z += translation_xyz[2]

        # Orientation offset must be applied with quaternion multiplication.
        # Note that SciPy uses xyzw notation!
        orientation_wxyz = self.get_input("orientation_wxyz", [1.0, 0.0, 0.0, 0.0])
        rot_cur = R.from_quat(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        )
        rot_offset = R.from_quat([orientation_wxyz[1], orientation_wxyz[2], orientation_wxyz[3], orientation_wxyz[0]])
        q_new = (rot_offset * rot_cur).as_quat()
        msg.pose.orientation.x = q_new[0]
        msg.pose.orientation.y = q_new[1]
        msg.pose.orientation.z = q_new[2]
        msg.pose.orientation.w = q_new[3]

        self._set_output("output_pose", msg)
        return Status.SUCCESS
