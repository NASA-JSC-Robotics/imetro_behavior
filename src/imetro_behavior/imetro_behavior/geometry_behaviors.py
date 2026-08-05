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
import yaml
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from py_trees.common import Access, Status
from py_trees.ports import BehaviourWithPorts, PortInformation
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster


class CreatePoseStamped(BehaviourWithPorts):
    """Create a PoseStamped ROS message."""

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "position_xyz": PortInformation(data_type=list[float], required=True),
            "orientation_xyzw": PortInformation(data_type=list[float], required=True),
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
        orientation_xyzw = self.get_input("orientation_xyzw")
        msg.pose.position.x = position_xyz[0]
        msg.pose.position.y = position_xyz[1]
        msg.pose.position.z = position_xyz[2]
        msg.pose.orientation.x = orientation_xyzw[0]
        msg.pose.orientation.y = orientation_xyzw[1]
        msg.pose.orientation.z = orientation_xyzw[2]
        msg.pose.orientation.w = orientation_xyzw[3]
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

        self.blackboard_client.register_key(key="/ros/tf_buffer", access=Access.READ)
        self.tf_buffer = self.blackboard_client.get("/ros/tf_buffer")

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
            "orientation_xyzw": PortInformation(data_type=list[float], required=False),
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
        orientation_xyzw = self.get_input("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])
        rot_cur = R.from_quat(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
        rot_offset = R.from_quat(
            [
                orientation_xyzw[0],
                orientation_xyzw[1],
                orientation_xyzw[2],
                orientation_xyzw[3],
            ]
        )
        q_new = (rot_offset * rot_cur).as_quat()
        msg.pose.orientation.x = q_new[0]
        msg.pose.orientation.y = q_new[1]
        msg.pose.orientation.z = q_new[2]
        msg.pose.orientation.w = q_new[3]

        self._set_output("output_pose", msg)
        return Status.SUCCESS


class YamlPoseToPoseStamped(BehaviourWithPorts):
    """Load a Pose object from a YAML and output a PoseStamped ROS message

    Valid yaml configuration for a pose:

    pose_name:
        frame_id: ""
        pose:
            position:
                x: 0.0
                y: 0.0
                z: 0.0
            orientation:
                x: 0.0
                y: 0.0
                z: 0.0
                w: 1.0
    """

    @classmethod
    def input_ports(cls) -> dict:
        """Return the port declarations."""
        return {
            "package_name": PortInformation(data_type=str, required=True),
            "yaml_file": PortInformation(data_type=str, required=True),
            "pose_name": PortInformation(data_type=str, required=True),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"msg": PortInformation(data_type=PoseStamped, required=True)}

    def setup(self, **kwargs):
        """Get access to the node for error statements."""
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

    def update(self) -> Status:
        """Load the YAML, create the message, and set it as an output port."""

        yaml_path = Path(get_package_share_directory(self.get_input("package_name"))) / self.get_input("yaml_file")
        if not yaml_path.is_file():
            self.node.get_logger().error(f"File at {yaml_path} could not be found or is not a file")
            return Status.FAILURE

        with open(yaml_path) as file:
            data = yaml.safe_load(file)
        pose_name = self.get_input("pose_name")
        pose_dict = data.get(pose_name)
        if pose_dict is None:
            self.node.get_logger().error(f"Failed to find pose {pose_name} in {yaml_path}")
            return Status.FAILURE
        frame_id = pose_dict["frame_id"]
        pose = pose_dict["pose"]

        msg = PoseStamped(
            header=Header(frame_id=frame_id),
            pose=Pose(
                position=Point(
                    x=pose["position"]["x"],
                    y=pose["position"]["y"],
                    z=pose["position"]["z"],
                ),
                orientation=Quaternion(
                    x=pose["orientation"]["x"],
                    y=pose["orientation"]["y"],
                    z=pose["orientation"]["z"],
                    w=pose["orientation"]["w"],
                ),
            ),
        )

        self._set_output("msg", msg)
        return Status.SUCCESS


class LookupTransform(BehaviourWithPorts):
    """Lookup transform between two frames from the tf2_ros server.

    Returns the transform required to convert from the source frame, to the target frame.
    Namely, `target_T_source`.
    """

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "source_frame": PortInformation(data_type=str, required=True),
            "target_frame": PortInformation(data_type=str, required=True),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {
            "target_T_source": PortInformation(data_type=TransformStamped, required=True),
        }

    def setup(self, **kwargs):
        """Get access to the TF buffer."""
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

        self.blackboard_client.register_key(key="/ros/tf_buffer", access=Access.READ)
        self.tf_buffer = self.blackboard_client.get("/ros/tf_buffer")

    def update(self) -> Status:
        """Look up the transform in TF and transform the frame."""
        source_frame = self.get_input("source_frame")
        target_frame = self.get_input("target_frame")
        try:
            target_T_source = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
        except Exception as e:
            self.node.get_logger().error(f"TF lookup failed: {e}")
            return Status.FAILURE

        self._set_output("target_T_source", target_T_source)
        return Status.SUCCESS

class TransformStampedToPoseStamped(BehaviourWithPorts):
    """Convert a TransformStamped ROS message to a PoseStamped ROS message."""

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "transform_stamped": PortInformation(data_type=TransformStamped, required=True),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {
            "pose_stamped": PortInformation(data_type=PoseStamped, required=True),
        }

    def update(self) -> Status:
        """Extract the transform's translation and rotation into a PoseStamped message."""
        t_stamped = self.get_input("transform_stamped")
        
        pose_stamped = PoseStamped()
        pose_stamped.header = t_stamped.header
        pose_stamped.pose.position.x = t_stamped.transform.translation.x
        pose_stamped.pose.position.y = t_stamped.transform.translation.y
        pose_stamped.pose.position.z = t_stamped.transform.translation.z
        pose_stamped.pose.orientation = t_stamped.transform.rotation
        
        self._set_output("pose_stamped", pose_stamped)
        return Status.SUCCESS


class PoseStampedToTransformStamped(BehaviourWithPorts):
    """Convert a PoseStamped ROS message to a TransformStamped ROS message."""

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "pose_stamped": PortInformation(data_type=PoseStamped, required=True),
            "child_frame_id": PortInformation(data_type=str, required=True),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {
            "transform_stamped": PortInformation(data_type=TransformStamped, required=True),
        }

    def update(self) -> Status:
        """Extract the pose's position and orientation into a TransformStamped message."""
        pose_stamped = self.get_input("pose_stamped")
        child_frame_id = self.get_input("child_frame_id")

        t_stamped = TransformStamped()
        t_stamped.header = pose_stamped.header
        t_stamped.child_frame_id = child_frame_id
        t_stamped.transform.translation.x = pose_stamped.pose.position.x
        t_stamped.transform.translation.y = pose_stamped.pose.position.y
        t_stamped.transform.translation.z = pose_stamped.pose.position.z
        t_stamped.transform.rotation = pose_stamped.pose.orientation

        self._set_output("transform_stamped", t_stamped)
        return Status.SUCCESS

class PublishTransform(BehaviourWithPorts):
    """Publish transform stamped message to the tf2_ros server."""

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "transform_stamped": PortInformation(data_type=TransformStamped, required=True),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {}

    def setup(self, **kwargs):
        """Setup transform broadcaster."""
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")
        self.tf_broadcaster = TransformBroadcaster(self.node)

    def update(self) -> Status:
        """Publish the transform frame."""
        transform_stamped = self.get_input("transform_stamped")
        self.tf_broadcaster.sendTransform(transform_stamped)
        return Status.SUCCESS
