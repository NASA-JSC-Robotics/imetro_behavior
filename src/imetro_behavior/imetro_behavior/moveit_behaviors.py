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
from typing import Any

import xml.etree.ElementTree as ET

from py_trees.common import Access, Status
from py_trees.ports import BehaviourWithPorts, PortInformation

import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node

from rclpy.time import Time
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PositionConstraint,
    OrientationConstraint,
    RobotTrajectory,
    PlanningScene,
    CollisionObject,
    AllowedCollisionEntry,
)
from moveit_msgs.srv import GetMotionPlan, GetCartesianPath, GetPlanningScene, ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive, Mesh, MeshTriangle
from std_msgs.msg import Header

from imetro_behavior_msgs.action import PreviewTrajectory
from imetro_behavior.ros_behaviors.action_client import RosActionClientBase
from imetro_behavior.ros_behaviors.service_client import RosServiceClientBase

# Handy dictionary for reporting failures.
# Painstakingly copied from https://github.com/moveit/moveit_msgs/blob/ros2/msg/MoveItErrorCodes.msg
MOVEIT_ERROR_CODE_DICT = {
    MoveItErrorCodes.SUCCESS: "SUCCESS",
    MoveItErrorCodes.UNDEFINED: "UNDEFINED",
    MoveItErrorCodes.FAILURE: "FAILURE",
    MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
    MoveItErrorCodes.INVALID_MOTION_PLAN: "INVALID_MOTION_PLAN",
    MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    MoveItErrorCodes.CONTROL_FAILED: "CONTROL_FAILED",
    MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
    MoveItErrorCodes.PREEMPTED: "PREEMPTED",
    MoveItErrorCodes.START_STATE_IN_COLLISION: "START_STATE_IN_COLLISION",
    MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    MoveItErrorCodes.START_STATE_INVALID: "START_STATE_INVALID",
    MoveItErrorCodes.GOAL_IN_COLLISION: "GOAL_IN_COLLISION",
    MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED: "GOAL_CONSTRAINTS_VIOLATED",
    MoveItErrorCodes.GOAL_STATE_INVALID: "GOAL_STATE_INVALID",
    MoveItErrorCodes.UNRECOGNIZED_GOAL_TYPE: "UNRECOGNIZED_GOAL_TYPE",
    MoveItErrorCodes.INVALID_GROUP_NAME: "INVALID_GROUP_NAME",
    MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS: "INVALID_GOAL_CONSTRAINTS",
    MoveItErrorCodes.INVALID_ROBOT_STATE: "INVALID_ROBOT_STATE",
    MoveItErrorCodes.INVALID_LINK_NAME: "INVALID_LINK_NAME",
    MoveItErrorCodes.INVALID_OBJECT_NAME: "INVALID_OBJECT_NAME",
    MoveItErrorCodes.FRAME_TRANSFORM_FAILURE: "FRAME_TRANSFORM_FAILURE",
    MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE: "COLLISION_CHECKING_UNAVAILABLE",
    MoveItErrorCodes.ROBOT_STATE_STALE: "ROBOT_STATE_STALE",
    MoveItErrorCodes.SENSOR_INFO_STALE: "SENSOR_INFO_STALE",
    MoveItErrorCodes.COMMUNICATION_FAILURE: "COMMUNICATION_FAILURE",
    MoveItErrorCodes.CRASH: "CRASH",
    MoveItErrorCodes.ABORT: "ABORT",
    MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
}


class PlanToJointState(RosServiceClientBase):
    """
    Uses MoveIt to plan a motion to a target joint state.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=GetMotionPlan, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "pipeline": PortInformation(data_type=str, required=False),
            "planner": PortInformation(data_type=str, required=False),
            "group_name": PortInformation(data_type=str, required=True),
            "joint_names": PortInformation(data_type=list[str], required=True),
            "joint_positions": PortInformation(data_type=list[float], required=True),
            "tolerance": PortInformation(data_type=float, required=True),
            "max_velocity_scaling": PortInformation(data_type=float, required=False),
            "max_acceleration_scaling": PortInformation(data_type=float, required=False),
            "num_planning_attempts": PortInformation(data_type=int, required=False),
            "allowed_planning_time": PortInformation(data_type=float, required=False),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"trajectory": PortInformation(data_type=RobotTrajectory)}

    def create_request(self) -> GetMotionPlan.Request:
        """Create a motion plan service request with joint constraints."""
        joint_names = self.get_input("joint_names")
        joint_positions = self.get_input("joint_positions")
        tolerance = self.get_input("tolerance")
        if len(joint_positions) != len(joint_names):
            raise RuntimeError("Joint names and joint positions must have the same length.")

        request = GetMotionPlan.Request()

        goal_constraints = Constraints()
        for name, pos in zip(joint_names, joint_positions):
            goal_constraints.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=pos,
                    tolerance_above=tolerance,
                    tolerance_below=tolerance,
                    weight=1.0,
                )
            )

        request.motion_plan_request.pipeline_id = self.get_input("pipeline", "")
        request.motion_plan_request.planner_id = self.get_input("planner", "")
        request.motion_plan_request.group_name = self.get_input("group_name")
        request.motion_plan_request.max_velocity_scaling_factor = self.get_input("max_velocity_scaling", 1.0)
        request.motion_plan_request.max_acceleration_scaling_factor = self.get_input("max_acceleration_scaling", 1.0)
        request.motion_plan_request.goal_constraints = [goal_constraints]
        request.motion_plan_request.num_planning_attempts = self.get_input("num_planning_attempts", 5)
        request.motion_plan_request.allowed_planning_time = self.get_input("allowed_planning_time", 1.0)
        return request

    def process_response(self, response: GetMotionPlan.Response) -> Status:
        """Process the motion planning service response."""
        error_code = response.motion_plan_response.error_code
        if error_code.val == MoveItErrorCodes.SUCCESS:
            self.node.get_logger().info("Motion plan succeeded!")
            self._set_output("trajectory", response.motion_plan_response.trajectory)
            return Status.SUCCESS
        else:
            error_code_str = MOVEIT_ERROR_CODE_DICT.get(error_code.val, "UNKNOWN")
            self.node.get_logger().error(f"Motion plan failed with error code: {error_code_str}")
            self.node.get_logger().error(f"Message: {error_code.message}")
            self.node.get_logger().error(f"Source: {error_code.source}")
            return Status.FAILURE


class PlanToPose(RosServiceClientBase):
    """
    Uses MoveIt to plan a motion to a target pose.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=GetMotionPlan, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "pipeline": PortInformation(data_type=str, required=False),
            "planner": PortInformation(data_type=str, required=False),
            "group_name": PortInformation(data_type=str, required=True),
            "target_frame": PortInformation(data_type=str, required=True),
            "target_pose": PortInformation(data_type=PoseStamped, required=True),
            "position_tolerance": PortInformation(data_type=float, required=True),
            "orientation_tolerance": PortInformation(data_type=list[float], required=True),
            "max_velocity_scaling": PortInformation(data_type=float, required=False),
            "max_acceleration_scaling": PortInformation(data_type=float, required=False),
            "num_planning_attempts": PortInformation(data_type=int, required=False),
            "allowed_planning_time": PortInformation(data_type=float, required=False),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"trajectory": PortInformation(data_type=RobotTrajectory)}

    def create_request(self) -> GetMotionPlan.Request:
        """Create a motion plan service request with position and orientation constraints."""
        target_pose = self.get_input("target_pose")
        target_frame = self.get_input("target_frame")
        source_frame = target_pose.header.frame_id

        request = GetMotionPlan.Request()
        goal_constraints = Constraints()

        # Position goal constraint is a sphere around the target pose.
        position_constraint = PositionConstraint()
        position_constraint.constraint_region.primitives.append(
            SolidPrimitive(
                type=SolidPrimitive.SPHERE,
                dimensions=[self.get_input("position_tolerance")],
            )
        )
        position_constraint.constraint_region.primitive_poses.append(Pose(position=target_pose.pose.position))
        position_constraint.link_name = target_frame
        position_constraint.header.frame_id = source_frame
        position_constraint.weight = 1.0
        goal_constraints.position_constraints.append(position_constraint)

        # Orientation goal constraint is simpler to set up, but optional.
        orientation_tolerance = self.get_input("orientation_tolerance")
        if orientation_tolerance is not None:
            orientation_constraint = OrientationConstraint()
            orientation_constraint.header.frame_id = source_frame
            orientation_constraint.link_name = target_frame
            orientation_constraint.parameterization = OrientationConstraint.XYZ_EULER_ANGLES
            orientation_constraint.orientation = target_pose.pose.orientation
            orientation_constraint.absolute_x_axis_tolerance = orientation_tolerance[0]
            orientation_constraint.absolute_y_axis_tolerance = orientation_tolerance[1]
            orientation_constraint.absolute_z_axis_tolerance = orientation_tolerance[2]
            orientation_constraint.weight = 1.0
            goal_constraints.orientation_constraints.append(orientation_constraint)

        request.motion_plan_request.pipeline_id = self.get_input("pipeline", "")
        request.motion_plan_request.planner_id = self.get_input("planner", "")
        request.motion_plan_request.group_name = self.get_input("group_name")
        request.motion_plan_request.max_velocity_scaling_factor = self.get_input("max_velocity_scaling", 1.0)
        request.motion_plan_request.max_acceleration_scaling_factor = self.get_input("max_acceleration_scaling", 1.0)
        request.motion_plan_request.goal_constraints = [goal_constraints]
        request.motion_plan_request.num_planning_attempts = self.get_input("num_planning_attempts", 5)
        request.motion_plan_request.allowed_planning_time = self.get_input("allowed_planning_time", 1.0)
        return request

    def process_response(self, response: GetMotionPlan.Response) -> Status:
        """Process the motion plan service response."""
        error_code = response.motion_plan_response.error_code
        if error_code.val == MoveItErrorCodes.SUCCESS:
            self.node.get_logger().info("Motion plan succeeded!")
            self._set_output("trajectory", response.motion_plan_response.trajectory)
            return Status.SUCCESS
        else:
            error_code_str = MOVEIT_ERROR_CODE_DICT.get(error_code.val, "UNKNOWN")
            self.node.get_logger().error(f"Motion plan failed with error code: {error_code_str}")
            self.node.get_logger().error(f"Message: {error_code.message}")
            self.node.get_logger().error(f"Source: {error_code.source}")
            return Status.FAILURE


class RequestPlanningScene(RosServiceClientBase):
    """
    Get planning scene of the robot. Useful for modifying the Allowed Collision Matrix.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=GetPlanningScene, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"planning_scene": PortInformation(data_type=PlanningScene)}

    def create_request(self) -> GetPlanningScene.Request:
        """Create a PlanningScene service request."""
        request = GetPlanningScene.Request()
        return request

    def process_response(self, response: GetPlanningScene.Response) -> Status:
        """Process the PlanningScene service response."""
        planning_scene = response.scene
        if planning_scene:
            self.node.get_logger().info("Got planning scene!")
            self._set_output("planning_scene", planning_scene)
            return Status.SUCCESS
        else:
            self.node.get_logger().error("Error: failed to get planning scene.")
            return Status.FAILURE


class ModifyCollisions(RosServiceClientBase):
    """
    Modifies the Allowed Collision Matrix to allow certain links of the robot to collide with other objects/links.
    The Allowed Collision Matrix is a matrix of pairs of links/objects that are allowed/disallowed to collide.
    Note: collision flag will be applied for all combinations of given link lists.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=ApplyPlanningScene, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "planning_scene": PortInformation(data_type=PlanningScene, required=True),
            "links_list_1": PortInformation(
                data_type=list[str],
                required=True,
                description="First list of robot links or scene objects for which to modify collisions",
            ),
            "links_list_2": PortInformation(
                data_type=list[str],
                required=True,
                description="Second list of robot links or scene objects to modify collisions against links_list_1",
            ),
            "allow_collision": PortInformation(
                data_type=bool,
                required=True,
                description="True: links are able to collide with each other."
                "False: collision is forbidden between links.",
            ),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {}

    def create_request(self) -> ApplyPlanningScene.Request:
        """Create an ApplyPlanningScene service request to modify the ACM."""
        request = ApplyPlanningScene.Request()
        planning_scene = self.get_input("planning_scene")
        links_list_1 = self.get_input("links_list_1")
        links_list_2 = self.get_input("links_list_2")
        allow_collision = self.get_input("allow_collision")

        acm = copy.deepcopy(planning_scene.allowed_collision_matrix)

        # Manually update ACM, add links if they dont already exist in the ACM
        all_links = set(list(links_list_1) + list(links_list_2))
        for link in all_links:
            if link not in acm.entry_names:
                acm.entry_names.append(link)

        num_entries = len(acm.entry_names)
        num_new_entries_needed = num_entries - len(acm.entry_values)

        if num_new_entries_needed > 0:
            # Extend the list of existing enables for the existing ACM entries.
            for entry in acm.entry_values:
                entry.enabled.extend([False] * num_new_entries_needed)

            # Create brand new all-false entries for the new entries added.
            acm.entry_values.extend(
                [AllowedCollisionEntry(enabled=[False] * num_entries) for _ in range(num_new_entries_needed)]
            )

        # Finally, apply the input port collision flag to the given link lists
        # Note: when allow_collision is True, links are able to collide with each other
        for link1 in links_list_1:
            index1 = acm.entry_names.index(link1)
            for link2 in links_list_2:
                index2 = acm.entry_names.index(link2)

                acm.entry_values[index1].enabled[index2] = allow_collision
                acm.entry_values[index2].enabled[index1] = allow_collision

        planning_scene.allowed_collision_matrix = acm
        planning_scene.is_diff = True
        request.scene = planning_scene
        return request

    def process_response(self, response: ApplyPlanningScene.Response) -> Status:
        """Process the ApplyPlanningScene service response."""
        if response.success:
            self.node.get_logger().info("Successfully modified the planning scene!")
            return Status.SUCCESS
        else:
            self.node.get_logger().error("Error: failed to apply modifications to planning scene.")
            return Status.FAILURE


class PlanCartesian(RosServiceClientBase):
    """
    Uses MoveIt to plan a motion to a target pose using cartesian path planning.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=GetCartesianPath, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "group_name": PortInformation(data_type=str, required=True),
            "waypoints": PortInformation(
                data_type=PoseStamped, required=True
            ),  # TODO support TransformStamped and support list of waypoints
            "max_step": PortInformation(data_type=float, required=False),
            "jump_threshold": PortInformation(data_type=float, required=False),
            "avoid_collisions": PortInformation(data_type=bool, required=False),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"trajectory": PortInformation(data_type=RobotTrajectory)}

    def create_request(self) -> GetCartesianPath.Request:
        """Create a cartesian path service request."""
        request = GetCartesianPath.Request()

        waypoints = self.get_input("waypoints")
        if isinstance(waypoints, PoseStamped):
            request.waypoints = [waypoints.pose]
        else:  # let's assume it's a list
            request.waypoints = waypoints

        request.group_name = self.get_input("group_name")
        request.max_step = self.get_input("max_step", 0.01)
        request.jump_threshold = self.get_input("jump_threshold", 1.25)
        request.avoid_collisions = self.get_input("avoid_collisions", True)
        return request

    def process_response(self, response: GetCartesianPath.Response) -> Status:
        """Process the cartesian path service response."""
        error_code = response.error_code
        if error_code.val == MoveItErrorCodes.SUCCESS:
            self.node.get_logger().info("Cartesian plan succeeded!")
            self._set_output("trajectory", response.solution)
            return Status.SUCCESS
        else:
            error_code_str = MOVEIT_ERROR_CODE_DICT.get(error_code.val, "UNKNOWN")
            self.node.get_logger().error(
                f"Cartesian plan failed with error code: {error_code_str}, computed {response.fraction}% of trajectory."
            )
            self.node.get_logger().error(f"Message: {error_code.message}")
            self.node.get_logger().error(f"Source: {error_code.source}")
            return Status.FAILURE


class PlanArcPath(RosServiceClientBase):
    """
    Uses MoveIt's Pilz CIRC tool to plan an arc motion path.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=GetMotionPlan, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "fixed_frame": PortInformation(data_type=str, required=True),
            "group_name": PortInformation(data_type=str, required=True),
            "target_frame": PortInformation(data_type=str, required=True),
            "rotation_pose": PortInformation(data_type=PoseStamped, required=True),
            "rotation_axis_xyz": PortInformation(data_type=list[float], required=True),
            "rotation_amount_rad": PortInformation(data_type=float, required=True),
            "keep_start_orientation": PortInformation(data_type=bool, required=True),
            "position_tolerance": PortInformation(data_type=float, required=True),
            "orientation_tolerance_xyz": PortInformation(data_type=list[float], required=True),
            "max_velocity_scaling": PortInformation(data_type=float, required=True),
            "max_acceleration_scaling": PortInformation(data_type=float, required=True),
            "planning_attempt_count": PortInformation(data_type=int, required=False),
            "planning_attempt_timeout": PortInformation(data_type=float, required=False),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"trajectory": PortInformation(data_type=RobotTrajectory)}

    def setup(self, **kwargs):
        """Get access to the TF buffer."""
        super().setup(**kwargs)

        self.blackboard_client.register_key(key="/ros/tf_buffer", access=Access.READ)
        self.tf_buffer = self.blackboard_client.get("/ros/tf_buffer")

    def rotate_about_frame(self):
        """
        Calculate the center and final pose of an arc path given a target_frame to plan from, a rotation_pose and
        rotation_axis to rotate about, and a rotation_amount

        Input ports:
            target_frame: Frame that the arc is planned from
            rotation_pose: Pose which the target_frame will be rotated about
            rotation_axis: Axis of rotation_pose that is rotated about
            rotation_amount: Amount of rotation to apply to target_frame about the rotation axis, in radians
            keep_start_orientation: If true, final_pose will have the same orientation as target_frame

        Returns:
            final_pose: End pose of the arc
            center_point: Center point of the arc

        """

        # Get input ports
        fixed_frame = self.get_input("fixed_frame")
        target_frame = self.get_input("target_frame")
        rotation_pose = self.get_input("rotation_pose")
        rotation_axis = np.array(self.get_input("rotation_axis_xyz"))
        rotation_amount = self.get_input("rotation_amount_rad")
        keep_start_orientation = self.get_input("keep_start_orientation")

        # Get target_frame
        target_tf = self.tf_buffer.lookup_transform(fixed_frame, target_frame, Time())
        rot_target = target_tf.transform.rotation
        q_target = R.from_quat([rot_target.x, rot_target.y, rot_target.z, rot_target.w])
        trans_target = target_tf.transform.translation
        p_target = np.array([trans_target.x, trans_target.y, trans_target.z])

        # Get rotation frame
        rot_rotation = rotation_pose.pose.orientation
        q_rotation = R.from_quat([rot_rotation.x, rot_rotation.y, rot_rotation.z, rot_rotation.w])
        trans_rotation = rotation_pose.pose.position
        p_rotation = np.array([trans_rotation.x, trans_rotation.y, trans_rotation.z])

        # Normalize rotation axis, transformed relative to fixed frame/world
        rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_axis_in_fixed = q_rotation.apply(rotation_axis_normalized)

        # Find center point relative to fixed frame
        p_rotation_to_target = p_target - p_rotation
        p_rotation_to_target_rotaxis_projection = np.dot(p_rotation_to_target, rotation_axis_in_fixed)
        p_center = p_rotation + p_rotation_to_target_rotaxis_projection * rotation_axis_in_fixed

        # Get rotation operator
        rotation = R.from_rotvec(rotation_axis_in_fixed * rotation_amount)

        # Get final pose
        p_center_to_target = p_target - p_center
        p_final = p_center + rotation.apply(p_center_to_target)
        if keep_start_orientation:
            q_final = q_target
        else:
            q_final = rotation * q_target

        final_pose = Pose()
        final_pose.position.x = p_final[0]
        final_pose.position.y = p_final[1]
        final_pose.position.z = p_final[2]
        quat_final = q_final.as_quat()
        final_pose.orientation.x = quat_final[0]
        final_pose.orientation.y = quat_final[1]
        final_pose.orientation.z = quat_final[2]
        final_pose.orientation.w = quat_final[3]

        center_point = Point(x=p_center[0], y=p_center[1], z=p_center[2])

        return final_pose, center_point

    def create_request(self) -> GetMotionPlan.Request:
        """Create a motion plan service request with position and orientation constraints."""
        target_pose, center_point = self.rotate_about_frame()
        fixed_frame = self.get_input("fixed_frame")
        target_frame = self.get_input("target_frame")
        request = GetMotionPlan.Request()
        request.motion_plan_request.pipeline_id = "pilz_industrial_motion_planner"
        request.motion_plan_request.planner_id = "CIRC"
        request.motion_plan_request.group_name = self.get_input("group_name")
        request.motion_plan_request.num_planning_attempts = self.get_input("planning_attempt_count", 10)
        request.motion_plan_request.allowed_planning_time = self.get_input("planning_attempt_timeout", 15.0)
        request.motion_plan_request.max_velocity_scaling_factor = self.get_input("max_velocity_scaling", 1.0)
        request.motion_plan_request.max_acceleration_scaling_factor = self.get_input("max_acceleration_scaling", 0.1)

        goal_constraints = Constraints()

        goal_position_constraint = PositionConstraint()
        goal_position_constraint.header.frame_id = fixed_frame
        goal_position_constraint.link_name = target_frame

        goal_bounding_volume = BoundingVolume()
        goal_bounding_volume.primitives = [
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[self.get_input("position_tolerance")])
        ]
        goal_bounding_volume.primitive_poses = [
            Pose(
                position=Point(x=target_pose.position.x, y=target_pose.position.y, z=target_pose.position.z),
                orientation=Quaternion(w=1.0),
            )
        ]
        goal_position_constraint.constraint_region = goal_bounding_volume
        goal_position_constraint.weight = 1.0
        goal_constraints.position_constraints.append(goal_position_constraint)

        goal_orientation_constraint = OrientationConstraint()
        goal_orientation_constraint.header.frame_id = fixed_frame
        goal_orientation_constraint.link_name = target_frame
        goal_orientation_constraint.parameterization = OrientationConstraint.XYZ_EULER_ANGLES
        goal_orientation_constraint.orientation = target_pose.orientation
        orientation_tolerance = self.get_input("orientation_tolerance_xyz")
        goal_orientation_constraint.absolute_x_axis_tolerance = orientation_tolerance[0]
        goal_orientation_constraint.absolute_y_axis_tolerance = orientation_tolerance[1]
        goal_orientation_constraint.absolute_z_axis_tolerance = orientation_tolerance[2]
        goal_orientation_constraint.weight = 1.0
        goal_constraints.orientation_constraints.append(goal_orientation_constraint)
        goal_constraints.joint_constraints = []
        request.motion_plan_request.goal_constraints = [goal_constraints]

        path_constraints = Constraints()
        path_constraints.name = "center"
        path_position_constraint = PositionConstraint()
        path_position_constraint.header.frame_id = fixed_frame
        path_position_constraint.link_name = target_frame
        center_bounding_volume = BoundingVolume()
        center_bounding_volume.primitives = [
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[self.get_input("position_tolerance")])
        ]
        center_bounding_volume.primitive_poses = [
            Pose(
                position=Point(x=center_point.x, y=center_point.y, z=center_point.z),
                orientation=Quaternion(w=1.0),
            )
        ]
        path_position_constraint.constraint_region = center_bounding_volume
        path_position_constraint.weight = 1.0
        path_constraints.position_constraints.append(path_position_constraint)
        request.motion_plan_request.path_constraints = path_constraints
        return request

    def process_response(self, response: GetMotionPlan.Response) -> Status:
        """Process the motion plan service response."""
        error_code = response.motion_plan_response.error_code
        if error_code.val == MoveItErrorCodes.SUCCESS:
            self.node.get_logger().info("Motion plan succeeded!")
            self._set_output("trajectory", response.motion_plan_response.trajectory)
            return Status.SUCCESS
        else:
            error_code_str = MOVEIT_ERROR_CODE_DICT.get(error_code.val, "UNKNOWN")
            self.node.get_logger().error(f"Motion plan failed with error code: {error_code_str}")
            self.node.get_logger().error(f"Message: {error_code.message}")
            self.node.get_logger().error(f"Source: {error_code.source}")
            return Status.FAILURE


class RequestTrajectoryApproval(RosActionClientBase):
    """Sends a planned trajectory to the behavior GUI for preview and approval."""

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, action_type=PreviewTrajectory, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {"trajectory": PortInformation(data_type=RobotTrajectory, required=True)}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {"approved": PortInformation(data_type=bool, required=True)}

    def create_goal(self) -> PreviewTrajectory.Goal:
        """Create a trajectory preview goal."""
        return PreviewTrajectory.Goal(trajectory=self.get_input("trajectory"))

    def process_result(self, result: PreviewTrajectory.Result) -> Status:
        """Process the trajectory preview action result."""
        approved = result.approved
        self._set_output("approved", approved)
        if approved:
            self.node.get_logger().info("Trajectory preview approved by user.")
            return Status.SUCCESS
        else:
            self.node.get_logger().error("Trajectory preview rejected by user.")
            return Status.FAILURE


class ExecuteTrajectoryBehavior(RosActionClientBase):
    """Sends an action goal to MoveIt to execute a robot trajectory."""

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, action_type=ExecuteTrajectory, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {"trajectory": PortInformation(data_type=RobotTrajectory, required=True)}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {}

    def create_goal(self) -> ExecuteTrajectory.Goal:
        """Create a trajectory execution goal."""
        return ExecuteTrajectory.Goal(trajectory=self.get_input("trajectory"))

    def process_result(self, result: ExecuteTrajectory.Result) -> Status:
        """Process the trajectory execution action result."""
        error_code = result.error_code
        if error_code.val == MoveItErrorCodes.SUCCESS:
            self.node.get_logger().info("Trajectory execution succeeded!")
            return Status.SUCCESS
        else:
            error_code_str = MOVEIT_ERROR_CODE_DICT.get(error_code.val, "UNKNOWN")
            self.node.get_logger().error(f"Trajectory execution failed with error code: {error_code_str}")
            self.node.get_logger().error(f"Message: {error_code.message}")
            self.node.get_logger().error(f"Source: {error_code.source}")
            return Status.FAILURE


class SetPlanningScene(RosServiceClientBase):
    """
    Use MoveIt apply_planning_scene service to change the planning scene.
    Useful if you have and intermediate step of changing the planning scene message.
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=ApplyPlanningScene, **kwargs)

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {"planning_scene": PortInformation(data_type=PlanningScene)}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {}

    def create_request(self) -> ApplyPlanningScene.Request:
        """Create a PlanningScene service request."""
        request = ApplyPlanningScene.Request()
        request.scene = self.get_input("planning_scene")
        return request

    def process_response(self, response: ApplyPlanningScene.Response) -> Status:
        """Process the ApplyPlanningScene service response."""
        if response.success:
            self.node.get_logger().info("Successfully modified the planning scene!")
            return Status.SUCCESS
        else:
            self.node.get_logger().error("Error: failed to apply modifications to planning scene.")
            return Status.FAILURE


class PlanningSceneFromRobotDescription(BehaviourWithPorts):
    """
    Parse the robot_description string, extract all of the collision from the links,
    convert them collision objects, and insert into the planning scene.
    Input ports:
        planning_scene: Planning message to append collision objects to.
        robot_description: std_msgs/msg/String that contains a URDF.

    Returns:
        modified_planning_scene: Planning scene with inserted collision objects.
    """

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {
            "planning_scene": PortInformation(data_type=PlanningScene, required=True),
            "robot_description": PortInformation(data_type=str, required=True),
        }

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {
            "modified_planning_scene": PortInformation(data_type=PlanningScene, required=True),
        }

    def setup(self, **kwargs):
        """Get access to the TF buffer."""
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")

    def update(self) -> Status:
        """Look up the transform in TF and transform the frame."""
        planning_scene = self.get_input("planning_scene")
        robot_description = self.get_input("robot_description")
        try:
            planning_scene.world.collision_objects = self.parse_xml(robot_description)

        except Exception as e:
            self.node.get_logger().error(f"Parsing robot description has failed with : {e}")
            return Status.FAILURE

        self._set_output("modified_planning_scene", planning_scene)
        return Status.SUCCESS

    def box_to_solid_primitive(self, size: list[float]) -> SolidPrimitive:
        solid_primitive = SolidPrimitive()
        solid_primitive.type = SolidPrimitive.BOX
        solid_primitive.dimensions = list(size)

        return solid_primitive

    def cylinder_to_solid_primitive(self, cylinder_radius: float, cylinder_length: float) -> SolidPrimitive:
        solid_primitive = SolidPrimitive()
        solid_primitive.type = SolidPrimitive.CYLINDER
        solid_primitive.dimensions = list([cylinder_radius, cylinder_length])

        return solid_primitive

    def sphere_to_solid_primitive(self, sphere_radius: float) -> SolidPrimitive:
        solid_primitive = SolidPrimitive()
        solid_primitive.type = SolidPrimitive.SPHERE
        solid_primitive.dimensions = [sphere_radius]

        return solid_primitive

    def load_mesh_to_message(self, file_name: str) -> Mesh:
        import trimesh
        import re
        import os
        from ament_index_python.packages import get_package_share_directory

        pattern = r"package://(.*?)?/"
        match = re.search(pattern, file_name)

        if not match:
            return

        package_name = match.group(1).strip()
        package_dir = get_package_share_directory(package_name)

        file_path = os.path.join(package_dir, "/".join(file_name.split("/")[3:]))

        # Load the .stl or .obj file using trimesh
        scene_or_mesh = trimesh.load(file_path)
        # Attempt to guess the units and convey that to the operator.
        units = trimesh.units.units_from_metadata(scene_or_mesh, True)
        self.node.get_logger().warning(f"Loaded mesh is assumed to be in [{units}].")
        # Handle scene objects (if obj contains multiple meshes)
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh

        mesh_msg = Mesh()

        # Populate vertices
        for v in mesh.vertices:
            pt = Point()
            pt.x = float(v[0])
            pt.y = float(v[1])
            pt.z = float(v[2])
            mesh_msg.vertices.append(pt)

        # Populate triangle faces
        for f in mesh.faces:
            tri = MeshTriangle()
            tri.vertex_indices = [int(f[0]), int(f[1]), int(f[2])]
            mesh_msg.triangles.append(tri)

        return mesh_msg

    def collision_primitve_to_collision_object(
        self, header: Header, pose: Pose, solid_primitive: SolidPrimitive, xyz: list[float], rpy: list[float]
    ) -> CollisionObject:
        collision_object = CollisionObject()
        collision_object.header = header
        collision_object.pose = pose

        collision_object.primitives.append(solid_primitive)

        primitive_pose = Pose()
        primitive_pose.position.x = xyz[0]
        primitive_pose.position.y = xyz[1]
        primitive_pose.position.z = xyz[2]

        rotation = R.from_euler("xyz", rpy)
        quat = rotation.as_quat()
        primitive_pose.orientation.x = quat[0]
        primitive_pose.orientation.y = quat[1]
        primitive_pose.orientation.z = quat[2]
        primitive_pose.orientation.w = quat[3]

        collision_object.primitive_poses.append(primitive_pose)

        return collision_object

    def mesh_to_collision_object(
        self, header: Header, pose: Pose, mesh: Mesh, xyz: list[float], rpy: list[float]
    ) -> CollisionObject:
        collision_object = CollisionObject()
        collision_object.header = header
        collision_object.pose = pose

        collision_object.meshes.append(mesh)

        mesh_pose = Pose()
        mesh_pose.position.x = xyz[0]
        mesh_pose.position.y = xyz[1]
        mesh_pose.position.z = xyz[2]

        rotation = R.from_euler("xyz", rpy)
        quat = rotation.as_quat()
        mesh_pose.orientation.x = quat[0]
        mesh_pose.orientation.y = quat[1]
        mesh_pose.orientation.z = quat[2]
        mesh_pose.orientation.w = quat[3]

        collision_object.mesh_poses.append(mesh_pose)

        return collision_object

    def get_local_pose(self, description_root, child_name) -> tuple[Header, Pose]:
        parent_name = None
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]

        for joint in description_root.findall("joint"):
            joint_child_name = joint.find("child").get("link")
            if joint_child_name != child_name:
                continue
            else:
                parent_name = joint.find("parent").get("link")
                origin = joint.find("origin")
                if origin is not None:
                    if origin.get("xyz"):
                        xyz = [float(x) for x in origin.get("xyz").split()]
                    if origin.get("rpy"):
                        rpy = [float(x) for x in origin.get("rpy").split()]
                break

        if parent_name is None:
            raise KeyError(f"Did not find parent frame name for link [ {child_name} ]")

        pose = Pose()

        pose.position.x = xyz[0]
        pose.position.y = xyz[1]
        pose.position.z = xyz[2]

        rotation = R.from_euler("xyz", rpy)
        quat = rotation.as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        header = Header()
        header.frame_id = parent_name

        return header, pose

    def parse_xml(self, robot_description) -> list[CollisionObject]:

        collision_objects = []

        root = ET.fromstring(robot_description)

        for link in root.findall("link"):
            link_name = link.get("name")
            collision = link.find("collision")

            if collision is not None:

                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]

                origin = collision.find("origin")
                if origin is not None:
                    if origin.get("xyz"):
                        xyz = [float(x) for x in origin.get("xyz").split()]
                    if origin.get("rpy"):
                        rpy = [float(x) for x in origin.get("rpy").split()]

                geometry = collision.find("geometry")
                if geometry is not None:
                    header, pose = self.get_local_pose(root, link_name)
                    for child in geometry:
                        collision_object = None
                        match child.tag:
                            case "box":
                                box_size = child.get("size")
                                box_size = [float(x) for x in box_size.split()]
                                solid_primitive = self.box_to_solid_primitive(box_size)
                                collision_object = self.collision_primitve_to_collision_object(
                                    header, pose, solid_primitive, xyz, rpy
                                )

                            case "cylinder":
                                cylinder_radius = float(child.get("radius"))
                                cylinder_length = float(child.get("length"))
                                solid_primitive = self.cylinder_to_solid_primitive(cylinder_radius, cylinder_length)
                                collision_object = self.collision_primitve_to_collision_object(
                                    header, pose, solid_primitive, xyz, rpy
                                )

                            case "sphere":
                                sphere_radius = float(child.get("radius"))
                                solid_primitive = self.sphere_to_solid_primitive(sphere_radius)
                                collision_object = self.collision_primitve_to_collision_object(
                                    header, pose, solid_primitive, xyz, rpy
                                )

                            case "mesh":
                                self.node.get_logger().warning(
                                    "Parsing meshes from URDF into shape_msgs/msg/Mesh is experimental!"
                                )

                                filename = child.get("filename")
                                mesh = self.load_mesh_to_message(filename)
                                collision_object = self.mesh_to_collision_object(header, pose, mesh, xyz, rpy)

                            case _:
                                self.node.get_logger().warning(
                                    f"Collision object tag {child.tag} is not recognized. Skipping the object."
                                )

                        if collision_object is not None:
                            collision_object.id = link_name
                            collision_objects.append(collision_object)

        return collision_objects
