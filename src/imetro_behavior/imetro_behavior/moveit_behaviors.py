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

from typing import Any

from py_trees.common import Status
from py_trees.ports import PortInformation

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PositionConstraint,
    OrientationConstraint,
    RobotTrajectory,
)
from moveit_msgs.srv import GetMotionPlan
from shape_msgs.msg import SolidPrimitive

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


class PreviewTrajectoryBehavior(RosActionClientBase):
    """Sends a planned trajectory to the behavior GUI for preview."""

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

    def create_goal(self) -> ExecuteTrajectory.Goal:
        """Create a trajectory preview goal."""
        return PreviewTrajectory.Goal(robot_id="clr", trajectory=self.get_input("trajectory"))

    def process_result(self, result: ExecuteTrajectory.Result) -> Status:
        """Process the trajectory preview action result."""
        approved = result.result.approved
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
        error_code = result.result.error_code
        if error_code.val == MoveItErrorCodes.SUCCESS:
            self.node.get_logger().info("Trajectory execution succeeded!")
            return Status.SUCCESS
        else:
            error_code_str = MOVEIT_ERROR_CODE_DICT.get(error_code.val, "UNKNOWN")
            self.node.get_logger().error(f"Trajectory execution failed with error code: {error_code_str}")
            self.node.get_logger().error(f"Message: {error_code.message}")
            self.node.get_logger().error(f"Source: {error_code.source}")
            return Status.FAILURE
