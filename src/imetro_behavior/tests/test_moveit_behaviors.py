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

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import AllowedCollisionEntry, MoveItErrorCodes, PlanningScene, RobotTrajectory
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetMotionPlan, GetPlanningScene
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer

from imetro_behavior_msgs.action import PreviewTrajectory
from imetro_behavior.moveit_behaviors import (
    ExecuteTrajectoryBehavior,
    ModifyCollisions,
    PlanArcPath,
    PlanCartesian,
    PlanToJointState,
    PlanToPose,
    RequestPlanningScene,
    RequestTrajectoryApproval,
    PlanningSceneFromRobotDescription,
)


def set_input(behavior, port_name, value):
    Blackboard.set(behavior._get_blackboard_key(port_name), value)


def make_motion_plan_response(error_code: int) -> GetMotionPlan.Response:
    response = GetMotionPlan.Response()
    response.motion_plan_response.error_code.val = error_code
    return response


def test_plan_to_joint_state_create_request(ros_node: Node) -> None:
    behavior = PlanToJointState(name="plan_to_joint_state", service_name="/plan")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    set_input(behavior, "group_name", "arm")
    set_input(behavior, "joint_names", ["joint1", "joint2"])
    set_input(behavior, "joint_positions", [0.5, -0.5])
    set_input(behavior, "tolerance", 0.01)

    request = behavior.create_request()
    assert request.motion_plan_request.group_name == "arm"
    assert request.motion_plan_request.pipeline_id == ""
    assert request.motion_plan_request.max_velocity_scaling_factor == 1.0
    assert request.motion_plan_request.max_acceleration_scaling_factor == 1.0

    joint_constraints = request.motion_plan_request.goal_constraints[0].joint_constraints
    assert len(joint_constraints) == 2
    assert joint_constraints[0].joint_name == "joint1"
    assert joint_constraints[0].position == 0.5
    assert joint_constraints[0].tolerance_above == 0.01
    assert joint_constraints[0].tolerance_below == 0.01
    assert joint_constraints[1].joint_name == "joint2"
    assert joint_constraints[1].position == -0.5


def test_plan_to_joint_state_mismatched_lengths(ros_node: Node) -> None:
    behavior = PlanToJointState(name="plan_to_joint_state", service_name="/plan")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    set_input(behavior, "group_name", "arm")
    set_input(behavior, "joint_names", ["joint1", "joint2"])
    set_input(behavior, "joint_positions", [0.5])
    set_input(behavior, "tolerance", 0.01)

    with pytest.raises(RuntimeError) as exc_info:
        behavior.create_request()
    assert "Joint names and joint positions must have the same length." in str(exc_info.value)


def test_plan_to_joint_state_process_response(ros_node: Node) -> None:
    behavior = PlanToJointState(name="plan_to_joint_state", service_name="/plan")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    response = make_motion_plan_response(MoveItErrorCodes.SUCCESS)
    assert behavior.process_response(response) == Status.SUCCESS
    assert behavior.get_last_output("trajectory") == response.motion_plan_response.trajectory

    assert behavior.process_response(make_motion_plan_response(MoveItErrorCodes.PLANNING_FAILED)) == Status.FAILURE


def test_plan_to_pose_create_request(ros_node: Node) -> None:
    behavior = PlanToPose(name="plan_to_pose", service_name="/plan")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    target_pose = PoseStamped()
    target_pose.header.frame_id = "world"
    target_pose.pose.position.x = 1.0
    target_pose.pose.orientation.w = 1.0

    set_input(behavior, "group_name", "arm")
    set_input(behavior, "target_frame", "tool0")
    set_input(behavior, "target_pose", target_pose)
    set_input(behavior, "position_tolerance", 0.01)
    set_input(behavior, "orientation_tolerance", [0.1, 0.2, 0.3])

    request = behavior.create_request()
    assert request.motion_plan_request.group_name == "arm"

    position_constraint = request.motion_plan_request.goal_constraints[0].position_constraints[0]
    assert position_constraint.link_name == "tool0"
    assert position_constraint.header.frame_id == "world"
    assert position_constraint.constraint_region.primitives[0].type == SolidPrimitive.SPHERE
    assert list(position_constraint.constraint_region.primitives[0].dimensions) == [0.01]
    assert position_constraint.constraint_region.primitive_poses[0].position.x == 1.0

    orientation_constraint = request.motion_plan_request.goal_constraints[0].orientation_constraints[0]
    assert orientation_constraint.link_name == "tool0"
    assert orientation_constraint.header.frame_id == "world"
    assert orientation_constraint.orientation == target_pose.pose.orientation
    assert orientation_constraint.absolute_x_axis_tolerance == 0.1
    assert orientation_constraint.absolute_y_axis_tolerance == 0.2
    assert orientation_constraint.absolute_z_axis_tolerance == 0.3


def test_plan_to_pose_process_response(ros_node: Node) -> None:
    behavior = PlanToPose(name="plan_to_pose", service_name="/plan")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    assert behavior.process_response(make_motion_plan_response(MoveItErrorCodes.SUCCESS)) == Status.SUCCESS
    assert behavior.process_response(make_motion_plan_response(MoveItErrorCodes.NO_IK_SOLUTION)) == Status.FAILURE


def test_request_planning_scene(ros_node: Node) -> None:
    behavior = RequestPlanningScene(name="request_planning_scene", service_name="/get_planning_scene")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    assert isinstance(behavior.create_request(), GetPlanningScene.Request)

    response = GetPlanningScene.Response()
    response.scene.name = "test_scene"
    assert behavior.process_response(response) == Status.SUCCESS
    assert behavior.get_last_output("planning_scene").name == "test_scene"


def make_planning_scene(entry_names: list[str], enabled: bool = False) -> PlanningScene:
    planning_scene = PlanningScene()
    acm = planning_scene.allowed_collision_matrix
    acm.entry_names = entry_names
    acm.entry_values = [AllowedCollisionEntry(enabled=[enabled] * len(entry_names)) for _ in entry_names]
    return planning_scene


def test_modify_collisions_allow_new_link(ros_node: Node) -> None:
    behavior = ModifyCollisions(name="modify_collisions", service_name="/apply_planning_scene")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    set_input(behavior, "planning_scene", make_planning_scene(["link_a", "link_b"]))
    set_input(behavior, "links_list_1", ["link_a"])
    set_input(behavior, "links_list_2", ["new_link"])
    set_input(behavior, "allow_collision", True)

    request = behavior.create_request()
    assert request.scene.is_diff

    acm = request.scene.allowed_collision_matrix
    assert acm.entry_names == ["link_a", "link_b", "new_link"]
    # The matrix should have been expanded to 3x3 for the new entry.
    assert all(len(entry.enabled) == 3 for entry in acm.entry_values)

    index_a = acm.entry_names.index("link_a")
    index_b = acm.entry_names.index("link_b")
    index_new = acm.entry_names.index("new_link")
    assert acm.entry_values[index_a].enabled[index_new]
    assert acm.entry_values[index_new].enabled[index_a]
    assert not acm.entry_values[index_a].enabled[index_b]
    assert not acm.entry_values[index_b].enabled[index_new]


def test_modify_collisions_multiple_new_links(ros_node: Node) -> None:
    behavior = ModifyCollisions(name="modify_collisions", service_name="/apply_planning_scene")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    set_input(behavior, "planning_scene", make_planning_scene(["link_a"]))
    set_input(behavior, "links_list_1", ["new_link_1"])
    set_input(behavior, "links_list_2", ["new_link_2"])
    set_input(behavior, "allow_collision", True)

    request = behavior.create_request()
    acm = request.scene.allowed_collision_matrix
    assert all(len(entry.enabled) == 3 for entry in acm.entry_values)

    index_a = acm.entry_names.index("link_a")
    index_1 = acm.entry_names.index("new_link_1")
    index_2 = acm.entry_names.index("new_link_2")
    assert acm.entry_values[index_1].enabled[index_2]
    assert acm.entry_values[index_2].enabled[index_1]
    # The new entries must be independent rows: diagonals and the existing link stay untouched.
    assert not acm.entry_values[index_1].enabled[index_1]
    assert not acm.entry_values[index_2].enabled[index_2]
    assert not acm.entry_values[index_1].enabled[index_a]
    assert not acm.entry_values[index_a].enabled[index_1]
    assert not acm.entry_values[index_a].enabled[index_2]


def test_modify_collisions_disallow(ros_node: Node) -> None:
    behavior = ModifyCollisions(name="modify_collisions", service_name="/apply_planning_scene")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    set_input(behavior, "planning_scene", make_planning_scene(["link_a", "link_b"], enabled=True))
    set_input(behavior, "links_list_1", ["link_a"])
    set_input(behavior, "links_list_2", ["link_b"])
    set_input(behavior, "allow_collision", False)

    request = behavior.create_request()
    acm = request.scene.allowed_collision_matrix
    assert not acm.entry_values[0].enabled[1]
    assert not acm.entry_values[1].enabled[0]
    # Diagonal entries are untouched.
    assert acm.entry_values[0].enabled[0]
    assert acm.entry_values[1].enabled[1]


def test_modify_collisions_process_response(ros_node: Node) -> None:
    behavior = ModifyCollisions(name="modify_collisions", service_name="/apply_planning_scene")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    assert behavior.process_response(ApplyPlanningScene.Response(success=True)) == Status.SUCCESS
    assert behavior.process_response(ApplyPlanningScene.Response(success=False)) == Status.FAILURE


def test_plan_cartesian_create_request(ros_node: Node) -> None:
    behavior = PlanCartesian(name="plan_cartesian", service_name="/plan_cartesian")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    waypoint = PoseStamped()
    waypoint.pose.position.x = 1.0
    waypoint.pose.orientation.w = 1.0

    set_input(behavior, "group_name", "arm")
    set_input(behavior, "waypoints", waypoint)

    request = behavior.create_request()
    assert request.group_name == "arm"
    # A single PoseStamped waypoint is packaged behind an empty starting pose.
    assert request.waypoints == [waypoint.pose]
    assert request.max_step == 0.01
    assert request.jump_threshold == 1.25
    assert request.avoid_collisions


def test_plan_cartesian_process_response(ros_node: Node) -> None:
    behavior = PlanCartesian(name="plan_cartesian", service_name="/plan_cartesian")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    response = GetCartesianPath.Response()
    response.error_code.val = MoveItErrorCodes.SUCCESS
    assert behavior.process_response(response) == Status.SUCCESS
    assert behavior.get_last_output("trajectory") == response.solution

    response.error_code.val = MoveItErrorCodes.PLANNING_FAILED
    assert behavior.process_response(response) == Status.FAILURE


@pytest.fixture()
def arc_behavior(ros_node: Node) -> PlanArcPath:
    """A PlanArcPath behavior with a 'tool' frame at x=1.0 in the 'world' frame, rotating 90 degrees about +Z."""
    buffer = Buffer()
    tform = TransformStamped()
    tform.header.frame_id = "world"
    tform.child_frame_id = "tool"
    tform.transform.translation.x = 1.0
    tform.transform.rotation.w = 1.0
    buffer.set_transform_static(tform, "test_authority")
    Blackboard.set("/ros/tf_buffer", buffer)

    behavior = PlanArcPath(name="plan_arc", service_name="/plan")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    rotation_pose = PoseStamped()
    rotation_pose.header.frame_id = "world"
    rotation_pose.pose.orientation.w = 1.0

    set_input(behavior, "fixed_frame", "world")
    set_input(behavior, "group_name", "arm")
    set_input(behavior, "target_frame", "tool")
    set_input(behavior, "rotation_pose", rotation_pose)
    set_input(behavior, "rotation_axis_xyz", [0.0, 0.0, 1.0])
    set_input(behavior, "rotation_amount_rad", np.pi / 2.0)
    set_input(behavior, "keep_start_orientation", True)
    set_input(behavior, "position_tolerance", 0.01)
    set_input(behavior, "orientation_tolerance_xyz", [0.1, 0.1, 0.1])
    set_input(behavior, "max_velocity_scaling", 0.5)
    set_input(behavior, "max_acceleration_scaling", 0.1)
    return behavior


def test_plan_arc_path_rotate_about_frame(arc_behavior: PlanArcPath) -> None:
    final_pose, center_point = arc_behavior.rotate_about_frame()

    # Rotating the tool at (1, 0, 0) by 90 degrees about +Z through the origin lands at (0, 1, 0).
    assert final_pose.position.x == pytest.approx(0.0, abs=1e-9)
    assert final_pose.position.y == pytest.approx(1.0)
    assert final_pose.position.z == pytest.approx(0.0, abs=1e-9)
    # keep_start_orientation retains the identity orientation.
    assert final_pose.orientation.w == pytest.approx(1.0)

    assert center_point.x == pytest.approx(0.0, abs=1e-9)
    assert center_point.y == pytest.approx(0.0, abs=1e-9)
    assert center_point.z == pytest.approx(0.0, abs=1e-9)


def test_plan_arc_path_rotated_orientation(arc_behavior: PlanArcPath) -> None:
    set_input(arc_behavior, "keep_start_orientation", False)

    final_pose, _ = arc_behavior.rotate_about_frame()
    # The final orientation is now also rotated by 90 degrees about +Z.
    assert final_pose.orientation.z == pytest.approx(np.sin(np.pi / 4.0))
    assert final_pose.orientation.w == pytest.approx(np.cos(np.pi / 4.0))


def test_plan_arc_path_create_request(arc_behavior: PlanArcPath) -> None:
    request = arc_behavior.create_request()
    assert request.motion_plan_request.pipeline_id == "pilz_industrial_motion_planner"
    assert request.motion_plan_request.planner_id == "CIRC"
    assert request.motion_plan_request.group_name == "arm"
    assert request.motion_plan_request.max_velocity_scaling_factor == 0.5

    goal_constraints = request.motion_plan_request.goal_constraints[0]
    goal_position = goal_constraints.position_constraints[0].constraint_region.primitive_poses[0].position
    assert goal_position.x == pytest.approx(0.0, abs=1e-9)
    assert goal_position.y == pytest.approx(1.0)

    path_constraints = request.motion_plan_request.path_constraints
    assert path_constraints.name == "center"
    center_position = path_constraints.position_constraints[0].constraint_region.primitive_poses[0].position
    assert center_position.x == pytest.approx(0.0, abs=1e-9)
    assert center_position.y == pytest.approx(0.0, abs=1e-9)


def test_request_trajectory_approval(ros_node: Node) -> None:
    behavior = RequestTrajectoryApproval(name="request_approval", action_name="/preview_trajectory")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    trajectory = RobotTrajectory()
    set_input(behavior, "trajectory", trajectory)

    goal = behavior.create_goal()
    assert goal.trajectory == trajectory

    result = PreviewTrajectory.Result(approved=True)
    assert behavior.process_result(result) == Status.SUCCESS
    assert behavior.get_last_output("approved")

    result.approved = False
    assert behavior.process_result(result) == Status.FAILURE
    assert not behavior.get_last_output("approved")


def test_execute_trajectory(ros_node: Node) -> None:
    behavior = ExecuteTrajectoryBehavior(name="execute_trajectory", action_name="/execute_trajectory")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    trajectory = RobotTrajectory()
    set_input(behavior, "trajectory", trajectory)
    assert behavior.create_goal().trajectory == trajectory

    result = ExecuteTrajectory.Result()
    result.error_code.val = MoveItErrorCodes.SUCCESS
    assert behavior.process_result(result) == Status.SUCCESS

    result.error_code.val = MoveItErrorCodes.CONTROL_FAILED
    assert behavior.process_result(result) == Status.FAILURE


from std_msgs.msg import String

@pytest.fixture()
def planning_scene_from_robot_description_behavior(ros_node: Node) -> PlanningSceneFromRobotDescription:
    # buffer = Buffer()
    # tform = TransformStamped()
    # tform.header.frame_id = "world"
    # tform.child_frame_id = "tool"
    # tform.transform.translation.x = 1.0
    # tform.transform.rotation.w = 1.0
    # buffer.set_transform_static(tform, "test_authority")
    # Blackboard.set("/ros/tf_buffer", buffer)

    test_urdf = r"""
    <?xml version="1.0"?>
      <robot name="test_robot">
        <link name="box_link">
          <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
              <box size="0.1 0.2 0.3"/>
            </geometry>
          </visual>
          <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
              <box size="0.1 0.2 0.3"/>
            </geometry>
          </collision>
        </link>

        <link name="cylinder_link">
          <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
              <cylinder raidus="0.1" length="0.2"/>
            </geometry>
          </visual>
          <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
              <cylinder raidus="0.1" length="0.2"/>
            </geometry>
          </collision>
        </link>


        <link name="sphere_link">
          <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
              <sphere raidus="0.1"/>
            </geometry>
          </visual>
          <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
              <sphere raidus="0.1"/>
            </geometry>
          </collision>
        </link>

        <joint name="box_to_cylinder_joint" type="revolute">
          <parent link="box_link"/>
          <child link="cylinder_link"/>
          <origin xyz="0.1 0.2 0.3" rpy="0 0 0"/>
          <axis xyz="1 0 0"/>
        </joint>

        <joint name="cylinder_to_sphere_joint" type="revolute">
          <parent link="cylinder_link"/>
          <child link="sphere_link"/>
          <origin xyz="0.1 0.2 0.3" rpy="0 0 0"/>
          <axis xyz="1 0 0"/>
        </joint>
      </robot>
    """

    behavior = PlanningSceneFromRobotDescription(name="parsing_planning_scene_test")
    behavior.setup_ports()
    behavior.setup(node=ros_node)

    planning_scene = PlanningScene()
    robot_description_string = String(data=test_urdf)

    set_input(behavior, "planning_scene", planning_scene)
    set_input(behavior, "robot_description", robot_description_string)
    return behavior



def test_planning_scene_from_robot_description_behavior(planning_scene_from_robot_description_behavior: PlanningSceneFromRobotDescription) -> None:

    assert planning_scene_from_robot_description_behavior.update() == Status.SUCCESS


    # final_pose, center_point = arc_behavior.rotate_about_frame()

    # # Rotating the tool at (1, 0, 0) by 90 degrees about +Z through the origin lands at (0, 1, 0).
    # assert final_pose.position.x == pytest.approx(0.0, abs=1e-9)
    # assert final_pose.position.y == pytest.approx(1.0)
    # assert final_pose.position.z == pytest.approx(0.0, abs=1e-9)
    # # keep_start_orientation retains the identity orientation.
    # assert final_pose.orientation.w == pytest.approx(1.0)

    # assert center_point.x == pytest.approx(0.0, abs=1e-9)
    # assert center_point.y == pytest.approx(0.0, abs=1e-9)
    # assert center_point.z == pytest.approx(0.0, abs=1e-9)