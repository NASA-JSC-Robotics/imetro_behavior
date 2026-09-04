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

from geometry_msgs.msg import PoseStamped
from imetro_behavior.nav_behaviors import NavigateToPoseBehavior
from nav2_msgs.action import NavigateToPose
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rclpy.node import Node


def test_navigate_to_pose_create_goal(ros_node: Node) -> None:
    behavior = NavigateToPoseBehavior(name="navigate", action_name="/navigate_to_pose")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    goal_pose = PoseStamped()
    goal_pose.header.frame_id = "map"
    goal_pose.pose.position.x = 1.5
    goal_pose.pose.orientation.w = 1.0
    Blackboard.set(behavior._get_blackboard_key("goal_pose"), goal_pose)

    goal = behavior.create_goal()
    assert goal.pose.header.frame_id == "map"
    assert goal.pose.pose.position.x == 1.5
    # The goal should be stamped with the current time.
    assert goal.pose.header.stamp.sec > 0


def test_navigate_to_pose_process_result(ros_node: Node) -> None:
    behavior = NavigateToPoseBehavior(name="navigate", action_name="/navigate_to_pose")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    result = NavigateToPose.Result()
    assert result.error_code == NavigateToPose.Result.NONE
    assert behavior.process_result(result) == Status.SUCCESS

    result.error_code = 42
    result.error_msg = "no valid path found"
    assert behavior.process_result(result) == Status.FAILURE
