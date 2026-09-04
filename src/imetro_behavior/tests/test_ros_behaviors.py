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
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from imetro_behavior.ros_behaviors.action_client import RosActionClientBase
from imetro_behavior.ros_behaviors.perception import GetSyncedImagePointCloudDepth
from imetro_behavior.ros_behaviors.service_client import CallTriggerService
from imetro_behavior.ros_behaviors.subscriber_base import GetStringTopic
from py_trees.common import Status
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger


class TrackingActionBehavior(RosActionClientBase):
    """Minimal concrete action behavior that records what process_result() receives."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, action_type=GripperCommand, **kwargs)
        self.received_result = None

    INPUT_PORTS = {}
    OUTPUT_PORTS = {}

    def create_goal(self) -> GripperCommand.Goal:
        return GripperCommand.Goal()

    def process_result(self, result: GripperCommand.Result) -> Status:
        self.received_result = result
        return Status.SUCCESS


@pytest.fixture()
def completed_action_behavior(ros_node: Node) -> TrackingActionBehavior:
    """Return a TrackingActionBehavior with its internal state staged as if a goal is in flight."""
    behavior = TrackingActionBehavior(name="tracking_action", action_name="/foo")
    behavior.setup(node=ros_node)
    behavior.setup_ports()
    behavior.initialise()
    behavior.client_ready = True
    behavior.send_goal_future = Future()
    return behavior


def set_action_result(behavior: RosActionClientBase, status: int, result) -> None:
    """Complete the staged goal with the given goal status and action result."""
    future = Future()
    future.set_result(GripperCommand.Impl.GetResultService.Response(status=status, result=result))
    behavior.get_result_future = future


def test_action_client_unwraps_successful_result(
    completed_action_behavior: TrackingActionBehavior,
) -> None:
    result = GripperCommand.Result(position=0.5)
    set_action_result(completed_action_behavior, GoalStatus.STATUS_SUCCEEDED, result)

    assert completed_action_behavior.update() == Status.SUCCESS
    assert completed_action_behavior.received_result == result


def test_action_client_fails_without_goal_success(
    completed_action_behavior: TrackingActionBehavior,
) -> None:
    set_action_result(completed_action_behavior, GoalStatus.STATUS_ABORTED, GripperCommand.Result())

    assert completed_action_behavior.update() == Status.FAILURE
    # process_result() should never be called when the goal did not succeed.
    assert completed_action_behavior.received_result is None


def test_call_trigger_service(ros_node: Node) -> None:
    behavior = CallTriggerService(name="trigger", service_name="/trigger")
    behavior.setup(node=ros_node)
    behavior.setup_ports()

    assert isinstance(behavior.create_request(), Trigger.Request)

    assert behavior.process_response(Trigger.Response(success=True)) == Status.SUCCESS
    assert behavior.process_response(Trigger.Response(success=False, message="oops")) == Status.FAILURE


@pytest.fixture()
def sync_behavior(ros_node: Node) -> GetSyncedImagePointCloudDepth:
    behavior = GetSyncedImagePointCloudDepth(
        name="get_synced_data",
        camera_info_topic="/camera/camera_info",
        rgb_image_topic="/camera/color",
        depth_image_topic="/camera/depth",
        point_cloud_topic="/camera/points",
        sync_timeout=1.0,
    )
    behavior.setup(node=ros_node)
    behavior.setup_ports()
    return behavior


def test_get_synced_data_success(sync_behavior: GetSyncedImagePointCloudDepth) -> None:
    # Inject a synchronized frame directly instead of calling initialise(),
    # which would create real topic subscriptions.
    sync_behavior.latest_data = (CameraInfo(), Image(), Image(), PointCloud2())

    assert sync_behavior.update() == Status.SUCCESS
    assert isinstance(sync_behavior.get_last_output("camera_info"), CameraInfo)
    assert isinstance(sync_behavior.get_last_output("rgb_image"), Image)
    assert isinstance(sync_behavior.get_last_output("depth_image"), Image)
    assert isinstance(sync_behavior.get_last_output("point_cloud"), PointCloud2)
    # The cached frame should be cleared so the same data is not processed twice.
    assert sync_behavior.latest_data is None


def test_get_synced_data_waiting_and_timeout(
    sync_behavior: GetSyncedImagePointCloudDepth,
) -> None:
    node = sync_behavior.node
    sync_behavior.latest_data = None

    sync_behavior.start_time = node.get_clock().now()
    assert sync_behavior.update() == Status.RUNNING

    sync_behavior.start_time = node.get_clock().now() - Duration(seconds=2.0)  # ty: ignore [invalid-assignment]
    assert sync_behavior.update() == Status.FAILURE


@pytest.fixture()
def get_string_topic_behavior(ros_node: Node) -> GetStringTopic:
    behavior = GetStringTopic(
        name="get_string_topic",
        topic_name="/string_topic",
        subscriber_timeout=1.0,
        qos_profile="default",
    )
    behavior.setup(node=ros_node)
    behavior.setup_ports()
    return behavior


def test_get_string_topic_behavior(get_string_topic_behavior: GetStringTopic) -> None:
    data = "string_payload"

    get_string_topic_behavior.latest_msg = String(data=data)

    assert get_string_topic_behavior.update() == Status.SUCCESS

    result = get_string_topic_behavior.get_last_output("message")
    assert isinstance(result, str)
    assert result == data
    # Cached message should be cleared
    assert get_string_topic_behavior.latest_msg is None
