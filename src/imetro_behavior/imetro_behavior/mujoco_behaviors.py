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

from mujoco_ros2_control_msgs.srv import ResetWorld
from py_trees.common import Status
from py_trees.ports import PortInformation

from imetro_behavior.ros_behaviors.service_client import RosServiceClientBase


class ResetMujocoWorld(RosServiceClientBase):
    """Calls the ResetWorld service to reset the MuJoCo simulation.

    Optionally resets to a named keyframe defined in the MJCF model.
    An empty keyframe string resets to the initial simulation state.

    Ports
    -----
    Inputs:
        keyframe (str, optional) – Name of the MJCF keyframe to reset to.
                                   Defaults to "" (initial state).
    """

    def __init__(self, name: str, **kwargs: Any):
        super().__init__(name, service_type=ResetWorld, **kwargs)

    INPUT_PORTS = {"keyframe": PortInformation(data_type=str, required=False)}

    OUTPUT_PORTS = {}

    def create_request(self) -> ResetWorld.Request:
        return ResetWorld.Request(
            keyframe=self.get_input("keyframe", ""),
        )

    def process_response(self, response: ResetWorld.Response) -> Status:
        assert self.node is not None, "No ROS node available"
        if response.success:
            self.node.get_logger().info(f"Reset world succeeded: {response.message}")
            return Status.SUCCESS
        else:
            self.node.get_logger().error(f"Reset world failed: {response.message}")
            return Status.FAILURE
