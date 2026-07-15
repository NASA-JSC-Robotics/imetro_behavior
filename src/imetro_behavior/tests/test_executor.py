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

import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from imetro_behavior.executor import BehaviorTreeExecutor


def test_create_executor():
    rclpy.init()
    node = Node("run_behavior")
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    BehaviorTreeExecutor(
        node, dt=0.1, search_paths=[os.path.join(get_package_share_directory("imetro_behavior"), "trees")]
    )

    for _ in range(10):
        executor.spin_once(timeout_sec=0.1)

    executor.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()
