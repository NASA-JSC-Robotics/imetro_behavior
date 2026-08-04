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

import rclpy
from rclpy.node import Node
from py_trees.blackboard import Blackboard


@pytest.fixture(autouse=True)
def clear_blackboard():
    """Clear the blackboard before each test, since it is shared process-wide."""
    Blackboard.clear()
    yield


@pytest.fixture()
def ros_node():
    """
    Common fixture that creates a ROS node, usable by behavior unit tests.

    Note this fixture does not start any executor or do any spinning.
    Ideally, tests don't require there being a spinning node in the background and
    instead are testing atomic functionality.
    """
    # Setup
    rclpy.init()
    node = Node("imetro_behavior_test_node")

    yield node

    # Teardown
    node.destroy_node()
    rclpy.try_shutdown()
