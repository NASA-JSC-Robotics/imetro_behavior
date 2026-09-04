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

from py_trees.behaviour import Behaviour
from rclpy.node import Node


def set_ros_node(behavior: Behaviour, **kwargs) -> None:
    """
    Helper function to extract a ROS node from keyword arguments and set it for this behavior tree node.

    This mutates the behavior by setting its 'node' and 'logger' attributes.
    """
    node = kwargs.get("node")
    if not isinstance(node, Node):
        raise KeyError(f"A valid ROS node is required to setup the '{behavior.qualified_name}' node.")

    behavior.node = node  # ty: ignore[unresolved-attribute]
    behavior.logger = node.get_logger()
