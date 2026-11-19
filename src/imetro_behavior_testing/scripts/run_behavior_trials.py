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

"""
Runs the trials of a behavior test file through a running behavior tree executor.
"""

import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from imetro_behavior_testing.test_runner import BehaviorTrialRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_file", help="Path of the test file.")
    parser.add_argument(
        "--output-dir",
        default="test_results",
        help="Folder in which to create the timestamped results folder (default: test_results).",
    )
    parser.add_argument(
        "--action-name",
        default="/execute_behavior",
        help="Name of the behavior tree executor's action (default: /execute_behavior).",
    )
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    # Keep the ROS context alive on Ctrl-C, so that the running trial can be canceled.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("behavior_trial_runner")
    try:
        runner = BehaviorTrialRunner(node, action_name=args.action_name)
        return 0 if runner.run(args.test_file, args.output_dir) else 1
    except (ValueError, RuntimeError) as e:
        node.get_logger().error(str(e))
        return 1
    except KeyboardInterrupt:
        node.get_logger().warning("Test run interrupted.")
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
