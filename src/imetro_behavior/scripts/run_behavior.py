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

import signal
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from imetro_behavior.executor import BehaviorTreeServer

if __name__ == "__main__":
    # By default, rclpy shuts down as soon as it gets a signal, after which we can no longer cancel
    # goals that are moving a robot. Instead, handle signals here and shut down ROS last.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    node = Node("run_behavior")
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    bt_server = BehaviorTreeServer(node)

    # The executor needs to keep spinning while the behavior stops, so it runs on its own thread.
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        bt_server.stop()
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.try_shutdown()
