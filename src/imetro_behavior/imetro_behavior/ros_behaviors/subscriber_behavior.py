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

from typing import Any, Type

from rclpy.duration import Duration
from rclpy.node import Node

from std_srvs.srv import Trigger

from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts


class RosSubscriberClientBase(BehaviourWithPorts):
    """
    Base class for behaviors that rely on ROS service clients.

    Modified from https://github.com/splintered-reality/py_trees_ros/blob/devel/py_trees_ros/service_clients.py
    """

    def __init__(
        self,
        name: str,
        topic_type: Type,
        *,
        topic_name: str,
        subscriber_timeout: float = 3.0,
        **kwargs: Any,
    ):
        """
        Constructs a ROS Service Client base behavior.

        Args:
            name: The name of the behavior (required by PyTrees)
            service_type: The ROS interface type of the service.
            service_name: The name of the ROS service to send a request to.
            service_server_timeout: Timeout, in seconds, to wait for the service server to be available.
                If None, waits indefinitely.
            service_timeout: Timeout, in seconds, to wait for the service to complete.
                If None, waits indefinitely.
            kwargs: Additional keyword arguments to pass through to ports.
        """
        self.topic_type = topic_type
        self.topic_name = topic_name
        self.subscriber_timeout = Duration(seconds=subscriber_timeout) if subscriber_timeout else None
        super().__init__(name, **kwargs)

    def setup(self, **kwargs):
        """
        Sets up the service client.
        """
        self.node = kwargs.get("node")
        if not isinstance(self.node, Node):
            raise KeyError(f"A valid ROS node is required to setup the '{self.qualified_name}' node.")


    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


    def initialise(self):
        """
        Reset the internal variables.
        """
        self.subscription = self.create_subscription(
            self.topic_type,
            self.topic_name,
            self.listener_callback,
            10)
        self.service_future = None
        self.service_start_time = None
        self.client_start_time = self.node.get_clock().now()
        self.client_ready = self.service_client.service_is_ready()

    def update(self):
        """
        Kick off a new service request and then check whether the service has completed or timed out.
        """
        if not self.client_ready:
            # Wait for the service server to be available until there is a timeout.
            if (
                self.service_server_timeout is not None
                and self.node.get_clock().now() - self.client_start_time > self.service_server_timeout
            ):
                self.node.get_logger().error(f"Timed out waiting for service server {self.service_name}.")
                return Status.FAILURE
            else:
                self.client_ready = self.service_client.service_is_ready()
                return Status.RUNNING

        elif self.service_future is None:
            # Send a service request if one hasn't yet been sent.
            try:
                request = self.create_request()  # Must be implemented
            except Exception as e:
                self.node.get_logger().error(f"Failed to create service request: {e}")
                return Status.FAILURE

            self.node.get_logger().debug("Sending service request...")
            self.service_future = self.service_client.call_async(request)
            self.service_start_time = self.node.get_clock().now()
            return Status.RUNNING

        elif self.service_future.done():
            # If the service completed, process the response and return the corresponding status.
            try:
                # Must be implemented
                return self.process_response(self.service_future.result())
            except Exception as e:
                self.node.get_logger().error(f"Failed to process action result: {e}")
                return Status.FAILURE

        elif (
            self.service_timeout is not None
            and self.node.get_clock().now() - self.service_start_time > self.service_timeout
        ):
            # If we made it here, the service is in progress. Check for timeouts.
            self.node.get_logger().error("Service call timed out.")
            return Status.FAILURE

        else:
            # Service call is in progress but not yet timed out.
            return Status.RUNNING

    def terminate(self, new_status: Status) -> None:
        """
        If running and the current service call has not already completed, cancel it.
        """
        if self.status == Status.RUNNING and new_status == Status.INVALID:
            self.remove_pending_request()

    def shutdown(self):
        """
        Clean up the service client when shutting down.
        """
        self.remove_pending_request()  # For good measure
        self.service_client.destroy()

    def remove_pending_request(self):
        """
        Remove the pending service request.
        """
        if (self.service_future is not None) and (not self.service_future.done()):
            self.service_client.remove_pending_request(self.service_future)
