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
import time
from collections.abc import Iterator

from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.decorators import Decorator
from py_trees.ports import PortInformation, PortsMixin


class SuccessIfVariableIsTrue(PortsMixin, Decorator):
    """
    Immediately returns success if a specific blackboard variable is true,
    without ticking the decorated child node.
    Otherwise, ticks the decorated child node and returns its underlying status.
    """

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {"variable": PortInformation(data_type=bool, required=True)}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {}

    def tick(self) -> Iterator[Behaviour]:
        """Either return success without ticking the child node, or tick it.

        The base decorator tick always ticks the child, so the skip must happen here
        rather than in update(). Stopping also invalidates a previously running child.
        """
        if self.get_input("variable"):
            self.stop(Status.SUCCESS)
            yield self
        else:
            yield from super().tick()

    def update(self) -> Status:
        """Reflect the child's status, since it is only ticked when the variable is false."""
        return self.decorated.status


class SuccessIfVariableIsFalse(PortsMixin, Decorator):
    """
    Immediately returns success if a specific blackboard variable is false,
    without ticking the decorated child node.
    Otherwise, ticks the decorated child node and returns its underlying status.
    """

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {"variable": PortInformation(data_type=bool, required=True)}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {}

    def tick(self) -> Iterator[Behaviour]:
        """Either return success without ticking the child node, or tick it.

        The base decorator tick always ticks the child, so the skip must happen here
        rather than in update(). Stopping also invalidates a previously running child.
        """
        if not self.get_input("variable"):
            self.stop(Status.SUCCESS)
            yield self
        else:
            yield from super().tick()

    def update(self) -> Status:
        """Reflect the child's status, since it is only ticked when the variable is true."""
        return self.decorated.status


class TimeoutPort(PortsMixin, Decorator):
    """
    Reimplementation of the Timeout behavior from the py_trees library
    [Link](https://py-trees.readthedocs.io/en/devel/modules.html#py_trees.decorators.Timeout) for reference.

    NOTE: This behavior is supposed to be temporary.

    Executes a child/subtree with a timeout.

    A decorator that applies a timeout pattern to an existing behaviour.
    If the timeout is reached, the encapsulated behaviour's
    :meth:`~py_trees.behaviour.Behaviour.stop` method is called with
    status :data:`~py_trees.common.Status.FAILURE` otherwise it will
    simply directly tick and return with the same status
    as that of its encapsulated behaviour.
    """

    finish_time: float = None

    @classmethod
    def input_ports(cls) -> dict:
        """Return the input port declarations."""
        return {"duration": PortInformation(data_type=float, required=True)}

    @classmethod
    def output_ports(cls) -> dict:
        """Return the output port declarations."""
        return {}

    def initialise(self) -> None:
        """Reset the feedback message and finish time on behaviour entry."""
        self.finish_time = time.monotonic() + self.get_input("duration")
        self.feedback_message = ""

    def update(self) -> Status:
        """
        Fail on timeout, or block / reflect the child's result accordingly.

        Terminate the child and return
        :data:`~py_trees.common.Status.FAILURE`
        if the timeout is exceeded.

        Returns:
            the behaviour's new status :class:`~py_trees.common.Status`
        """

        current_time = time.monotonic()

        if self.decorated.status == Status.RUNNING and current_time > self.finish_time:
            self.feedback_message = "timed out"
            self.logger.debug(f"{self.__class__.__name__}.update() {self.feedback_message}")
            # invalidate the decorated (i.e. cancel it), could also put this logic in a terminate() method
            self.decorated.stop(Status.INVALID)
            return Status.FAILURE
        if self.decorated.status == Status.RUNNING:
            self.feedback_message = f"time still ticking ... [remaining: {self.finish_time - current_time}s]"
        else:
            self.feedback_message = "child finished before timeout triggered"
        return self.decorated.status
