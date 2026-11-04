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

from collections.abc import Iterator
from typing import Any

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

    INPUT_PORTS = {"variable": PortInformation(data_type=bool, required=True)}
    OUTPUT_PORTS = {}

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

    INPUT_PORTS = {"variable": PortInformation(data_type=bool, required=True)}
    OUTPUT_PORTS = {}

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


class ForEachItem(PortsMixin, Decorator):
    """
    Runs the decorated child once per item of a list, writing the current item to an output port.

    This is a ports-based version of `py_trees.decorators.ForEach`.
    It moves to the next item each time the child succeeds, and succeeds after the last one.
    A failing child ends the loop with failure, so wrap the child in `FailureIsSuccess` to keep going.
    An empty list succeeds without ticking the child.
    """

    INPUT_PORTS = {"items": PortInformation(data_type=list, required=True)}
    OUTPUT_PORTS = {"item": PortInformation(data_type=Any, required=True)}

    def initialise(self) -> None:
        """Start again from the first item."""
        self._items = list(self.get_input("items"))
        self._index = 0
        if self._items:
            self._set_output("item", self._items[0])

    def tick(self) -> Iterator[Behaviour]:
        """Tick the child, unless there is nothing to iterate over."""
        if self.status != Status.RUNNING and not self.get_input("items"):
            self.stop(Status.SUCCESS)
            yield self
        else:
            yield from super().tick()

    def update(self) -> Status:
        """Reflect the child's status, moving to the next item when it succeeds."""
        if self.decorated.status != Status.SUCCESS:
            return self.decorated.status
        self._index += 1
        if self._index >= len(self._items):
            return Status.SUCCESS
        self._set_output("item", self._items[self._index])
        return Status.RUNNING


class OutputChildSuccess(PortsMixin, Decorator):
    """
    Outputs whether the decorated child succeeded, and succeeds regardless once the child is done.

    This allows behaviors that come after the child to act on its outcome, even if the child fails.
    """

    INPUT_PORTS = {}
    OUTPUT_PORTS = {"succeeded": PortInformation(data_type=bool, required=True)}

    def update(self) -> Status:
        """Stay running until the child is done, and then output its outcome."""
        if self.decorated.status == Status.RUNNING:
            return Status.RUNNING
        self._set_output("succeeded", self.decorated.status == Status.SUCCESS)
        return Status.SUCCESS
