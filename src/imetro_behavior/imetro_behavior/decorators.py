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
