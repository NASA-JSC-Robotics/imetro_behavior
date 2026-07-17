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

from py_trees.common import Status
from py_trees.decorators import Decorator
from py_trees.ports import PortInformation, PortsMixin


class SuccessIfVariableIsTrue(PortsMixin, Decorator):
    """
    Immediately returns success if a specific blackboard variable is true.
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

    def update(self) -> Status:
        """Either return success or tick the child node."""
        if self.get_input("variable"):
            return Status.SUCCESS
        else:
            self.decorated.tick_once()
            return self.decorated.status


class SuccessIfVariableIsFalse(PortsMixin, Decorator):
    """
    Immediately returns success if a specific blackboard variable is false.
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

    def update(self) -> Status:
        """Either return success or tick the child node."""
        if not self.get_input("variable"):
            return Status.SUCCESS
        else:
            return self.decorated.status
