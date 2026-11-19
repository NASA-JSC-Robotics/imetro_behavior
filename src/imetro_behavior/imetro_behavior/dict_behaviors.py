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
Behaviors for working with dictionaries, and reading and writing them as YAML.

Keys can be nested, in which case they are addressed with dot-separated names such as `pose.position`.
"""

import copy
from pathlib import Path
from typing import Any

import yaml
from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts, NoDataAvailable, PortInformation


def _to_yaml_value(value: Any) -> Any:
    """Converts a value to plain Python types that YAML can write, such as NumPy scalars to floats."""
    if isinstance(value, dict):
        return {str(k): _to_yaml_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_yaml_value(v) for v in value]
    if hasattr(value, "tolist"):  # NumPy scalars and arrays
        return _to_yaml_value(value.tolist())
    return value


class ReadYamlFile(BehaviourWithPorts):
    """Reads a YAML file containing a mapping into a dictionary."""

    INPUT_PORTS = {
        "file_path": PortInformation(data_type=str, required=True, description="Path of the YAML file to read."),
    }
    OUTPUT_PORTS = {
        "dict": PortInformation(data_type=dict, required=True, description="The contents of the file."),
    }

    def update(self) -> Status:
        """Reads the file."""
        file_path = Path(self.get_input("file_path"))
        try:
            contents = yaml.safe_load(file_path.read_text())
        except (OSError, yaml.YAMLError) as e:
            self.logger.error(f"Could not read YAML file {file_path}: {e}")
            return Status.FAILURE

        # An empty file is an empty dictionary.
        contents = {} if contents is None else contents
        if not isinstance(contents, dict):
            self.logger.error(f"YAML file {file_path} does not contain a mapping.")
            return Status.FAILURE

        self._set_output("dict", contents)
        return Status.SUCCESS


class WriteYamlFile(BehaviourWithPorts):
    """Writes a dictionary to a YAML file, creating its folder if needed and replacing any existing file."""

    INPUT_PORTS = {
        "dict": PortInformation(data_type=dict, required=True, description="The dictionary to write."),
        "file_path": PortInformation(data_type=str, required=True, description="Path of the YAML file to write."),
    }
    OUTPUT_PORTS = {}

    def update(self) -> Status:
        """Writes the file."""
        file_path = Path(self.get_input("file_path"))
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(yaml.safe_dump(_to_yaml_value(self.get_input("dict")), sort_keys=False))
        except (OSError, yaml.YAMLError) as e:
            self.logger.error(f"Could not write YAML file {file_path}: {e}")
            return Status.FAILURE

        self.logger.info(f"Wrote {file_path}.")
        return Status.SUCCESS


class GetDictValue(BehaviourWithPorts):
    """
    Gets the value of a key in a dictionary.

    This fails if the key does not exist or its value is null, since null values cannot be passed between behaviors.
    """

    INPUT_PORTS = {
        "dict": PortInformation(data_type=dict, required=True, description="The dictionary to read from."),
        "key": PortInformation(data_type=str, required=True, description="The key, with dots separating nested keys."),
    }
    OUTPUT_PORTS = {
        "value": PortInformation(data_type=object, required=True, description="The value of the key."),
    }

    def update(self) -> Status:
        """Gets the value."""
        key = self.get_input("key")
        value = self.get_input("dict")
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                self.logger.error(f"Dictionary has no key '{key}'.")
                return Status.FAILURE
            value = value[part]

        if value is None:
            self.logger.error(f"The value of key '{key}' is null.")
            return Status.FAILURE

        # Copy the value so that changing it later does not change the dictionary.
        self._set_output("value", copy.deepcopy(value))
        return Status.SUCCESS


class SetDictValue(BehaviourWithPorts):
    """
    Sets the value of a key in a dictionary, returning a new dictionary (the input is left untouched).

    Any missing parents of a nested key are created.
    To build up a dictionary, wire `result` back into `dict` of the next SetDictValue.

    * If `dict` is not wired, it is treated as an empty dictionary.
    * If `value` has no data, for example because the behavior that produces it never ran,
      the value is set to null so that a written out dictionary shows that the value is missing.
    """

    INPUT_PORTS = {
        "dict": PortInformation(
            data_type=dict,
            required=False,
            default_value={},
            description="The dictionary to set the value in. Defaults to a new empty dictionary if not wired.",
        ),
        "key": PortInformation(data_type=str, required=True, description="The key, with dots separating nested keys."),
        "value": PortInformation(data_type=object, required=False, description="The value to set."),
    }
    OUTPUT_PORTS = {
        "result": PortInformation(data_type=dict, required=True, description="The dictionary with the value set."),
    }

    def update(self) -> Status:
        """Sets the value."""
        key = self.get_input("key")
        try:
            value = copy.deepcopy(self.get_input("value"))
        except NoDataAvailable:
            self.logger.warning(f"No data for key '{key}', so setting it to null.")
            value = None

        result = copy.deepcopy(self.get_input("dict"))
        *parents, name = key.split(".")
        parent = result
        for part in parents:
            parent = parent.setdefault(part, {})
            if not isinstance(parent, dict):
                self.logger.error(f"Cannot set key '{key}' because the value of '{part}' is not a dictionary.")
                return Status.FAILURE
        parent[name] = value

        self._set_output("result", result)
        return Status.SUCCESS
