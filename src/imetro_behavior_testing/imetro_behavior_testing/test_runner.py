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

"""Runs the trials of a behavior test file. See the README of this package for the test file format."""

import copy
import re
import shutil
from datetime import datetime
from itertools import product
from pathlib import Path
from string import Template
from typing import Any

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from diagnostic_msgs.msg import KeyValue
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future

from imetro_behavior_msgs.action import ExecuteBehavior

TRIAL_PLACEHOLDERS = (
    "trial_name",  # The name of the trial.
    "trial_dir",  # The folder of the trial's results.
    "params_file",  # The YAML file of the trial's parameters, written before the trial.
    "results_file",  # The YAML file the tree can write results to, which are copied into the trial summary.
    "session_dir",  # The folder of the results of the whole test run.
)

SUCCEEDED = "succeeded"  # The status of a trial whose behavior succeeded.

TEST_FILE_KEYS = {"behavior", "input_mappings", "stop_on_failure", "repeats", "params", "sweeps"}
SWEEP_KEYS = {"name", "repeats", "params", "vary"}

_PATH_PART = re.compile(r"([^.\[\]]+)((?:\[\d+\])*)")


def _parse_param_path(path: str) -> list[str | int]:
    """Splits a parameter path such as `a.b[1]` into its keys and indices, such as `['a', 'b', 1]`."""
    steps: list[str | int] = []
    for part in path.split("."):
        match = _PATH_PART.fullmatch(part)
        if match is None:
            raise ValueError(f"Malformed parameter name '{path}'.")
        steps.append(match.group(1))
        steps.extend(int(index) for index in re.findall(r"\[(\d+)\]", match.group(2)))
    return steps


def set_param(params: dict[str, Any], path: str, value: Any) -> None:
    """
    Sets an existing parameter, addressed by a path such as `a.b[1]`, in place.

    Raises:
        ValueError: If the parameter does not exist.
    """
    steps = _parse_param_path(path)
    container: Any = params
    for i, step in enumerate(steps):
        exists = (
            isinstance(container, dict) and step in container
            if isinstance(step, str)
            else isinstance(container, list) and step < len(container)
        )
        if not exists:
            raise ValueError(f"Parameter '{path}' does not exist in the top level params.")
        if i < len(steps) - 1:
            container = container[step]
    container[steps[-1]] = value


def _format_value(value: Any) -> str:
    """Formats a varied value for a trial name."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return re.sub(r"\s+", "", str(value))
    return f"{value:g}"


def _mapping_to_string(value: Any) -> str:
    """Converts an input mapping value from YAML to a string, as it would be written in the tree XML."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return ", ".join(_mapping_to_string(v) for v in value)
    return str(value)


def expand_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expands the sweeps of a test file into a list of trials.

    Args:
        config: The test file contents.

    Returns:
        The trials in order, each with a unique `name` and its `params`.

    Raises:
        ValueError: If the test file is invalid, or two trials have the same name.
    """
    base_params = config.get("params") or {}
    sweeps: list[dict[str, Any]] = config.get("sweeps") or [{"name": "baseline"}]
    trials: list[dict[str, Any]] = []
    for sweep in sweeps:
        name = sweep.get("name")
        if not name:
            raise ValueError("Every sweep must have a name.")
        unknown = set(sweep) - SWEEP_KEYS
        if unknown:
            raise ValueError(f"Unknown keys in sweep '{name}': {sorted(unknown)}")

        sweep_params = copy.deepcopy(base_params)
        for path, value in (sweep.get("params") or {}).items():
            set_param(sweep_params, path, value)

        vary = sweep.get("vary") or {}
        for path, values in vary.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"The values of '{path}' in sweep '{name}' must be a nonempty list.")

        for values in product(*vary.values()):
            params = copy.deepcopy(sweep_params)
            for path, value in zip(vary, values):
                set_param(params, path, value)
            label = "_".join(_format_value(value) for value in values)
            for repeat in range(sweep.get("repeats", config.get("repeats", 1))):
                trial_name = "_".join(part for part in (name, label, f"run_{repeat}") if part)
                trials.append({"name": trial_name, "params": params})

    names: list[str] = [trial["name"] for trial in trials]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate trial names: {duplicates}")
    return trials


def resolve_input_mappings(config: dict[str, Any], placeholders: dict[str, str]) -> dict[str, str]:
    """
    Fills in the placeholders of a test file's input mappings.

    Args:
        config: The test file contents.
        placeholders: The value of each of TRIAL_PLACEHOLDERS for a trial.

    Returns:
        The value of each input mapping, as it would be written in the tree XML.

    Raises:
        ValueError: If a mapping uses an unknown placeholder.
    """
    mappings = {}
    for key, value in (config.get("input_mappings") or {}).items():
        try:
            resolved = Template(_mapping_to_string(value)).substitute(placeholders)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Input mapping '{key}' has an unknown placeholder {e}, the options are {list(TRIAL_PLACEHOLDERS)}."
            ) from e
        mappings[str(key)] = resolved
    return mappings


def load_test_file(test_file: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Loads and checks a test file.

    Returns:
        The test file contents, and its trials as from `expand_trials`.

    Raises:
        ValueError: If the test file is invalid.
    """
    config = yaml.safe_load(Path(test_file).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Test file {test_file} does not contain a mapping.")  # noqa: TRY004 -- every invalid test file raises ValueError
    unknown = set(config) - TEST_FILE_KEYS
    if unknown:
        raise ValueError(f"Unknown keys in test file {test_file}: {sorted(unknown)}")
    if not isinstance(config.get("behavior"), str) or not config["behavior"]:
        raise ValueError(f"Test file {test_file} must name a behavior tree to run in `behavior`.")

    # Check the input mappings up front, rather than partway through the test run.
    resolve_input_mappings(config, {placeholder: "" for placeholder in TRIAL_PLACEHOLDERS})
    return config, expand_trials(config)


class BehaviorTrialRunner:
    """Runs behavior test files trial by trial, using an ExecuteBehavior action client."""

    def __init__(self, node: Node, action_name: str = "/execute_behavior", server_timeout: float = 10.0):
        """
        Initializes the runner.

        Args:
            node: The ROS node to create the action client on. The runner spins it while waiting.
            action_name: The name of the behavior tree executor's ExecuteBehavior action.
            server_timeout: Seconds to wait for the action server at the start of a test run.
        """
        self._node = node
        self._logger = node.get_logger()
        self._server_timeout = server_timeout
        self._action_name = action_name
        self._client = ActionClient(node, ExecuteBehavior, action_name)
        self._goal_handle = None

    def _wait_for(self, future: Future) -> Any:
        # Spin in short slices, since a KeyboardInterrupt is only raised once rclpy returns to Python.
        while not future.done():
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=0.1)
        return future.result()

    def run(self, test_file: str | Path, output_dir: str | Path = "test_results") -> bool:
        """
        Runs all the trials of a test file, unless one fails and the test file says to stop on failure.

        Args:
            test_file: Path of the test file.
            output_dir: Folder in which to create the timestamped folder for this test run.

        Returns:
            Whether every trial ran and its behavior succeeded. A behavior can succeed and still report
            in its results that the trial did not go as hoped.

        Raises:
            ValueError: If the test file is invalid.
            RuntimeError: If the action server is not available.
            KeyboardInterrupt: If interrupted, after canceling the running trial.
        """
        config, trials = load_test_file(test_file)
        stop_on_failure = config.get("stop_on_failure", True)

        if not self._client.wait_for_server(timeout_sec=self._server_timeout):
            raise RuntimeError(f"Action server {self._action_name} is not available.")

        session_dir = Path(output_dir).absolute() / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True)
        shutil.copyfile(test_file, session_dir / "test_file.yaml")
        self._logger.info(f"Running {len(trials)} trials of {config['behavior']}, saving results to {session_dir}.")

        statuses = []
        for index, trial in enumerate(trials):
            self._logger.info(f"Starting trial {index + 1} of {len(trials)}: {trial['name']}.")
            status = self.run_trial(config, trial, session_dir)
            statuses.append(status)
            self._logger.info(f"Trial {trial['name']} {status}.")
            if status != SUCCEEDED and stop_on_failure:
                self._logger.error(f"Ending the test run after {len(statuses)} of {len(trials)} trials.")
                break

        self._logger.info(
            f"Ran {len(statuses)} of {len(trials)} trials, of which {statuses.count(SUCCEEDED)} succeeded. "
            f"Results are in {session_dir}."
        )
        return statuses.count(SUCCEEDED) == len(trials)

    def run_trial(self, config: dict[str, Any], trial: dict[str, Any], session_dir: Path) -> str:
        """
        Runs one trial and writes its summary.

        Returns:
            The status of the behavior.
        """
        trial_dir = session_dir / trial["name"]
        trial_dir.mkdir()
        params_file = trial_dir / "params.yaml"
        results_file = trial_dir / "results.yaml"
        params_file.write_text(yaml.safe_dump(trial["params"], sort_keys=False))

        goal = ExecuteBehavior.Goal()
        goal.tree_file_name = config["behavior"]
        mappings = resolve_input_mappings(
            config,
            {
                "trial_name": trial["name"],
                "trial_dir": str(trial_dir),
                "params_file": str(params_file),
                "results_file": str(results_file),
                "session_dir": str(session_dir),
            },
        )
        goal.input_mappings = [KeyValue(key=key, value=value) for key, value in mappings.items()]

        try:
            status, message = self._execute(goal)
        except KeyboardInterrupt:
            self._cancel()
            self._write_summary(trial, trial_dir, results_file, "canceled", "Interrupted by the user.")
            raise

        if message:
            self._logger.info(f"Behavior message: {message}")
        self._write_summary(trial, trial_dir, results_file, status, message)
        return status

    def _execute(self, goal: ExecuteBehavior.Goal) -> tuple[str, str]:
        """Sends the goal and waits for its result."""
        self._goal_handle = self._wait_for(self._client.send_goal_async(goal))
        if not self._goal_handle.accepted:
            self._goal_handle = None
            return "rejected", "The behavior tree executor rejected the goal, so it may be running another behavior."

        response = self._wait_for(self._goal_handle.get_result_async())
        self._goal_handle = None
        result = response.result
        if response.status == GoalStatus.STATUS_SUCCEEDED:
            return (SUCCEEDED if result.success else "failed"), result.message
        if response.status == GoalStatus.STATUS_CANCELED:
            return "canceled", result.message
        return "aborted", result.message

    def _cancel(self):
        """Cancels the running trial, if any, and waits for it to stop."""
        if self._goal_handle is None:
            return
        self._logger.warning("Canceling the running trial.")
        self._wait_for(self._goal_handle.cancel_goal_async())
        rclpy.spin_until_future_complete(self._node, self._goal_handle.get_result_async(), timeout_sec=10.0)
        self._goal_handle = None

    def _write_summary(self, trial: dict[str, Any], trial_dir: Path, results_file: Path, status: str, message: str):
        results = None
        if results_file.exists():
            try:
                results = yaml.safe_load(results_file.read_text())
            except yaml.YAMLError as e:
                self._logger.error(f"Could not read the results in {results_file}: {e}")

        summary = {
            "name": trial["name"],
            "status": status,
            "message": message,
            "params": trial["params"],
            "results": results,
        }
        (trial_dir / "trial.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
