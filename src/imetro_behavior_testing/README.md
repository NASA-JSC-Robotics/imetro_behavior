# imetro_behavior_testing

Runs a behavior tree once per trial of a test file, and saves the results of each trial.

With a behavior tree executor running:

```bash
ros2 run imetro_behavior_testing run_behavior_trials.py my_test.yaml --output-dir test_results
```

## Test files

```yaml
behavior: my_trial_tree

# Set on the tree's main BehaviorTree for every trial.
input_mappings:
  params_file: ${params_file}
  results_file: ${results_file}

stop_on_failure: true  # End the run if a trial's tree fails (default true)
repeats: 3             # Default 1, and a sweep can override it

params:
  speed: 0.02
  offset_xyz: [0.0, 0.0, 0.0]

sweeps:
  - name: baseline
  - name: fast_offset_x
    params:
      speed: 0.04
    vary:
      offset_xyz[0]: [-0.01, 0.01]
```

Each sweep runs every combination of its `vary` values.
A parameter in a sweep can be nested (`a.b`) or a list item (`a[0]`), and must exist in the top level `params`.
Trials are named `<sweep>_<values>_run_<repeat>`, such as `fast_offset_x_-0.01_run_0`.

Input mappings can use `${trial_name}`, `${trial_dir}`, `${params_file}`, `${results_file}`, and `${session_dir}`.
The tree can read its parameters with `ReadYamlFile` and `GetDictValue`, and write results with `SetDictValue` and `WriteYamlFile`.

## Results

Each run gets a timestamped folder with a copy of the test file, and a folder per trial with:

* `params.yaml`: the trial's parameters.
* `results.yaml`: whatever the tree wrote to `${results_file}`.
* `trial.yaml`: the tree's `status` and `message`, plus the params and results. Written when the trial ends.
