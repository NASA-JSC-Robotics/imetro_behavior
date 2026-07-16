# imetro_behavior

Python based robotics behavior stack for iMETRO.

> [!WARNING]
> Work in progress!

## Getting Started

This repo uses [Pixi](https://pixi.prefix.dev/latest/) to manage the environment, but can be brought into other projects as a regular set of ROS packages.

To get started,

1. Install Pixi: https://pixi.prefix.dev/latest/installation/

2. `pixi run build`

Then, you can open up a `pixi shell` and do all your development and testing there!


## Running the Core

Once you have your workspace built, you can run the 2 main applications.

```bash
ros2 run imetro_behavior run_behavior.py

ros2 run imetro_behavior behavior_gui.py
```

Or launch them both together!

```bash
ros2 launch imetro_behavior run_behavior.launch.xml gui:=true
```

One your behavior tree executor is up and running, you can send it an action goal.
For example,

```bash
ros2 action send_goal /execute_behavior imetro_behavior_msgs/action/ExecuteBehavior '{tree_file_name: nav_tree}'
```
