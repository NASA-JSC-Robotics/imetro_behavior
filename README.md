# imetro_behavior

Python based robotics behavior stack for iMETRO.

> [!WARNING]
> Work in progress!

## Getting Started

This repo uses [Pixi](https://pixi.prefix.dev/latest/) to manage the environment, but can be brought into other projects as a regular set of ROS packages.

To get started,

1. Install Pixi: <https://pixi.prefix.dev/latest/installation/>

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

## Implemented Behaviors

Generally behavior names should be self-explanatory, but to understand how to use it effectively be sure to check the header, where the inputs and outputs are defined.

* ROS Behaviors
  * [Ros Action Client Base](src/imetro_behavior/imetro_behavior/ros_behaviors/action_client.py)
  * [Get Synced Image PointCloud Depth](src/imetro_behavior/imetro_behavior/ros_behaviors/perception.py)
  * Ros Service Client Base
  * Call Trigger Service
* Basic Behaviors
  * [WaitForDuration](src/imetro_behavior/imetro_behavior/basic_behaviors.py)
* Color Behaviors
  * [Detect Color Blobs](src/imetro_behavior/imetro_behavior/color_behaviors.py)
* Control Behaviors
  * [Get Ros Controller Info](src/imetro_behavior/imetro_behavior/control_behaviors.py#34)
  * [Switch Ros Controllers](src/imetro_behavior/imetro_behavior/control_behaviors.py#67)
  * [Command Gripper](src/imetro_behavior/imetro_behavior/control_behaviors.py#148)
* Decorators
  * [SuccessIfVariableIsTrue](src/imetro_behavior/imetro_behavior/decorators.py#25)
  * [SuccessIfVariableIsFalse](src/imetro_behavior/imetro_behavior/decorators.py#50)
* Geometry Behaviors
  * [Create Pose Stamped](src/imetro_behavior/imetro_behavior/geometry_behaviors.py#37)
  * [Transform Pose](src/imetro_behavior/imetro_behavior/geometry_behaviors.py#72)
  * [Align Pose To Nearest Axis](src/imetro_behavior/imetro_behavior/geometry_behaviors.py#114)
  * [Offset Pose Stamped](src/imetro_behavior/imetro_behavior/geometry_behaviors.py#164)
  * [Yaml Pose To Pose Stamped](src/imetro_behavior/imetro_behavior/geometry_behaviors.py#208)
* MoveIt Behaviors
  * [Plan To Joint State](src/imetro_behavior/imetro_behavior/moveit_behaviors.py#85)
  * [Plan To Pose](src/imetro_behavior/imetro_behavior/moveit_behaviors.py#157)
  * [Plan Arc Path](src/imetro_behavior/imetro_behavior/moveit_behaviors.py#245)
  * [Request Trajectory Approval](src/imetro_behavior/imetro_behavior/moveit_behaviors.py#441)
  * [Execute Trajectory Behavior](src/imetro_behavior/imetro_behavior/moveit_behaviors.py#473)
* Nav Behaviors
  * [Navigate To Pose Behavior](src/imetro_behavior/imetro_behavior/nav_behaviors.py)
