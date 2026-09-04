"""
Python-based robotics behavior stack for iMETRO.

This package provides a library of reusable behavior tree nodes for
ROS 2 based robotic systems, built on top of [py_trees](https://py-trees.readthedocs.io/).

## Modules

- **apriltag_behaviors** — Behaviors to detect AprilTags
- **basic_behaviors** — General-purpose behaviors (e.g., `WaitForDuration`)
- **color_behaviors** — Behaviors to detect colored objects in images
- **control_behaviors** — Controller management, gripper commands, etc.
- **decorators** — Conditional decorators for behavior tree flow control
- **executor** — Behavior tree executor and lifecycle management
- **geometry_behaviors** — Pose creation, transforms, and TF operations
- **joint_behaviors** — Utilities to load joint states from files
- **mockup_behaviors** — Behaviors to interact with mockup state managers in ROS
- **moveit_behaviors** — Motion planning and trajectory execution via MoveIt
- **mujoco_behaviors** — Utilities for interacting with mujoco_ros2_control
- **nav_behaviors** — Navigation behaviors using Nav2
- **ros_behaviors** — Base classes for ROS action and service clients
"""
