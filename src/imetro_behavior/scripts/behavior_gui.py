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

import signal
import sys
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.node import Node
from rclpy.task import Future

from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from imetro_behavior_msgs.action import PreviewTrajectory


class RosWorker(QThread):
    """Worker thread to handle the rclpy spin loop so the GUI stays responsive."""

    def __init__(self, node: Node) -> None:
        super().__init__()
        self.node = node
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

    def run(self) -> None:
        try:
            self.executor.spin()
        except (ExternalShutdownException, KeyboardInterrupt):
            pass

    def quit(self) -> None:
        self.executor.remove_node(self.node)
        self.executor.shutdown()
        rclpy.try_shutdown()


class TrajectoryInterfaceNode(Node):
    """ROS 2 Node handling the interface to MoveIt trajectory previews and execution events."""

    def __init__(self, gui_signal: Signal) -> None:
        super().__init__("trajectory_gui_interface_node")
        self.gui_signal = gui_signal
        self.latest_trajectory = None
        self.latest_joint_state = JointState()

        self.joint_state_sub = self.create_subscription(JointState, "joint_states", self.joint_state_cb, 10)

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL  # latching
        self.display_pub = self.create_publisher(DisplayTrajectory, "display_planned_path", qos)
        self.stop_pub = self.create_publisher(String, "trajectory_execution_event", 10)

        self.action_server = ActionServer(
            self,
            PreviewTrajectory,
            "preview_trajectory",
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info("Trajectory preview ROS node is ready.")

    def goal_callback(self, goal_request: PreviewTrajectory.Goal) -> GoalResponse:
        """Accepts or rejects incoming goal requests."""
        self.get_logger().info("Received a new PreviewTrajectory action goal request.")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        """Handles external cancel requests."""
        self.get_logger().info("Received request to cancel trajectory goal.")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle: ServerGoalHandle) -> PreviewTrajectory.Result:
        """Asynchronously executes the action, yielding the thread cleanly until approved."""
        self.get_logger().info("Goal accepted. Awaiting user approval via GUI...")
        self.current_goal_handle = goal_handle
        self.latest_trajectory = goal_handle.request.trajectory
        self.gui_signal.emit(self.latest_trajectory)

        # Yield control and wait for a result from the GUI.
        result = PreviewTrajectory.Result()
        self.approve_future = Future()
        result.approved = await self.approve_future

        # Process the post-await state
        if not goal_handle.is_active:
            self.get_logger().info("Action was aborted or canceled.")
            goal_handle.abort()
        else:
            if result.approved:
                self.get_logger().info("Trajectory approved by user. Finalizing action goal.")
            else:
                self.get_logger().info("Trajectory rejected by user. Finalizing action goal.")
            goal_handle.succeed()

        self.current_goal_handle = None
        return result

    def accept_current_trajectory(self) -> None:
        """Registers that the trajectory was accepted to convey to the action execution callback."""
        if self.approve_future and not self.approve_future.done():
            self.approve_future.set_result(True)

    def reject_current_trajectory(self) -> None:
        """Registers that the trajectory was rejected to convey to the action execution callback."""
        if self.approve_future and not self.approve_future.done():
            self.approve_future.set_result(False)

    def joint_state_cb(self, msg: JointState) -> None:
        """Stores the latest joint state for use when sending preview requests."""
        self.latest_joint_state = msg

    def display_trajectory(self) -> None:
        """Publishes to RViz to display a trajectory."""
        if self.latest_trajectory is None:
            self.get_logger().warning("No trajectory to display. Skipping.")
            return

        display_msg = DisplayTrajectory(trajectory=[self.latest_trajectory])
        display_msg.trajectory_start.joint_state = self.latest_joint_state
        self.display_pub.publish(display_msg)

    def stop_motion(self) -> None:
        """Publishes to a Move Group node to stop motion execution."""
        self.stop_pub.publish(String(data="stop"))


class TrajectoryGUIPanel(QWidget):
    """Main UI window."""

    ros_msg_signal = Signal(RobotTrajectory)

    def __init__(self) -> None:
        super().__init__()

        # Initialize ROS 2
        rclpy.init()
        self.ros_node = TrajectoryInterfaceNode(self.ros_msg_signal)
        self.ros_worker = RosWorker(self.ros_node)
        self.ros_worker.start()

        # Connect ROS signal to GUI updater
        self.ros_msg_signal.connect(self.handle_new_trajectory)

        self.init_layout()

    def init_layout(self) -> None:
        """Sets up the GUI layout"""
        window_title = "Trajectory Preview"
        ns_stripped = self.ros_node.get_namespace().lstrip("/")
        if ns_stripped:
            window_title += f" ({ns_stripped})"
        self.setWindowTitle(window_title)
        self.resize(500, 200)

        layout = QVBoxLayout()

        button_style = "font-size: 20px; font-weight: bold; padding: 25px;"

        self.status_label = QLabel("Waiting for trajectory")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: gray;")
        layout.addWidget(self.status_label)

        self.btn_accept = QPushButton("Accept Preview")
        self.btn_accept.setStyleSheet(button_style)
        self.btn_accept.clicked.connect(self.on_accept_clicked)
        layout.addWidget(self.btn_accept)

        self.btn_replay = QPushButton("Replay Preview")
        self.btn_replay.setStyleSheet(button_style)
        self.btn_replay.clicked.connect(self.on_replay_clicked)
        layout.addWidget(self.btn_replay)

        self.btn_reject = QPushButton("Reject Preview")
        self.btn_reject.setStyleSheet(button_style)
        self.btn_reject.clicked.connect(self.on_reject_clicked)
        layout.addWidget(self.btn_reject)

        self.btn_stop = QPushButton("Stop Motion")
        self.btn_stop.setStyleSheet("background-color: #d9534f; color: white; " + button_style)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        layout.addWidget(self.btn_stop)

        self.set_trajectory_button_state(False)
        self.setLayout(layout)

    def set_trajectory_button_state(self, state: bool) -> None:
        """Helper to manage the state of the trajectory preview related buttons."""
        self.btn_accept.setEnabled(state)
        self.btn_replay.setEnabled(state)
        self.btn_reject.setEnabled(state)
        if state:
            self.status_label.setText("Trajectory received")
            self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: blue;")
        else:
            self.status_label.setText("Waiting for trajectory")
            self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: gray;")

    @Slot(RobotTrajectory)
    def handle_new_trajectory(self, msg: RobotTrajectory) -> None:
        """Called when a new trajectory message comes in over ROS."""
        self.set_trajectory_button_state(True)

        # Make sure the window is brought to the top to notify users
        self.raise_()
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show()
        # Turn the hint back off immediately so the window isn't permanently stuck on top
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

    def on_accept_clicked(self):
        """Called when the user clicks the 'Accept' button."""
        self.ros_node.get_logger().info("Accepting trajectory...")
        self.ros_node.accept_current_trajectory()
        self.set_trajectory_button_state(False)

    def on_reject_clicked(self):
        """Called when the user clicks the 'Reject' button."""
        if self.ros_node.latest_trajectory is not None:
            self.ros_node.get_logger().info("Rejecting trajectory...")
            self.ros_node.reject_current_trajectory()
            self.set_trajectory_button_state(False)

    def on_replay_clicked(self):
        """Called when the user clicks the 'Replay' button."""
        self.ros_node.get_logger().info("Replay preview requested...")
        self.ros_node.display_trajectory()

    def on_stop_clicked(self):
        """Called when the user clicks the 'Stop' button."""
        self.ros_node.get_logger().info("Stop motion requested!")
        self.ros_node.stop_motion()

    def closeEvent(self, event):
        """Ensure clean shutdown of ROS when closing the window."""
        self.ros_worker.quit()
        self.ros_worker.wait()
        event.accept()


def sigint_handler(*args):
    """Helps the application shut down more robustly on interrupt from a user."""
    QApplication.quit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    app = QApplication(sys.argv)

    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)  # To wake up interpreter to help shutdown

    gui = TrajectoryGUIPanel()
    gui.show()
    sys.exit(app.exec())
