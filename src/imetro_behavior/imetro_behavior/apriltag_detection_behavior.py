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

import threading
import time
from py_trees.common import Status
from py_trees.ports import BehaviourWithPorts, PortInformation
from geometry_msgs.msg import PoseStamped
from pupil_apriltags import Detector
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R


class DetectAprilTagLocal(BehaviourWithPorts):
    """An Apriltag detector background thread that handles retries until success or maximum attempts are reached."""

    @classmethod
    def input_ports(cls) -> dict:
        return {
            "rgb_image": PortInformation(data_type=object, required=True),
            "camera_info": PortInformation(data_type=object, required=True),
            "tag_id": PortInformation(data_type=int, required=True),
            "tag_size": PortInformation(data_type=float, required=True),
        }

    @classmethod
    def output_ports(cls) -> dict:
        return {
            "tag_pose": PortInformation(data_type=PoseStamped, required=True),
        }

    def setup(self, **kwargs):
        self.node = kwargs.get("node")
        self.bridge = CvBridge()
        self.detector = Detector(families="tag36h11", nthreads=2, quad_decimate=3.0)
        
        self.lock = threading.Lock()
        self.thread = None
        self.result_pose = None
        self.is_running = False
        self.has_finished = False
        self.success = False
        
        # Retry tracking
        self.max_attempts = 10
        self.attempt_count = 0

    def initialise(self):
        with self.lock:
            self.result_pose = None
            self.is_running = False
            self.has_finished = False
            self.success = False
            self.attempt_count = 0

    def update(self) -> Status:
            with self.lock:
                if self.has_finished:
                    if self.success:
                        # Use the correct py_trees internal method
                        self._set_output("tag_pose", self.result_pose)
                        return Status.SUCCESS
                    else:
                        # Reset finished flag so it can retry on the next ticks if allowed
                        self.has_finished = False

                if self.is_running:
                    return Status.RUNNING

                # Check if we've exhausted our attempts
                if self.attempt_count >= self.max_attempts:
                    if self.node:
                        self.node.get_logger().error("DetectAprilTagLocal: Exceeded max attempts to find tag.")
                    return Status.FAILURE

                self.attempt_count += 1
                
                rgb_msg = self.get_input("rgb_image")
                camera_info = self.get_input("camera_info")
                target_id = self.get_input("tag_id")
                tag_size = self.get_input("tag_size")

                if rgb_msg is None or camera_info is None:
                    return Status.RUNNING  # Keep running until image data arrives

                self.is_running = True
                self._run_detection(rgb_msg, camera_info, target_id, tag_size)
                # self.thread = threading.Thread(
                #     target=self._run_detection, 
                #     args=(rgb_msg, camera_info, target_id, tag_size)
                # )
                # self.thread.daemon = True
                # self.thread.start()
                return Status.RUNNING

    def _run_detection(self, rgb_msg, camera_info, target_id, tag_size):
        start_time = time.perf_counter()
        try:
            
            image_det_start = time.perf_counter()
            cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="mono8")
            fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
            camera_params = (fx, fy, cx, cy)
            image_det_time = time.perf_counter() - image_det_start

            detection_start = time.perf_counter()
            tags = self.detector.detect(
                cv_img, 
                estimate_tag_pose=True, 
                camera_params=camera_params, 
                tag_size=tag_size
            )
            detection_time = time.perf_counter() - detection_start 
            

            target_tag = next((t for t in tags if t.tag_id == target_id), None)
            if target_tag is not None:
                pose_msg = PoseStamped()
                pose_msg.header = rgb_msg.header
                pose_msg.pose.position.x = target_tag.pose_t[0][0]
                pose_msg.pose.position.y = target_tag.pose_t[1][0]
                pose_msg.pose.position.z = target_tag.pose_t[2][0]

                if self.node:
                 self.node.get_logger().info(
                    f"Found AprilTag ID {target_id}! Position -> "
                    f"x: {pose_msg.pose.position.x:.3f}, "
                    f"y: {pose_msg.pose.position.y:.3f}, "
                    f"z: {pose_msg.pose.position.z:.3f}"
                 )
                 self.node.get_logger().info(
                    f"Image conversion took {image_det_time:.4f} seconds!"
                 )
                 self.node.get_logger().info(f"Apriltag detection took {detection_time:.4f} seconds!")    
                 self.node.get_logger().info(f"The size of image {cv_img.shape}") 
                r = R.from_matrix(target_tag.pose_R)
                q = r.as_quat()
                pose_msg.pose.orientation.x = q[0]
                pose_msg.pose.orientation.y = q[1]
                pose_msg.pose.orientation.z = q[2]
                pose_msg.pose.orientation.w = q[3]

                with self.lock:
                    self.result_pose = pose_msg
                    self.success = True
            else:
                with self.lock:
                    self.success = False
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"AprilTag detection error: {e}")
            with self.lock:
                self.success = False
        finally:
            total_time = time.perf_counter() - start_time
            self.node.get_logger().info(f"Apriltag detection total process took {total_time:.4f} seconds!") 
            with self.lock:
                self.is_running = False
                self.has_finished = True

    # def terminate(self, new_status: Status):
    #     # In the case the behavior is interrupted or cancelled mid-execution
    #     with self.lock:
    #         self.is_running = False
    #     if self.thread is not None and self.thread.is_alive():
    #         self.thread.join(timeout=1.5)