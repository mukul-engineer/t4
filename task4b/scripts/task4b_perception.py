#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script detects ArUco markers and bad fruits,
*  estimates their 3D positions from RGB + Depth data,
*  transforms them to base_link frame, and publishes TFs.
*
*****************************************************************************************
'''

# Team ID:          [ eYRC#5167 ]
# Author List:      [ Your Team Member Names ]
# Filename:         task1b_fruits_tf.py
# Functions:        [ caminfo_callback, synced_callback, main ]
# Nodes:            fruit_pose_publisher
# Publishing Topics - [ /tf ]
# Subscribing Topics - [ /camera/image_raw, /camera/depth/image_raw, /camera/camera_info ]

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2, numpy as np
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PoseStamped
import tf2_geometry_msgs.tf2_geometry_msgs as tf2_geom
from message_filters import Subscriber, ApproximateTimeSynchronizer

class FruitPosePublisher(Node):
    """
    ROS2 Node for detecting bad fruits and ArUco markers,
    and publishing their TFs in base_link frame.
    """

    def __init__(self):
        super().__init__('fruit_pose_publisher')

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.team_id = "5167"

        # TF broadcaster + listener
        self.br = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Camera info
        self.sub_caminfo = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.caminfo_callback, 10
        )

        # RGB + Depth synchronizer
        self.sub_img = Subscriber(self, Image, '/camera/image_raw')
        self.sub_depth = Subscriber(self, Image, '/camera/depth/image_raw')
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_img, self.sub_depth], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

    # ------------------- CAMERA INFO CALLBACK -------------------
    def caminfo_callback(self, msg: CameraInfo):
        self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        self.get_logger().info("Camera intrinsics received.")
        self.destroy_subscription(self.sub_caminfo)

    # ------------------- SYNCED RGB + DEPTH CALLBACK -------------------
    def synced_callback(self, img_msg: Image, depth_msg: Image):
        if self.fx is None:
            return

        frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        # ==============================================================
        # ======================= ARUCO MARKER DETECTION ==========================
        if not hasattr(self, "aruco_cache"):
            self.aruco_cache = {}  # store last known ArUco poses

        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        detected_ids = []

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, corner in enumerate(corners):
                marker_id = int(ids[i])
                detected_ids.append(marker_id)
                u, v = np.mean(corner[0], axis=0).astype(int)
                Z = float(np.nanmedian(depth[v-2:v+2, u-2:u+2]))
                if np.isnan(Z) or Z <= 0.1:
                    continue

                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
                X_link, Y_link, Z_link = Z, -X, -Y   # same pattern as fruits

                pose_cam = PoseStamped()
                pose_cam.header.stamp = self.get_clock().now().to_msg()
                pose_cam.header.frame_id = "camera_link"
                pose_cam.pose.position.x = float(X_link)
                pose_cam.pose.position.y = float(Y_link)
                pose_cam.pose.position.z = float(Z_link)
                pose_cam.pose.orientation.x = 0.6533 
                pose_cam.pose.orientation.y = -0.6533 
                pose_cam.pose.orientation.z = 0.2706
                pose_cam.pose.orientation.w = -0.2706

                try:
                    pose_base = self.tf_buffer.transform(
                        pose_cam, "base_link", timeout=rclpy.duration.Duration(seconds=0.1)
                    )

                    # cache the pose
                    self.aruco_cache[marker_id] = pose_base

                except Exception as e:
                    self.get_logger().warn(f"Aruco transform failed: {e}")

        # ---------- Re-broadcast all cached ArUco TFs (even if not visible) ----------
        for marker_id, pose_base in self.aruco_cache.items():
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            if marker_id == 3 :
                t.child_frame_id = f"{self.team_id}_fertilizer_1"
            else:
                t.child_frame_id = f"{self.team_id}_aruco_{marker_id}"       
            t.transform.translation.x = pose_base.pose.position.x
            t.transform.translation.y = pose_base.pose.position.y
            t.transform.translation.z = pose_base.pose.position.z
            t.transform.rotation = pose_base.pose.orientation
            self.br.sendTransform(t)


        # ==============================================================
        # 🍎 BAD FRUIT DETECTION AND TF PUBLISHING
        # ==============================================================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([15, 23, 71])
        upper_gray = np.array([18, 118, 255])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            u, v = int(x + w / 2), int(y + h / 2)
            Z = np.nanmedian(depth[y:y + h, x:x + w])
            if np.isnan(Z) or Z <= 0.1:
                continue

            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy
            X_link, Y_link, Z_link = Z, -X, -Y

            qx, qy, qz, qw = -0.6533, 0.6533, -0.2706, 0.2706
            pose_cam = PoseStamped()
            pose_cam.header.stamp = self.get_clock().now().to_msg()
            pose_cam.header.frame_id = "camera_link"
            pose_cam.pose.position.x = float(X_link)
            pose_cam.pose.position.y = float(Y_link)
            pose_cam.pose.position.z = float(Z_link)
            pose_cam.pose.orientation.x = qx
            pose_cam.pose.orientation.y = qy
            pose_cam.pose.orientation.z = qz
            pose_cam.pose.orientation.w = qw

            fruit_id += 1
            fruit_frame = f"{self.team_id}_bad_fruit_{fruit_id}"

            try:
                pose_base = self.tf_buffer.transform(
                    pose_cam, "base_link", timeout=rclpy.duration.Duration(seconds=0.1)
                )
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "base_link"
                t.child_frame_id = fruit_frame
                t.transform.translation.x = pose_base.pose.position.x
                t.transform.translation.y = pose_base.pose.position.y
                t.transform.translation.z = pose_base.pose.position.z
                t.transform.rotation = pose_base.pose.orientation
                self.br.sendTransform(t)
            except Exception as e:
                self.get_logger().warn(f"Fruit TF transform failed: {e}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "bad fruit", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ==============================================================
        # 🪞 Visualization
        # ==============================================================
        cv2.imshow("Detection", frame)
        cv2.waitKey(1)



# ------------------- MAIN -------------------
def main(args=None):
    rclpy.init(args=args)
    node = FruitPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()