#!/usr/bin/env python3
"""
Task 4B - Navigation Node
- Custom P-controller (NO Nav2, NO teleop)
- Waypoint following with detection stops
- Communicates with shape_detector and message_publisher
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32, Float32MultiArray
import math
import time


class Task4BNavigation(Node):
    def __init__(self):
        super().__init__('task4b_navigation')
        
        # ==================== PUBLISHERS ====================
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.collect_pub = self.create_publisher(String, '/collect_point', 10)
        self.detection_request_pub = self.create_publisher(String, '/detection_request', 10)
        
        # ==================== SUBSCRIBERS ====================
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float32, '/orientation', self.orientation_callback, 10)
        self.create_subscription(Float32MultiArray, '/ultrasonic_sensor_std_float', self.ultra_callback, 10)
        self.create_subscription(String, '/shape_detected', self.shape_callback, 10)
        
        # ==================== ROBOT STATE ====================
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.imu_yaw = 0.0
        self.ultra_left = 0.0
        self.ultra_right = 0.0
        
        # ==================== NAVIGATION PARAMETERS ====================
        self.kp_lin = 0.5
        self.kp_ang = 1.2
        self.dist_tolerance = 0.12
        self.yaw_tolerance = 0.1
        self.max_lin_vel = 0.22
        self.max_ang_vel = 0.8
        
        # ==================== MAIN WAYPOINTS (Navigation Path) ====================
        self.waypoints = [
            # Start from origin
            [0.002, -0.001, 0.0],
            
            # Lane 1: Detection points (moving right, +x direction)
            [1.066, -1.625, 0.0],      # WP1 - Near Plant 1
            [1.825, -1.605, 0.0],      # WP2 - Near Plant 2  
            [2.537, -1.657, 0.0],      # WP3 - Near Plant 3
            [3.285, -1.606, 0.0],      # WP4 - Near Plant 4
            [4.013, -1.543, 0.0],      # WP5 - End of lane 1
            
            # Corner turn to Lane 3
            [4.805, -0.900, 1.57],     # WP6 - Corner
            [4.653, 1.621, 1.57],      # WP7 - Corner
            
            # Lane 3: Detection points (moving left, -x direction)
            [3.939, 1.660, 3.14],      # WP8 - Near Plant 5
            [3.187, 1.715, 3.14],      # WP9 - Near Plant 6
            [2.450, 1.691, 3.14],      # WP10 - Near Plant 7
            [1.683, 1.691, 3.14],      # WP11 - Near Plant 8
            [1.002, 1.589, 3.14],      # WP12 - End of lane 3
            
            # Corner turn to Lane 2
            [0.554, 1.028, -1.57],     # WP13 - Corner
            
            # Lane 2: Navigation only (no plants)
            [1.019, 0.015, 0.0],       # WP14
            [2.515, 0.055, 0.0],       # WP15
            [3.987, 0.102, 0.0],       # WP16
            
            # Return to dock
            [2.118, -1.665, 0.0],      # WP17 - Home position
        ]
        
        self.current_wp_index = 0
        
        # ==================== SPECIAL WAYPOINTS (Exact Shape Positions) ====================
        # These are the EXACT positions in front of shapes for detection
        self.special_waypoints = [
            [1.459, -1.531, 0.0],      # Plant 1 shape position
            [2.133, -1.532, 0.0],      # Plant 2 shape position
            [2.892, -1.477, 0.0],      # Plant 3 shape position
            [3.636, -1.406, 0.0],      # Plant 4 shape position
            [3.671, 1.798, 3.14],      # Plant 5 shape position
            [2.833, 1.789, 3.14],      # Plant 6 shape position
            [2.074, 1.734, 3.14],      # Plant 7 shape position
            [1.386, 1.609, 3.14],      # Plant 8 shape position
        ]
        
        # ==================== STATE VARIABLES ====================
        self.special_index = 0
        self.special_active = False
        self.skip_next_special = False
        
        # Shape detection state
        self.shape_triggered = False
        self.shape_type = None
        self.shape_published = False
        
        # Detection waypoints (where we trigger /collect_point)
        self.detection_wps = {1, 2, 3, 4, 8, 9, 10, 11}  # WP indices for plants
        
        # ==================== STATE MACHINE ====================
        self.state = "NAVIGATE"  # NAVIGATE, WAITING_DETECTION, STOPPED, DONE
        self.stop_start_time = 0.0
        self.wait_for_detection = False
        
        # Timers (mutually exclusive like task3b)
        self.main_timer = self.create_timer(0.1, self.main_wp_callback)
        self.special_timer = None
        self.wait_timer = None
        
        # ==================== MAIN TIMER ====================
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("🚀 Task 4B Navigation Started")
        self.get_logger().info("=" * 50)

    # ==================== CALLBACKS ====================
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def orientation_callback(self, msg):
        self.imu_yaw = msg.data
        if self.imu_yaw > math.pi:
            self.imu_yaw -= 2 * math.pi

    def ultra_callback(self, msg):
        if len(msg.data) > 5:
            self.ultra_left = msg.data[4]
            self.ultra_right = msg.data[5]

    def shape_callback(self, msg):
        """Receive shape detection result (like task3b)"""
        incoming = msg.data
        
        # Handle wait state (from /collect_point)
        if self.wait_for_detection and incoming in ["BAD_HEALTH", "FERTILIZER_REQUIRED", "False"]:
            self.stop_robot()
            self.get_logger().info(f"✅ Received detection: {incoming}. Exiting wait state.")
            
            # Stop wait timer
            if self.wait_timer is not None:
                self.wait_timer.cancel()
                self.destroy_timer(self.wait_timer)
                self.wait_timer = None
            
            self.wait_for_detection = False
            
            # Check what was received
            if incoming in ["BAD_HEALTH", "FERTILIZER_REQUIRED"]:
                self.get_logger().warn(f"⚠️ Shape Triggered: {incoming}")
                self.shape_triggered = True
                self.shape_type = incoming
                self.shape_published = False
                self.special_active = True
                self.skip_next_special = False
                self.switch_to_special_mode()
            
            elif incoming == "False":
                self.get_logger().warn("⏭ Special waypoint skipped (False)")
                self.special_active = True
                self.skip_next_special = True
                self.switch_to_special_mode()
            
            return
        
        # Handle unplanned interrupts
        if incoming in ["BAD_HEALTH", "FERTILIZER_REQUIRED"]:
            if not self.special_active and not self.wait_for_detection:
                self.get_logger().warn(f"⚠️ Shape Triggered (Unplanned): {incoming}")
                self.shape_triggered = True
                self.shape_type = incoming
                self.shape_published = False
                self.special_active = True
                self.skip_next_special = False
                self.switch_to_special_mode()
        
        elif incoming == "False":
            if not self.special_active and not self.wait_for_detection:
                self.get_logger().warn("⏭ Special waypoint skipped (False) (Unplanned)")
                self.special_active = True
                self.skip_next_special = True
                self.switch_to_special_mode()

    # ==================== SWITCHING LOGIC (like task3b) ====================
    def switch_to_special_mode(self):
        """Disables main timer and enables special timer."""
        # Ensure main timer is stopped
        if self.main_timer is not None:
            self.main_timer.cancel()
            self.destroy_timer(self.main_timer)
            self.main_timer = None
            self.get_logger().info("🛑 Main WP Timer Stopped.")
        
        # Ensure wait timer is stopped
        if self.wait_timer is not None:
            self.wait_timer.cancel()
            self.destroy_timer(self.wait_timer)
            self.wait_timer = None
            self.wait_for_detection = False
        
        # Start special timer
        if self.special_timer is None:
            self.special_timer = self.create_timer(0.1, self.special_wp_callback)
            self.get_logger().info("▶ Special WP Timer Started.")

    def switch_to_main_mode(self):
        """Disables special timer and enables main timer."""
        # Reset flags
        self.special_active = False
        self.skip_next_special = False
        self.shape_triggered = False
        self.shape_type = None
        self.shape_published = False
        self.stop_robot()
        
        if self.special_timer is not None:
            self.special_timer.cancel()
            self.destroy_timer(self.special_timer)
            self.special_timer = None
            self.get_logger().info("🛑 Special WP Timer Stopped.")
        
        # Start main timer
        if self.main_timer is None:
            self.main_timer = self.create_timer(0.1, self.main_wp_callback)
            self.get_logger().info("▶ Main WP Timer Started.")

    def wait_for_detection_callback(self):
        """Timer callback for waiting state."""
        self.stop_robot()
        self.get_logger().info(f"⏳ Waiting for /shape_detected result at WP {self.current_wp_index}...")

    # ==================== SPECIAL WAYPOINT CALLBACK ====================
    def special_wp_callback(self):
        """Handles special waypoints (exact shape positions)."""
        if not self.special_active:
            self.switch_to_main_mode()
            return
        
        if self.skip_next_special:
            self.get_logger().info(f"⏭ Skipped special WP {self.special_index+1}")
            self.stop_robot()
            self.special_index += 1
            self.switch_to_main_mode()
            return
        
        if self.special_index >= len(self.special_waypoints):
            self.get_logger().warn("⚠️ All special waypoints exhausted. Switching to Main.")
            self.switch_to_main_mode()
            return
        
        # Navigate to special waypoint
        tx, ty, tyaw = self.special_waypoints[self.special_index]
        reached = self.navigate_to_waypoint(tx, ty, tyaw)
        
        if reached:
            self.stop_robot()
            self.get_logger().info(f"✅ Reached special WP {self.special_index}")
            
            # Stop for 2 seconds (required by task)
            import time
            stop_time = time.time()
            while time.time() - stop_time < 2.0:
                self.stop_robot()
                rclpy.spin_once(self, timeout_sec=0.01)
            
            # Publish detection
            if not self.shape_published and self.shape_type:
                plant_id = self.special_index + 1  # Plants 1-8
                msg = String()
                msg.data = f"{self.shape_type},{self.x:.2f},{self.y:.2f},{plant_id}"
                self.detection_request_pub.publish(msg)
                self.get_logger().info(f"📤 Published: {msg.data}")
                self.shape_published = True
            
            self.special_index += 1
            self.switch_to_main_mode()

    # ==================== MAIN WAYPOINT CALLBACK ====================
    def main_wp_callback(self):
        """Handles main waypoints (navigation path)."""
        if self.special_active or self.wait_for_detection:
            return
        
        if self.current_wp_index >= len(self.waypoints):
            # Add dock station detection at end
            self.stop_robot()
            import time
            time.sleep(2.0)
            dock_msg = String()
            dock_msg.data = f"DOCK_STATION,{self.x:.2f},{self.y:.2f},0"
            self.detection_request_pub.publish(dock_msg)
            self.get_logger().info(f"🏁 Published DOCK_STATION: {dock_msg.data}")
            self.get_logger().info("🎉 Navigation Complete!")
            return
        
        tx, ty, tyaw = self.waypoints[self.current_wp_index]
        reached = self.navigate_to_waypoint(tx, ty, tyaw)
        
        if reached:
            self.get_logger().info(f"✅ Reached main WP {self.current_wp_index}")
            
            # Check if this is a detection waypoint
            if self.current_wp_index in self.detection_wps:
                cp_msg = String()
                cp_msg.data = "true"
                self.collect_pub.publish(cp_msg)
                self.get_logger().info("📡 /collect_point → true. Waiting for /shape_detected...")
                
                # Switch to wait mode
                self.wait_for_detection = True
                
                if self.main_timer is not None:
                    self.main_timer.cancel()
                    self.destroy_timer(self.main_timer)
                    self.main_timer = None
                
                self.wait_timer = self.create_timer(0.5, self.wait_for_detection_callback)
                self.current_wp_index += 1
                return
            
            # Regular waypoint
            self.current_wp_index += 1

    # ==================== NAVIGATION ====================
    def navigate_to_waypoint(self, tx, ty, tyaw):
        dx = tx - self.x
        dy = ty - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        cmd = Twist()
        
        # Phase 1: Move to position
        if distance > self.dist_tolerance:
            path_angle = math.atan2(dy, dx)
            ang_error = self.normalize_angle(path_angle - self.yaw)
            
            cmd.linear.x = self.kp_lin * distance
            cmd.angular.z = self.kp_ang * ang_error
            
            # Clamp
            cmd.linear.x = max(min(cmd.linear.x, self.max_lin_vel), -self.max_lin_vel)
            cmd.angular.z = max(min(cmd.angular.z, self.max_ang_vel), -self.max_ang_vel)
            
            # Slow down during turns
            if abs(ang_error) > 0.5:
                cmd.linear.x *= 0.3
            
            self.cmd_pub.publish(cmd)
            return False
        
        # Phase 2: Adjust yaw
        yaw_error = self.normalize_angle(tyaw - self.yaw)
        if abs(yaw_error) > self.yaw_tolerance:
            cmd.angular.z = self.kp_ang * yaw_error
            cmd.angular.z = max(min(cmd.angular.z, self.max_ang_vel), -self.max_ang_vel)
            self.cmd_pub.publish(cmd)
            return False
        
        self.stop_robot()
        return True

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = Task4BNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
