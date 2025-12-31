#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from tf_transformations import euler_from_quaternion
import time
from std_msgs.msg import String


class PController(Node):
    def __init__(self):
        super().__init__('p_controller_waypoints')

        # Publishers
        self.det_pub = self.create_publisher(String, '/detection_status', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.collect_pub = self.create_publisher(String, '/collect_point', 10)

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(String, '/shape_detected', self.shape_callback, 10)

        # Main Waypoints (NOT REMOVED)
        self.waypoints = [
            [-1.1617, -6.4596, 0.361],
            [0.121 , -6.290, -0.002],
            [0.263, -4.7, 1.40],
            [0.263,-3.406, 1.400],
            [0.26, -1.95, 1.57], 
            [0.333, -0.621, 1.57],
            [0.225, 1.097, 1.4018],
            [-0.234, 1.609, 2.446],
            [-1.4378, 1.5227, -3.1264], #wp8
            [-3.0358 , 1.5761 , -3.1332],
            [-3.3196 , 0.7429 , -1.5975],           
            [-3.3562, -0.4887, -1.45],
            [-3.2793, -2.0259, -1.45],
            [-3.3688, -3.3330, -1.45],
            [-3.4308, -5.1546, -1.45],
            [-3.4120, -5.5187, -1.5297],
            [-3.2824, -5.34898, -0.0347],
            [-1.3189, -5.3980, -0.0347],
            [-1.5056, -5.2538, 1.5643],
            [-1.5057, -4.8314, 1.5643],
            [-1.5656, -4.9678, 1.4920],
            [-1.5339 , -6.6156, -1.57],
            [-1.3533, 1.2639, 1.57]
            
        ]
        self.current_wp_index = 0

        # Special Waypoints (NOT REMOVED)
        self.special_waypoints = [
            [0.324 , -4.094, 1.57],
            [0.324, -2.770 , 1.57],
            [0.324 , -1.404, 1.57],
            [0.324 , -0.046, 1.57],
            [-3.1940, -0.1185, -1.57],
            [-3.1940, -1.4044, -1.57],
            [-3.1940, -2.5360, -1.57],
            [-3.1940, -4.0941, -1.57]
        ]

        # Special PD gains (MODIFIED)
        self.kp_lin_sp = 0.6
        self.kd_lin_sp = 0.0 # NEW: Derivative gain for linear velocity
        self.kp_ang_sp = 1.5 # Adjusted P gain for better performance with D
        self.kd_ang_sp = 0.0 # NEW: Derivative gain for angular velocity
        self.dist_tol_sp = 0.15
        self.yaw_tol_sp =  0.15

        self.special_index = 0
        self.special_active = False # Controls which timer is active
        self.skip_next_special = False

        # State variables (NOT REMOVED)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        # PD State Variables (NEW)
        self.prev_dist_error = 0.0
        self.prev_ang_error = 0.0
        self.prev_yaw_error = 0.0
        self.last_time = self.get_clock().now()
        
        # Main P gains (NOT REMOVED)
        self.kp_lin = 0.6
        self.kp_ang = 1.5
        self.dist_tolerance = 0.15
        self.yaw_tolerance = 0.15

        # Flags for shape interrupt (NOT REMOVED)
        self.shape_triggered = False
        self.shape_type = None
        self.shape_published = False

        # --- NEW: State for stopping and waiting for shape detection ---
        self.wait_for_detection = False
        self.detection_wps = {2,3,4,5,10,11,12,13,15} # Waypoints where /collect_point is published and waiting starts 2,3,4,5,9,11,12

        # -------------------- MUTUALLY EXCLUSIVE TIMERS ------------------------
        # Initial state: Only main waypoints timer is active
        self.main_timer = self.create_timer(0.1, self.main_wp_callback)
        self.special_timer = None # Will be created when special_active is True
        self.wait_timer = None # NEW: For stopping and waiting

        self.get_logger().info("🚀 P Controller Started with Mutually Exclusive Timers")

    # -------------------- CALLBACKS ------------------------
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def shape_callback(self, msg: String):
        incoming = msg.data

        # --- CORE LOGIC START: Handling the wait state (from /collect_point) ---
        if self.wait_for_detection and incoming in ["BAD_HEALTH", "FERTILIZER_REQUIRED", "False"]:
            
            self.stop_robot()
            self.get_logger().info(f"✅ Received detection: {incoming}. Exiting wait state.")

            # Stop the waiting timer and clear the flag
            if self.wait_timer is not None:
                self.wait_timer.cancel()
                self.destroy_timer(self.wait_timer)
                self.wait_timer = None
                self.get_logger().info("🛑 Wait Timer Stopped.")
            
            self.wait_for_detection = False 

            # Now, check what was received and trigger the appropriate next step
            if incoming in ["BAD_HEALTH", "FERTILIZER_REQUIRED"]:
                self.get_logger().warn(f"⚠️ Shape Triggered: {incoming}")
                self.shape_triggered = True
                self.shape_type = incoming
                self.shape_published = False
                
                # SWITCHING LOGIC: Enable Special, Disable Main
                self.special_active = True      
                self.skip_next_special = False
                self.switch_to_special_mode()
            
            elif incoming == "False":
                self.get_logger().warn("⏭ Special waypoint skipped (False)")
                # SWITCHING LOGIC: Enable Special (to handle skip logic), Disable Main
                self.special_active = True
                self.skip_next_special = True   
                self.switch_to_special_mode()
            
            return

        # --- Original Interrupt Logic (Should only happen if not waiting at a collection point) ---
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


    # -------------------- SWITCHING LOGIC ------------------------
    def switch_to_special_mode(self):
        """Disables main timer and enables special timer."""
        # Reset PD error terms when switching modes (NEW)
        self.prev_dist_error = 0.0
        self.prev_ang_error = 0.0
        self.prev_yaw_error = 0.0
        self.last_time = self.get_clock().now()
        
        # Ensure main timer is stopped
        if self.main_timer is not None:
            self.main_timer.cancel()
            self.destroy_timer(self.main_timer)
            self.main_timer = None
            self.get_logger().info("🛑 Main WP Timer Stopped.")
        
        # Ensure wait timer is stopped (safety)
        if self.wait_timer is not None:
            self.wait_timer.cancel()
            self.destroy_timer(self.wait_timer)
            self.wait_timer = None
            self.wait_for_detection = False
            self.get_logger().info("🛑 Wait Timer Stopped.")

        # Start special timer
        if self.special_timer is None:
            self.special_timer = self.create_timer(0.1, self.special_wp_callback)
            self.get_logger().info("▶ Special WP Timer Started.")

    def switch_to_main_mode(self):
        """Disables special timer and enables main timer."""
        # Reset flags before switching back
        self.special_active = False
        self.skip_next_special = False
        self.shape_triggered = False
        self.shape_type = None
        self.shape_published = False
        self.stop_robot() # Stop before transitioning
        
        # Reset PD error terms when switching modes (NEW)
        self.prev_dist_error = 0.0
        self.prev_ang_error = 0.0
        self.prev_yaw_error = 0.0

        if self.special_timer is not None:
            self.special_timer.cancel()
            self.destroy_timer(self.special_timer)
            self.special_timer = None
            self.get_logger().info("🛑 Special WP Timer Stopped.")

        # Start main timer
        if self.main_timer is None:
            self.main_timer = self.create_timer(0.1, self.main_wp_callback)
            self.get_logger().info("▶ Main WP Timer Started.")
    
    # -------------------- NEW: WAIT TIMER CALLBACK ------------------------
    def wait_for_detection_callback(self):
        """Timer callback for the wait_for_detection state."""
        self.stop_robot()
        self.get_logger().info(f"⏳ Waiting for /shape_detected result at WP {self.current_wp_index} (Collection point)...")

    # -------------------- SPECIAL WAYPOINT CALLBACK (MODIFIED: PD Controller) ------------------------
    def special_wp_callback(self):
        """Handles special waypoints using a PD controller."""
        if not self.special_active:
            self.switch_to_main_mode() 
            return

        if self.skip_next_special:
            self.get_logger().info(f"⏭ Skipped special WP {self.special_index+1}")
            self.stop_robot()
            self.special_index += 1 
            # 🔴 SWITCH BACK TO MAIN MODE
            self.switch_to_main_mode()
            return

        # Check if special index is valid (optional, depends on logic)
        if self.special_index >= len(self.special_waypoints):
            self.get_logger().warn("⚠️ All special waypoints exhausted. Switching to Main.")
            self.switch_to_main_mode()
            return

        # Time management for Derivative component (NEW)
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # Target and Error Calculation
        tx, ty, tyaw = self.special_waypoints[self.special_index]
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        ang = math.atan2(dy, dx)
        ang_err = self.normalize_angle(ang - self.yaw)

        cmd = Twist()
        
        # PD components for distance and angular error (NEW)
        dist_deriv = (dist - self.prev_dist_error) / dt if dt > 0 else 0
        ang_deriv = (ang_err - self.prev_ang_error) / dt if dt > 0 else 0
        self.prev_dist_error = dist
        self.prev_ang_error = ang_err

        # 1. Linear Movement (PD Control)
        if dist > self.dist_tol_sp:
            # Linear PD
            cmd.linear.x = self.kp_lin_sp * dist + self.kd_lin_sp * dist_deriv
            # Angular PD
            cmd.angular.z = self.kp_ang_sp * ang_err + self.kd_ang_sp * ang_deriv
            
            self.get_logger().info(f"🚗 Moving (PD) to special WP {self.special_index}, distance: {dist:.3f}, angle error: {ang_err:.3f}")

        # 2. Final Yaw Adjustment (PD Control)
        else:
            yaw_err = self.normalize_angle(tyaw - self.yaw)
            yaw_deriv = (yaw_err - self.prev_yaw_error) / dt if dt > 0 else 0
            self.prev_yaw_error = yaw_err
            
            if abs(yaw_err) > self.yaw_tol_sp:
                # Yaw PD
                cmd.angular.z = self.kp_ang_sp * yaw_err + self.kd_ang_sp * yaw_deriv
                self.get_logger().info(f"↪ Adjusting yaw (PD) at special WP {self.special_index+1}, yaw error: {yaw_err:.3f}")
            
            # 3. Reached Target
            else:
                self.stop_robot()
                self.get_logger().info(f"✅ Reached special WP {self.special_index}")

                self.special_index += 1 
                
                # 2s stop (now SAFE) (NOTE: Using blocking sleep in a timer is bad practice 
                # for real ROS, but keeping it as in your original logic)
                stop_time = time.time()
                while time.time() - stop_time < 2.0:
                    self.stop_robot()
                    rclpy.spin_once(self, timeout_sec=0.01)

                # Publish detection
                if not self.shape_published and self.shape_type:
                    msg = String()
                    waypoint_id = self.special_index 
                    if waypoint_id == 5:
                        waypoint_id = 8
                    elif waypoint_id == 6:
                        waypoint_id = 7
                    elif waypoint_id == 7:
                        waypoint_id = 6
                    elif waypoint_id == 8:
                        waypoint_id = 5
                    msg.data = f"{self.shape_type},{self.x:.3f},{self.y:.3f},{waypoint_id}"
                    self.det_pub.publish(msg)
                    self.get_logger().info(f"✅ Published{msg.data} to /detection_status")
                    self.shape_published = True

                # 🔴 SWITCH BACK TO MAIN MODE
                self.switch_to_main_mode()
                return # Stop processing this loop iteration


        # Clamp velocities
        cmd.linear.x = max(min(cmd.linear.x, 0.2), -0.2)
        cmd.angular.z = max(min(cmd.angular.z, 0.8), -0.8)
        self.cmd_pub.publish(cmd)


    # -------------------- MAIN WAYPOINT CALLBACK ------------------------
    def main_wp_callback(self):
        """Handles main waypoints (Logic from control_loop)."""

        if self.special_active or self.wait_for_detection: # Safety check
            return
            
        if self.current_wp_index >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info("🎉 All main waypoints completed.")
            return

        target_x, target_y, target_yaw = self.waypoints[self.current_wp_index]
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        path_angle = math.atan2(dy, dx)
        ang_error = self.normalize_angle(path_angle - self.yaw)

        cmd = Twist()
        
        # 1. Linear Movement
        if distance > self.dist_tolerance:
            cmd.linear.x = self.kp_lin * distance
            cmd.angular.z = self.kp_ang * ang_error
            self.get_logger().info(f"🚗 Moving to main WP {self.current_wp_index}, distance: {distance:.3f}, angle error: {ang_error:.3f}")
        
        # 2. Final Yaw Adjustment
        else:
            yaw_error = self.normalize_angle(target_yaw - self.yaw)
            if abs(yaw_error) > self.yaw_tolerance:
                cmd.angular.z = self.kp_ang * yaw_error
                self.get_logger().info(f"↪ Adjusting yaw at main WP {self.current_wp_index+1}, yaw error: {yaw_error:.3f}")
            
            # 3. Reached Target
            else:
                self.stop_robot()
                self.get_logger().info(f"✅ Reached main WP {self.current_wp_index}")

                # Dock station example (NOT REMOVED)
                if self.current_wp_index == 4:
                    dock_msg = String()
                    dock_msg.data = f"DOCK_STATION,{self.x:.3f},{self.y:.3f},0"
                    self.det_pub.publish(dock_msg)
                    self.get_logger().info(f"📤 DOCK_STATION Published: {dock_msg.data}")

                     # Stop for specific time at dock station
                    stop_time = time.time()
                    dock_wait_duration = 25.0                                                                   # ⬅️ YE PARAMETER CHANGE KARO (seconds mein)
                    while time.time() - stop_time < dock_wait_duration:
                        self.stop_robot()
                        rclpy.spin_once(self, timeout_sec=0.01)
                    self.get_logger().info(f"⏱️ Waited {dock_wait_duration}s at DOCK_STATION")

                # Collection points (MODIFIED LOGIC)
                if self.current_wp_index in self.detection_wps:
                    cp_msg = String()
                    cp_msg.data = "true"
                    self.collect_pub.publish(cp_msg)
                    self.get_logger().info("📡 /collect_point → true. Waiting for /shape_detected...")
                    
                    # --- NEW: Switch to Wait Mode ---
                    self.wait_for_detection = True
                    
                    if self.main_timer is not None:
                        self.main_timer.cancel()
                        self.destroy_timer(self.main_timer)
                        self.main_timer = None
                        self.get_logger().info("🛑 Main WP Timer Stopped.")
                    
                    self.wait_timer = self.create_timer(0.5, self.wait_for_detection_callback)
                    self.get_logger().info("▶ Wait Timer Started (Replacing 7.5s sleep).")
                    
                    self.current_wp_index += 1 # Increment main WP index immediately
                    return # Exit the main_wp_callback to enter the waiting state
                
                # Original logic for non-collection points
                self.current_wp_index += 1
                self.get_logger().info(f"➡ Next main WP index {self.current_wp_index}")
                return

        # Clamp velocities
        cmd.linear.x = max(min(cmd.linear.x, 0.3), -0.3)
        cmd.angular.z = max(min(cmd.angular.z, 1.0), -1.0)
        self.cmd_pub.publish(cmd)

    # -------------------- UTILITIES ------------------------
    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    @staticmethod
    def normalize_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = PController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()