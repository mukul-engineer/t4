#!/usr/bin/env python3
"""
Simple Single Waypoint Test for Remote Hardware
- P-controller navigation to one waypoint
- Uses hardware topics: /odom, /orientation, /ultrasonic_sensor_std_float
- NO Nav2, NO teleop
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray
import math


class SimpleWaypointTest(Node):
    def __init__(self):
        super().__init__('simple_waypoint_test')
        
        # ==================== PUBLISHERS ====================
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # ==================== SUBSCRIBERS ====================
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float32, '/orientation', self.orientation_callback, 10)
        self.create_subscription(Float32MultiArray, '/ultrasonic_sensor_std_float', self.ultra_callback, 10)
        
        # ==================== ROBOT STATE ====================
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.imu_yaw = 0.0
        self.ultra_left = 0.0
        self.ultra_right = 0.0
        
        # ==================== NAVIGATION PARAMETERS ====================
        self.kp_lin = 0.4
        self.kp_ang = 1.0
        self.dist_tolerance = 0.15
        self.yaw_tolerance = 0.1
        self.max_lin_vel = 0.2
        self.max_ang_vel = 0.6
        
        # ==================== TARGET WAYPOINT ====================
        # Change these coordinates to test different positions
        self.target_x = 1.0
        self.target_y = 0.0
        self.target_yaw = 0.0
        
        self.reached = False
        
        # ==================== TIMER ====================
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("🚀 Simple Waypoint Test Started")
        self.get_logger().info(f"   Target: ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_yaw:.2f})")
        self.get_logger().info("   Hardware topics: /odom, /orientation, /ultrasonic_sensor_std_float")
        self.get_logger().info("=" * 50)

    # ==================== CALLBACKS ====================
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # Convert quaternion to yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def orientation_callback(self, msg):
        """Hardware IMU orientation (0-6.28 radians)"""
        self.imu_yaw = msg.data
        if self.imu_yaw > math.pi:
            self.imu_yaw -= 2 * math.pi

    def ultra_callback(self, msg):
        """Ultrasonic sensors from hardware"""
        if len(msg.data) > 5:
            self.ultra_left = msg.data[4]
            self.ultra_right = msg.data[5]

    # ==================== CONTROL LOOP ====================
    def control_loop(self):
        if self.reached:
            self.stop_robot()
            return
        
        # Calculate distance and angle to target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        cmd = Twist()
        
        # Phase 1: Move to position
        if distance > self.dist_tolerance:
            path_angle = math.atan2(dy, dx)
            ang_error = self.normalize_angle(path_angle - self.yaw)
            
            # P-controller
            cmd.linear.x = self.kp_lin * distance
            cmd.angular.z = self.kp_ang * ang_error
            
            # Clamp velocities
            cmd.linear.x = max(min(cmd.linear.x, self.max_lin_vel), -self.max_lin_vel)
            cmd.angular.z = max(min(cmd.angular.z, self.max_ang_vel), -self.max_ang_vel)
            
            # Slow down during sharp turns
            if abs(ang_error) > 0.5:
                cmd.linear.x *= 0.4
            
            # Safety: Only use ultrasonics when moving backward
            # (ultrasonics are at the BACK of the robot)
            if cmd.linear.x < 0 and (self.ultra_left < 0.3 or self.ultra_right < 0.3):
                cmd.linear.x = 0.0
                self.get_logger().warn(f"⚠️ Rear obstacle detected! Left: {self.ultra_left:.2f}m, Right: {self.ultra_right:.2f}m")
            
            self.cmd_pub.publish(cmd)
            
            # Log progress every 2 seconds
            if int(self.get_clock().now().nanoseconds / 1e9) % 2 == 0:
                self.get_logger().info(f"🚗 Moving to target: dist={distance:.2f}m, ang_err={math.degrees(ang_error):.1f}°")
            
            return
        
        # Phase 2: Adjust final yaw
        yaw_error = self.normalize_angle(self.target_yaw - self.yaw)
        if abs(yaw_error) > self.yaw_tolerance:
            cmd.angular.z = self.kp_ang * yaw_error
            cmd.angular.z = max(min(cmd.angular.z, self.max_ang_vel), -self.max_ang_vel)
            self.cmd_pub.publish(cmd)
            
            self.get_logger().info(f"↪ Adjusting yaw: error={math.degrees(yaw_error):.1f}°")
            return
        
        # Phase 3: Reached target
        self.stop_robot()
        self.reached = True
        self.get_logger().info("=" * 50)
        self.get_logger().info("✅ TARGET REACHED!")
        self.get_logger().info(f"   Final position: ({self.x:.3f}, {self.y:.3f})")
        self.get_logger().info(f"   Final yaw: {math.degrees(self.yaw):.1f}°")
        self.get_logger().info(f"   IMU yaw: {math.degrees(self.imu_yaw):.1f}°")
        self.get_logger().info("=" * 50)

    def stop_robot(self):
        """Stop the robot"""
        self.cmd_pub.publish(Twist())

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def set_target(self, x, y, yaw=0.0):
        """Change target waypoint (call this to test different positions)"""
        self.target_x = x
        self.target_y = y
        self.target_yaw = yaw
        self.reached = False
        self.get_logger().info(f"🎯 New target set: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}°)")


def main(args=None):
    rclpy.init(args=args)
    node = SimpleWaypointTest()
    
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