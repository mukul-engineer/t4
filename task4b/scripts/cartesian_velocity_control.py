#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


class CartesianTwistTimed(Node):

    def __init__(self):
        super().__init__('cartesian_twist_timed')

        self.publisher = self.create_publisher(
            TwistStamped,
            '/delta_twist_cmds',
            10
        )

        # ===== PARAMETERS =====
        self.duration_sec = 3.0      # publish for 3 seconds
        self.publish_rate = 10.0     # Hz
        # ======================

        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_cb)

        self.get_logger().info('Cartesian Twist timed publisher started')

    def timer_cb(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if elapsed >= self.duration_sec:
            self.get_logger().info('Cartesian motion complete. Stopping.')
            rclpy.shutdown()
            return

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Linear velocity
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.1   # move up

        # Angular velocity
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CartesianTwistTimed()
    rclpy.spin(node)


if __name__ == '__main__':
    main()

