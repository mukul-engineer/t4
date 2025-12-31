#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from control_msgs.msg import JointJog


class JointJogTimed(Node):

    def __init__(self):
        super().__init__('joint_jog_timed')

        self.publisher = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        # ===== PARAMETERS =====
        self.duration_sec = 3.0      # publish for 3 seconds
        self.publish_rate = 10.0     # Hz
        # ======================

        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_cb)

        self.get_logger().info('JointJog timed publisher started')

    def timer_cb(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if elapsed >= self.duration_sec:
            self.get_logger().info('JointJog duration complete. Stopping.')
            rclpy.shutdown()
            return

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        msg.velocities = [
            0.0,
            0.0,
            0.1,   # elbow joint moving
            0.0,
            0.0,
            0.0
        ]

        msg.duration = 0.1
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointJogTimed()
    rclpy.spin(node)


if __name__ == '__main__':
    main()

