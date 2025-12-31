#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class TCPPosePrinter(Node):

    def __init__(self):
        super().__init__('tcp_pose_printer')

        # Latest TCP pose storage
        self.latest_pose = None

        # Subscriber
        self.tcp_pose_sub = self.create_subscription(
            Float64MultiArray,
            '/tcp_pose_raw',
            self.tcp_pose_callback,
            10
        )

        # Timer: print once every second
        self.timer = self.create_timer(1.0, self.print_pose)

        self.get_logger().info('TCP Pose printer started (1 Hz)')

    def tcp_pose_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 6:
            self.get_logger().warn('Received TCP pose with insufficient data')
            return

        # Store latest pose
        self.latest_pose = msg.data

    def print_pose(self):
        if self.latest_pose is None:
            self.get_logger().info('Waiting for /tcp_pose_raw data...')
            return

        x, y, z, rx, ry, rz = self.latest_pose

        self.get_logger().info(
            f'TCP Pose | '
            f'Position [x={x:.4f}, y={y:.4f}, z={z:.4f}] | '
            f'Orientation [rx={rx:.4f}, ry={ry:.4f}, rz={rz:.4f}]'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TCPPosePrinter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

