#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class MagnetClient(Node):

    def __init__(self):
        super().__init__('magnet_client')

        # Service client
        self.magnet_client = self.create_client(SetBool, '/magnet')

        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Magnet service not available, waiting...')

        # Call service with TRUE once
        self.call_magnet(True)

    def call_magnet(self, state: bool):
        request = SetBool.Request()
        request.data = state  # True = ON, False = OFF

        future = self.magnet_client.call_async(request)
        future.add_done_callback(self.magnet_response_callback)

    def magnet_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Magnet activated successfully: {response.message}')
            else:
                self.get_logger().warn(f'Magnet activation failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

        # Shutdown after single call
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MagnetClient()
    rclpy.spin(node)


if __name__ == '__main__':
    main()

