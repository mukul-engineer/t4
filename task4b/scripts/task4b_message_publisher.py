#!/usr/bin/env python3
"""
Task 4B - Message Publisher Node
- Publishes to /detection_status in required format
- Format: Status,x,y,plant_ID
- Handles FERTILIZER_REQUIRED, BAD_HEALTH, DOCK_STATION
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Task4BMessagePublisher(Node):
    def __init__(self):
        super().__init__('task4b_message_publisher')
        
        # Subscribe to detection requests from navigation
        self.create_subscription(String, '/detection_request', self.request_callback, 10)
        
        # Publisher for official detection status
        self.detection_pub = self.create_publisher(String, '/detection_status', 10)
        
        # Track published detections
        self.published_detections = []
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("📤 Task 4B Message Publisher Started")
        self.get_logger().info("   Publishing to /detection_status")
        self.get_logger().info("   Format: Status,x,y,plant_ID")
        self.get_logger().info("=" * 50)

    def request_callback(self, msg):
        """
        Receive detection request from navigation node
        Format: Status,x,y,plant_ID
        """
        data = msg.data.strip()
        
        if not data:
            return
        
        parts = data.split(',')
        if len(parts) != 4:
            self.get_logger().warn(f"⚠️ Invalid format: {data}")
            return
        
        status = parts[0]
        x = parts[1]
        y = parts[2]
        plant_id = parts[3]
        
        # Validate status
        valid_statuses = ["FERTILIZER_REQUIRED", "BAD_HEALTH", "DOCK_STATION"]
        if status not in valid_statuses:
            self.get_logger().warn(f"⚠️ Invalid status: {status}")
            return
        
        # Create and publish message
        out_msg = String()
        out_msg.data = f"{status},{x},{y},{plant_id}"
        self.detection_pub.publish(out_msg)
        
        # Log
        self.published_detections.append(out_msg.data)
        self.get_logger().info(f"✅ Published: {out_msg.data}")
        
        # Summary
        self.print_summary()

    def print_summary(self):
        """Print detection summary"""
        self.get_logger().info("-" * 40)
        self.get_logger().info(f"📊 Total detections: {len(self.published_detections)}")
        
        fert_count = sum(1 for d in self.published_detections if "FERTILIZER_REQUIRED" in d)
        bad_count = sum(1 for d in self.published_detections if "BAD_HEALTH" in d)
        dock_count = sum(1 for d in self.published_detections if "DOCK_STATION" in d)
        
        self.get_logger().info(f"   FERTILIZER_REQUIRED: {fert_count}")
        self.get_logger().info(f"   BAD_HEALTH: {bad_count}")
        self.get_logger().info(f"   DOCK_STATION: {dock_count}")
        self.get_logger().info("-" * 40)


def main(args=None):
    rclpy.init(args=args)
    node = Task4BMessagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
