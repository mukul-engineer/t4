#!/usr/bin/env python3
"""
Shape Detection Test for Remote Hardware (NO GUI)
- Tests shape detection without cv2.imshow
- Publishes results to ROS topics for monitoring
- Use ros2 topic echo to verify detection
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import numpy as np
import random
import time


class ShapeDetectionTest(Node):
    def __init__(self):
        super().__init__('shape_detection_test')
        
        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(String, '/collect_point', self.collect_callback, 10)
        
        # Publishers for testing
        self.pub_detect = self.create_publisher(String, '/shape_detected', 10)
        self.pub_debug = self.create_publisher(String, '/shape_debug', 10)
        self.pub_lidar_info = self.create_publisher(String, '/lidar_info', 10)
        
        # Collection state
        self.collecting = False
        self.delay_timer = None
        self.collection_delay = 2.0
        self.collect_window = 5.0
        self.collect_start_time = 0.0
        self.collected_shapes = []
        
        # RANSAC parameters (tuned for real hardware)
        self.prev_ranges = None
        self.iterations = 30
        self.ransac_threshold = 0.03  # 3cm threshold
        self.min_inliers = 8
        self.max_inliers = 50
        self.min_line_length = 0.04   # 4cm
        self.max_line_length = 0.4    # 40cm
        self.max_range = 2.0          # Detect within 2m
        
        # Line memory
        self.line_memory = []
        self.memory_duration = 1.5
        
        # Debug counters
        self.scan_count = 0
        self.detection_count = 0
        
        # Status timer
        self.status_timer = self.create_timer(2.0, self.publish_status)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🔍 Shape Detection Test Started (HEADLESS MODE)")
        self.get_logger().info("   Monitor topics:")
        self.get_logger().info("   - ros2 topic echo /shape_detected")
        self.get_logger().info("   - ros2 topic echo /shape_debug")
        self.get_logger().info("   - ros2 topic echo /lidar_info")
        self.get_logger().info("   Trigger detection:")
        self.get_logger().info("   - ros2 topic pub /collect_point std_msgs/String 'data: true'")
        self.get_logger().info("=" * 60)

    def collect_callback(self, msg):
        """Triggered manually for testing"""
        if msg.data.strip().lower() == "true":
            if self.collecting or self.delay_timer is not None:
                return
            
            self.get_logger().info(f"⏳ Detection triggered, waiting {self.collection_delay}s...")
            self.publish_debug("DETECTION_TRIGGERED")
            self.delay_timer = self.create_timer(self.collection_delay, self.start_collection)

    def start_collection(self):
        if self.delay_timer:
            self.delay_timer.cancel()
            self.destroy_timer(self.delay_timer)
            self.delay_timer = None
        
        self.get_logger().info(f"📍 Collecting for {self.collect_window}s...")
        self.publish_debug(f"COLLECTION_STARTED_FOR_{self.collect_window}s")
        self.collecting = True
        self.collect_start_time = time.time()
        self.collected_shapes = []

    def scan_callback(self, msg):
        self.scan_count += 1
        
        # Publish LiDAR info every 50 scans
        if self.scan_count % 50 == 0:
            valid_ranges = np.array(msg.ranges)
            valid_ranges = valid_ranges[(valid_ranges > msg.range_min) & 
                                      (valid_ranges < msg.range_max) & 
                                      (valid_ranges < self.max_range)]
            
            info = f"SCAN_{self.scan_count}:points={len(valid_ranges)},min={np.min(valid_ranges):.2f},max={np.max(valid_ranges):.2f}"
            self.publish_lidar_info(info)
        
        # Convert to cartesian
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        
        valid = (ranges > msg.range_min) & (ranges < msg.range_max) & (ranges < self.max_range)
        ranges, angles = ranges[valid], angles[valid]
        
        if len(ranges) < 20:
            if self.collecting:
                self.publish_debug(f"INSUFFICIENT_POINTS:{len(ranges)}")
            return
        
        ranges = self.smooth_ranges(ranges)
        
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x, y)).T
        
        # Detect lines
        lines = self.detect_lines(points)
        
        # Update memory
        t_now = time.time()
        for line in lines:
            self.line_memory.append((t_now, line))
        self.line_memory = [(t, l) for (t, l) in self.line_memory if t_now - t < self.memory_duration]
        active_lines = [l for (_, l) in self.line_memory]
        
        # Detect shapes
        triangles = self.detect_triangle(active_lines)
        squares = self.detect_square(active_lines)
        
        # Debug info during collection
        if self.collecting:
            debug_msg = f"LINES:{len(active_lines)},TRI:{len(triangles)},SQ:{len(squares)}"
            self.publish_debug(debug_msg)
        
        # Collect during window
        if self.collecting:
            for _ in triangles:
                self.collected_shapes.append("triangle")
            for _ in squares:
                self.collected_shapes.append("square")
            
            if time.time() - self.collect_start_time >= self.collect_window:
                self.finish_collection()

    def smooth_ranges(self, ranges):
        # Median filter
        ksize = 5
        pad = ksize // 2
        padded = np.pad(ranges, (pad, pad), mode='edge')
        filtered = np.array([np.median(padded[i:i+ksize]) for i in range(len(ranges))])
        
        # Temporal smoothing
        if self.prev_ranges is None or len(self.prev_ranges) != len(filtered):
            self.prev_ranges = filtered.copy()
            return filtered
        
        smoothed = 0.6 * filtered + 0.4 * self.prev_ranges
        self.prev_ranges = smoothed.copy()
        return smoothed

    def detect_lines(self, points):
        """RANSAC line detection"""
        if len(points) < 10:
            return []
        
        pts_list = [tuple(p) for p in points.tolist()]
        lines = []
        
        for _ in range(self.iterations):
            if len(pts_list) < 2:
                break
            
            p1, p2 = random.sample(pts_list, 2)
            
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = p2[0] * p1[1] - p1[0] * p2[1]
            norm = np.sqrt(A*A + B*B)
            if norm < 1e-6:
                continue
            
            inliers = []
            for p in pts_list:
                dist = abs(A * p[0] + B * p[1] + C) / norm
                if dist < self.ransac_threshold:
                    inliers.append(p)
            
            if self.min_inliers <= len(inliers) <= self.max_inliers:
                inliers_np = np.array(inliers)
                x_min, y_min = np.min(inliers_np, axis=0)
                x_max, y_max = np.max(inliers_np, axis=0)
                
                length = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
                if self.min_line_length < length < self.max_line_length:
                    angle = np.arctan2(y_max - y_min, x_max - x_min)
                    
                    # Check duplicate
                    is_dup = False
                    for ex in lines:
                        angle_diff = abs(angle - ex['angle'])
                        angle_diff = min(angle_diff, np.pi - angle_diff)
                        mid_new = np.array([(x_min + x_max)/2, (y_min + y_max)/2])
                        mid_old = np.array([(ex['x1'] + ex['x2'])/2, (ex['y1'] + ex['y2'])/2])
                        if angle_diff < 0.3 and np.linalg.norm(mid_new - mid_old) < 0.1:
                            is_dup = True
                            break
                    
                    if not is_dup:
                        lines.append({
                            'x1': x_min, 'y1': y_min,
                            'x2': x_max, 'y2': y_max,
                            'angle': angle, 'length': length
                        })
                        pts_list = [p for p in pts_list if p not in inliers]
        
        return lines

    def detect_triangle(self, lines):
        """Triangle: 2 lines at ~50-70 degrees"""
        if len(lines) < 2:
            return []
        
        results = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                angle_diff = abs(lines[i]['angle'] - lines[j]['angle'])
                angle_diff = min(angle_diff, np.pi - angle_diff)
                angle_deg = np.degrees(angle_diff)
                
                if 40 <= angle_deg <= 80:
                    if self.lines_close(lines[i], lines[j]):
                        results.append(True)
        
        return results

    def detect_square(self, lines):
        """Square: 2 lines at ~90 degrees"""
        if len(lines) < 2:
            return []
        
        results = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                angle_diff = abs(lines[i]['angle'] - lines[j]['angle'])
                angle_diff = min(angle_diff, np.pi - angle_diff)
                angle_deg = np.degrees(angle_diff)
                
                if 70 <= angle_deg <= 110:
                    if self.lines_close(lines[i], lines[j]):
                        results.append(True)
        
        return results

    def lines_close(self, l1, l2):
        """Check if lines are close enough"""
        pts = [(l1['x1'], l1['y1']), (l1['x2'], l1['y2']),
               (l2['x1'], l2['y1']), (l2['x2'], l2['y2'])]
        
        min_dist = float('inf')
        for i in range(2):
            for j in range(2, 4):
                d = np.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
                min_dist = min(min_dist, d)
        
        return min_dist < 0.2

    def finish_collection(self):
        """Publish final detection result"""
        self.collecting = False
        self.detection_count += 1
        
        if not self.collected_shapes:
            msg = String()
            msg.data = "False"
            self.pub_detect.publish(msg)
            self.publish_debug(f"DETECTION_{self.detection_count}:NO_SHAPE")
            self.get_logger().info("⚠️ No shape detected")
            return
        
        from collections import Counter
        counts = Counter(self.collected_shapes)
        self.publish_debug(f"DETECTION_{self.detection_count}:COUNTS_{dict(counts)}")
        
        most_common = counts.most_common(1)[0]
        shape, count = most_common
        
        if count >= 2:
            result = "FERTILIZER_REQUIRED" if shape == "triangle" else "BAD_HEALTH"
            msg = String()
            msg.data = result
            self.pub_detect.publish(msg)
            self.publish_debug(f"DETECTION_{self.detection_count}:RESULT_{result}")
            self.get_logger().info(f"✅ Detected: {shape} → {result}")
        else:
            msg = String()
            msg.data = "False"
            self.pub_detect.publish(msg)
            self.publish_debug(f"DETECTION_{self.detection_count}:LOW_CONFIDENCE_{count}")
            self.get_logger().info(f"⚠️ Low confidence ({count})")
        
        self.collected_shapes = []

    def publish_debug(self, message):
        """Publish debug info"""
        msg = String()
        msg.data = message
        self.pub_debug.publish(msg)

    def publish_lidar_info(self, message):
        """Publish LiDAR info"""
        msg = String()
        msg.data = message
        self.pub_lidar_info.publish(msg)

    def publish_status(self):
        """Publish periodic status"""
        status = f"STATUS:scans={self.scan_count},detections={self.detection_count},collecting={self.collecting}"
        self.publish_debug(status)


def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetectionTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()