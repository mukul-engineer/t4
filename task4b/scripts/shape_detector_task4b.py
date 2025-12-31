#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import cv2
import random
import time


class LineRansacNode(Node):
    def __init__(self):
        super().__init__('lidar_line_ransac')
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # --------- ADDED ODOM SUBSCRIBER ----------
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)


        # --------- ADDED PUBLISHER ----------
        self.pub_detect = self.create_publisher(String, '/shape_detected', 10)
        self.sub_collect = self.create_subscription(
            String,
            '/collect_point',
            self.collect_callback,
            10
        )
        # ------------------------------------

        # --------- MODIFIED: collection state ----------
        self.collecting = False
        self.delay_timer = None                       # NEW: Timer for non-blocking delay
        self.collection_delay_seconds = 3.0           # NEW: 2 seconds delay
        self.collect_start_time = 0.0                 # Initialized correctly
        
        # NOTE: self.collected_centers now stores (timestamp, shape_type_str, (cx, cy))
        self.collected_centers = []
        self.collect_window = 7.0                                                            # seconds to collect
        # -------------------------------------------

        self.get_logger().info("🚀 LiDAR Line Detection (RANSAC + Shape Recognition) started")

        # --- RANSAC + memory params ---
        self.prev_ranges = None
        self.iterations = 30
        self.threshold = 1.7
        self.min_inliers = 20
        self.max_inliers = 100
        self.line_memory = []
        self.memory_duration = 2.0

        # --- Visualization parameters ---
        self.declare_parameter('lidar_point_size', 3)
        self.declare_parameter('display_scale', 200)
        self.declare_parameter('max_line_length', 150)
        self.declare_parameter('duplicate_angle_thresh', 0.65)
        self.declare_parameter('duplicate_dist_thresh', 80)

        self.lidar_point_size = self.get_parameter('lidar_point_size').get_parameter_value().integer_value
        self.display_scale = self.get_parameter('display_scale').get_parameter_value().integer_value
        self.max_line_length = self.get_parameter('max_line_length').get_parameter_value().integer_value
        self.duplicate_angle_thresh = self.get_parameter('duplicate_angle_thresh').get_parameter_value().double_value
        self.duplicate_dist_thresh = self.get_parameter('duplicate_dist_thresh').get_parameter_value().double_value

    # ---------------------------------------------------------------------
    #                  ADDED: ODOM CALLBACK
    # ---------------------------------------------------------------------
    def odom_callback(self, msg):
        self.robot_linear_vel = msg.twist.twist.linear.x
        self.robot_angular_vel = msg.twist.twist.angular.z

        # NEW: store odom position
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    # ---------------------------------------------------------------------
    
    # ---------------------------------------------------------------------
    #                  MODIFIED: COLLECT CALLBACK (Starts the delay)
    # ---------------------------------------------------------------------
    def collect_callback(self, msg: String):
        """Starts a delay timer when external trigger arrives."""
        if msg.data.strip().lower() == "true":
            # If collection or delay already running → ignore
            if self.collecting or self.delay_timer is not None:
                self.get_logger().warn("📍 Collection already active or delayed. Ignoring new trigger.")
                return
            
            self.get_logger().info(f"⏳ Collection START triggered. Waiting for {self.collection_delay_seconds} seconds delay...")
            
            # Start a one-shot timer for the delay
            self.delay_timer = self.create_timer(self.collection_delay_seconds, self.start_actual_collection)

    # ---------------------------------------------------------------------
    #                  NEW: START ACTUAL COLLECTION (Called after delay)
    # ---------------------------------------------------------------------
    def start_actual_collection(self):
        """Starts the 7.0 second collection window after the delay."""
        
        # Stop and destroy the delay timer
        if self.delay_timer:
            self.delay_timer.cancel()
            self.destroy_timer(self.delay_timer)
            self.delay_timer = None
        
        # Start the collection process
        self.get_logger().info(f"📍 Delay finished. Starting {self.collect_window}s collection window now.")
        self.collecting = True
        self.collect_start_time = time.time()
        self.collected_centers = []


    # ---------------------------------
    def smooth_ranges(self, current_ranges):
        if self.prev_ranges is None or len(self.prev_ranges) != len(current_ranges):
            self.prev_ranges = current_ranges.copy()
            return current_ranges

        noise = np.std(current_ranges - self.prev_ranges)
        alpha = np.clip(0.6 - noise * 5, 0.1, 0.6)
        smoothed = alpha * current_ranges + (1 - alpha) * self.prev_ranges
        self.prev_ranges = smoothed.copy()
        return smoothed

    # ---------------------------------
    def scan_callback(self, msg):
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        ranges, angles = ranges[valid], angles[valid]

        def median_filter(data, ksize=5):
            pad = ksize // 2
            padded = np.pad(data, (pad, pad), mode='edge')
            return np.array([np.median(padded[i:i + ksize]) for i in range(len(data))])

        ranges = median_filter(ranges, 5)
        ranges = self.smooth_ranges(ranges)

        # --- Convert to Cartesian coordinates ---
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.vstack((x, y)).T

        if len(points) < 20:
            return

        # --- Visualization canvas ---
        scale = self.display_scale
        canvas_size = 800
        origin = np.array([canvas_size // 2, canvas_size // 2])
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

        # Correct Y-axis flip for OpenCV display
        pts_img = np.column_stack((
            points[:, 0] * scale + origin[0],
            origin[1] - points[:, 1] * scale
        )).astype(np.int32)

        # Draw LiDAR points
        for p in pts_img:
            cv2.circle(canvas, tuple(p), self.lidar_point_size, (255, 255, 255), -1)

        # --- Detect lines via RANSAC ---
        new_lines = self.segment_lines_ransac(pts_img)
        t_now = time.time()
        for line in new_lines:
            self.line_memory.append((t_now, line))
        self.line_memory = [(t, l) for (t, l) in self.line_memory if t_now - t < self.memory_duration]
        active_lines = [l for (_, l) in self.line_memory]

        # Clamp long lines
        clamped_lines = []
        for (x1, y1, x2, y2, inliers) in active_lines:
            length = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
            if length <= self.max_line_length:
                clamped_lines.append((x1, y1, x2, y2, inliers))

        # Draw remaining lines
        for (x1, y1, x2, y2, inliers) in clamped_lines:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 5)
            cv2.putText(canvas, f"{inliers}", (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # --- Detect square shape ---
        squares = self.detect_square_shape(clamped_lines)
        squares = self.dedupe_centers(squares, min_dist=50)  # Deduplicate nearby detections
        for (cx, cy) in squares:
            cv2.putText(canvas, "Square", (int(cx) - 30, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(canvas, (int(cx), int(cy)), 6, (0, 255, 0), -1)

        # --- Detect triangle shape ---
        triangles = self.detect_triangle_shape(clamped_lines, triangle_angle=50, tol=10)
        triangles = self.dedupe_centers(triangles, min_dist=50)  # Deduplicate nearby detections
        for (cx, cy) in triangles:
            cv2.putText(canvas, "Triangle", (int(cx)-30, int(cy)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.circle(canvas, (int(cx), int(cy)), 6, (0,0,255), -1)

        # -------------------------------
        #   COLLECTION WINDOW (MODIFIED LOGIC)
        # -------------------------------
        if self.collecting:

            current_time = time.time() # Capture time for this scan frame

            # collect triangles & squares
            for c in triangles:
                # Store (timestamp, shape_type, center)
                self.collected_centers.append((current_time, "triangle", (float(c[0]), float(c[1]))))
            for c in squares:
                # Store (timestamp, shape_type, center)
                self.collected_centers.append((current_time, "square", (float(c[0]), float(c[1]))))

            # check if window finished
            if current_time - self.collect_start_time >= self.collect_window:

                if len(self.collected_centers) > 0:
                    
                    # ----------------------------------------------------
                    # NEW LOGIC: Find the LAST detected shape (by timestamp)
                    # ----------------------------------------------------
                    
                    # Extract all timestamps from collected_centers
                    timestamps = [item[0] for item in self.collected_centers]
                    
                    # Find the index of the maximum (latest) timestamp
                    latest_idx = int(np.argmax(timestamps))
                    
                    # Retrieve the latest detected shape entry
                    _, stype, (cx_img, cy_img) = self.collected_centers[latest_idx]

                    detected = "FERTILIZER_REQUIRED" if stype == "triangle" else "BAD_HEALTH"

                    # Prepare message
                    msg = String()
                    msg.data = detected

                    self.pub_detect.publish(msg)
                    self.get_logger().info(f"📤 Shape Published (LAST DETECTED): {msg.data} (Type: {stype}, Image Center: {cx_img:.1f}, {cy_img:.1f})")


                else:
                    msg = String()
                    msg.data = "False"
                    self.pub_detect.publish(msg)
                    self.get_logger().info("⚠️ No shape detected — Published False")


                # reset
                self.collecting = False
                self.collected_centers = []

        # -------------------------------

        cv2.circle(canvas, tuple(origin), 5, (255, 0, 0), -1)
        cv2.putText(canvas, "Robot", (origin[0] - 20, origin[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("LiDAR RANSAC + Shape Detection", canvas)
        cv2.waitKey(1)

    # ---------------------------------------------------------------------
    #        ADDED: FUNCTION TO GET NEAREST CENTER (NO CV CHANGE)
    # ---------------------------------------------------------------------
    def get_nearest_point(self, centers):
        arr = np.array(centers)
        dist = np.linalg.norm(arr, axis=1)
        idx = np.argmin(dist)
        return arr[idx]
    
    # ---------------------------------------------------------------------
    #        ADDED: DEDUPLICATE NEARBY SHAPE CENTERS
    # ---------------------------------------------------------------------
    def dedupe_centers(self, centers, min_dist=50):
        """Remove duplicate detections that are too close together."""
        if len(centers) <= 1:
            return centers
        
        unique = []
        for c in centers:
            is_dup = False
            for u in unique:
                if np.linalg.norm(np.array(c) - np.array(u)) < min_dist:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
        return unique
    # ---------------------------------------------------------------------

    # --------------------------------- (rest of your original functions untouched)
    def segment_lines_ransac(self, pts):
        pts_list = [tuple(p) for p in pts.tolist()]
        lines = []

        def line_angle(x1, y1, x2, y2):
            return np.arctan2(y2 - y1, x2 - x1)

        for i in range(self.iterations):
            if len(pts_list) < 2:
                break

            p1, p2 = random.sample(pts_list, 2)
            x1, y1 = p1
            x2, y2 = p2

            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2

            inliers = []
            for p in pts_list:
                dist = abs(A * p[0] + B * p[1] + C) / np.sqrt(A * A + B * B)
                if dist < self.threshold:
                    inliers.append(p)

            if self.min_inliers <= len(inliers) <= self.max_inliers:
                inliers_np = np.array(inliers)
                x_min, y_min = np.min(inliers_np, axis=0)
                x_max, y_max = np.max(inliers_np, axis=0)
                new_line = (int(x_min), int(y_min), int(x_max), int(y_max), len(inliers))
                new_angle = line_angle(x_min, y_min, x_max, y_max)

                duplicate_found = False
                for existing in lines:
                    x1e, y1e, x2e, y2e, _ = existing
                    ex_angle = line_angle(x1e, y1e, x2e, y2e)
                    angle_diff = abs(new_angle - ex_angle)
                    angle_diff = min(angle_diff, np.pi - angle_diff)
                    mid_new = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
                    mid_old = np.array([(x1e + x2e) / 2, (y1e + y2e) / 2])
                    dist_mid = np.linalg.norm(mid_new - mid_old)
                    if angle_diff < self.duplicate_angle_thresh and dist_mid < self.duplicate_dist_thresh:
                        duplicate_found = True
                        break

                if duplicate_found:
                    continue

                merged = False
                for j, existing in enumerate(lines):
                    x1e, y1e, x2e, y2e, inl_e = existing
                    ex_angle = line_angle(x1e, y1e, x2e, y2e)
                    angle_diff = abs(new_angle - ex_angle)
                    angle_diff = min(angle_diff, np.pi - angle_diff)
                    mid_new = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
                    mid_old = np.array([(x1e + x2e) / 2, (y1e + y2e) / 2])
                    dist_mid = np.linalg.norm(mid_new - mid_old)
                    if angle_diff < 0.01 and dist_mid < 70:                           #################   merge difference
                        merged_line = (
                            int(min(x1e, x_min)),
                            int(min(y1e, y_min)),
                            int(max(x2e, x_max)),
                            int(max(y2e, y_max)),
                            max(inl_e, len(inliers))
                        )
                        lines[j] = merged_line
                        merged = True
                        break

                if not merged:
                    lines.append(new_line)

                pts_list = [p for p in pts_list if p not in inliers]

        return lines


    def detect_square_shape(self, lines):
        if len(lines) < 3:
            return []

        def angle(l):
            return np.degrees(np.arctan2(l[3] - l[1], l[2] - l[0])) % 180

        def intersect(l1, l2):
            x1, y1, x2, y2, _ = l1
            x3, y3, x4, y4, _ = l2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
                  (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
                  (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return (px, py)

        shape_centers = []
        angles = [angle(l) for l in lines]

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                diff = abs(angles[i] - angles[j])
                if diff < 5 or abs(diff - 180) < 5:
                    for k in range(len(lines)):
                        if k == i or k == j:
                            continue
                        perp_diff = abs(angles[k] - angles[i])
                        if 80 <= perp_diff <= 100:
                            p1 = intersect(lines[i], lines[k])
                            p2 = intersect(lines[j], lines[k])
                            if p1 and p2:
                                d = np.linalg.norm(np.array(p1) - np.array(p2))
                                if 30 < d < 200:
                                    cx, cy = np.mean([p1[0], p2[0]]), np.mean([p1[1], p2[1]])
                                    shape_centers.append((cx, cy))
        return shape_centers


    def detect_triangle_shape(self, lines, triangle_angle=50, tol=10):
        if len(lines) < 2:
            return []

        centers = []
        angles = []

        lines = np.array(lines, dtype=float)

        for (x1, y1, x2, y2, _) in lines:
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(ang)

        # -----------------------------
        # segment-only intersection
        # -----------------------------
        def segment_intersection(l1, l2):
            x1, y1, x2, y2, _ = l1
            x3, y3, x4, y4, _ = l2

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None

            px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
                (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
                (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

            # check intersection lies within both segments
            def within(a, b, c):
                return min(a, b) - 1 <= c <= max(a, b) + 1

            if (within(x1, x2, px) and within(y1, y2, py) and
                within(x3, x4, px) and within(y3, y4, py)):
                return (px, py)

            return None
        # -----------------------------

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                diff = abs(angles[i] - angles[j])
                diff = min(diff, 180 - diff)

                if abs(diff - triangle_angle) <= tol:
                    p = segment_intersection(lines[i], lines[j])
                    if p:
                        centers.append(p)

        return centers




def main(args=None):
    rclpy.init(args=args)
    node = LineRansacNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()