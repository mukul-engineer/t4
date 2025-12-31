#!/usr/bin/env python3
"""
Hybrid Waypoint + Manual Motion Node (v7-final)

Based on your v6-final node. Adds fruit-handling sequence after reaching W2:

Sequence (high level):
- Same v6 flow (W1 -> attach fertiliser_can -> back to start -> rotate Pose A ->
  go to custom waypoint -> detach fertiliser_can -> return to Pose A -> joint motion -> W2)
- After reaching W2: for each fruit TF (KC_05_bad_fruit_1..3):
    * lookup TF (base_link -> KC_05_bad_fruit_i)
    * move to fruit position (pos tolerance 0.05) but keep orientation = orientation recorded at W2
    * call attach (model1_name='bad_fruit')
    * move back to W2
    * move to W3
    * call detach (model1_name='bad_fruit')
    * move back to W2
- After all fruits processed -> stop and shutdown
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray
from linkattacher_msgs.srv import AttachLink, DetachLink
import numpy as np
import time

# TF imports
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

# ----------------------------- Quaternion helpers -----------------------------
def quat_normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([0.0, 0.0, 0.0, 1.0])

def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def small_angle_from_quat_error(target_q, current_q):
    tq = quat_normalize(target_q)
    cq = quat_normalize(current_q)
    qe = quat_mul(tq, quat_conjugate(cq))
    if qe[3] < 0:
        qe = -qe
    return 2.0 * qe[:3]

# ----------------------------- Main Node -----------------------------
class HybridWaypointNode(Node):
    def __init__(self):
        super().__init__('hybrid_waypoint_servo_node_v7')
        self.get_logger().info("HybridWaypointNode v7 running.")

        # ROS interfaces
        self.pose_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.pose_callback, 10)
        self.twist_pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, '/delta_joint_cmds', 10)
        self.attach_cli = self.create_client(AttachLink, '/attach_link')
        self.detach_cli = self.create_client(DetachLink, '/detach_link')

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Gains and limits (same as v6)
        self.Kp_pos = 1.0
        self.Kp_ori = 1.0
        self.POS_TOL = 0.05      # positional tolerance
        self.ORI_TOL = 0.05
        self.MAX_VEL = 0.5
        self.MAX_ANG = 1.0

        # Motion times
        self.TIME_A = 8.0
        self.ANGULAR_Y_SPEED = 0.24

        # Internal states
        self.current_pose = None         # (pos np.array(3), quat np.array(4))
        self.start_pose = None
        self.poseA = None                # pose recorded after rotation
        self.pose_at_W2 = None           # orientation recorded when reaching W2 (used for fruit approach)
        self.phase = 0
        self.motion_end = None
        self.hold_until = None
        self.joint_motion_end = None

        # Waypoints (same coordinates as your v6-final)
        self.waypoints = [
            (np.array([-0.214, -0.532, 0.6]), quat_normalize(np.array([0.707, 0.028, 0.034, 0.707]))),  # W1
            (np.array([-0.259, 0.501, 0.515]), quat_normalize(np.array([0.029, 0.997, 0.045, 0.033]))), # W2
            (np.array([-0.806, 0.010, 0.182]), quat_normalize(np.array([-0.684, 0.726, 0.05, 0.008]))), # W3
        ]
        self.target_index = 0

        # Fruit TF frames (the TF frames exist but positions are random each run)
        self.fruit_frames = [
            '5167_bad_fruit_1',
            '5167_bad_fruit_2',
            '5167_bad_fruit_3'
        ]
        self.current_fruit_idx = 0
        self.processing_fruit = False

        # Timer for main control loop (50 Hz)
        self.timer = self.create_timer(0.02, self.loop)

    # ------------------------- Pose callback -------------------------
    def pose_callback(self, msg):
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                         msg.pose.orientation.z, msg.pose.orientation.w], dtype=float)
        self.current_pose = (pos, quat)
        if self.start_pose is None:
            self.start_pose = (pos.copy(), quat.copy())
            self.get_logger().info("Start pose recorded.")

    def goto_pose_xy_then_z(self, target, pos_tol=None):
        """
        Stepwise approach: first correct XY (keep Z fixed), then adjust Z.
        Uses same Kp control and limits.
        """
        if pos_tol is None:
            pos_tol = self.POS_TOL

        cur_pos, cur_quat = self.current_pose
        target_pos, target_quat = target

        # 1️⃣ Phase 1: Correct XY first
        xy_err_vec = np.array([target_pos[0] - cur_pos[0],
                               target_pos[1] - cur_pos[1],
                               0.0])
        xy_err = np.linalg.norm(xy_err_vec)

        # 2️⃣ Phase 2: After XY small enough, correct Z
        z_err = target_pos[2] - cur_pos[2]
        z_err_abs = abs(z_err)

        # If XY error is significant, correct XY only (Z=0)
        tw = Twist()
        if xy_err > pos_tol:
            lin = self.Kp_pos * xy_err_vec
            lin_norm = np.linalg.norm(lin)
            if lin_norm > self.MAX_VEL:
                lin *= self.MAX_VEL / lin_norm
            tw.linear.x, tw.linear.y, tw.linear.z = lin.tolist()
        else:
            # XY okay → now correct Z
            if z_err_abs > pos_tol:
                tw.linear.z = np.clip(self.Kp_pos * z_err, -self.MAX_VEL, self.MAX_VEL)
            else:
                self.publish_zero_twist()
                return True

        self.twist_pub.publish(tw)
        return False


    # ------------------------- Control Loop -------------------------
    def loop(self):
        # wait for pose
        if self.current_pose is None:
            return

        # ---------- Phase 0: Go to W1 ----------
        if self.phase == 0:
            if self.goto_pose(self.waypoints[self.target_index]):
                self.get_logger().info("Reached W1 → attaching fertiliser_can...")
                self.call_attach_service(model1='fertiliser_can')
                self.start_hold(3, "returning to start")
                self.phase = 1

        # ---------- Phase 1: Return to start ----------
        elif self.phase == 1:
            if self.hold_done():
                if self.goto_pose(self.start_pose):
                    self.get_logger().info("Back to start → Pose A rotation")
                    self.start_hold(2, "Pose A rotation")
                    self.phase = 2

        # ---------- Phase 2: Rotate (Pose A) ----------
        elif self.phase == 2:
            if self.hold_done():
                if self.motion_end is None:
                    self.motion_end = time.time() + self.TIME_A
                    self.get_logger().info("Rotating (angular Y)...")

                if time.time() < self.motion_end:
                    tw = Twist()
                    tw.angular.y = self.ANGULAR_Y_SPEED
                    self.twist_pub.publish(tw)
                else:
                    self.publish_zero_twist()
                    cur_pos, cur_quat = self.current_pose
                    self.poseA = (cur_pos.copy(), cur_quat.copy())
                    self.motion_end = None
                    self.get_logger().info("Pose A recorded → moving to custom waypoint [0.710, 0.006, 0.346]")
                    custom_wp = (np.array([0.800, 0.20, 0.390]), cur_quat.copy())
                    self.custom_waypoint = custom_wp
                    self.phase = 2.5

        # ---------- Phase 2.5: Go to custom waypoint ----------
        elif self.phase == 2.5:
            if self.goto_pose(self.custom_waypoint):
                self.get_logger().info("Reached custom waypoint → Detaching fertiliser_can...")
                self.call_detach_service(model1='fertiliser_can')
                self.start_hold(3, "returning to Pose A")
                self.phase = 3

        # ---------- Phase 3: Return to Pose A ----------
        elif self.phase == 3:
            if self.hold_done():
                if self.goto_pose(self.poseA):
                    self.get_logger().info("Returned to Pose A → starting joint motion (15s)")
                    # publish joint command for 5s (we'll publish in the loop)
                    self.joint_motion_end = time.time() + 5.0
                    self.phase = 4

        # ---------- Phase 4: Publish joint command for 15s ----------
        elif self.phase == 4:
            if time.time() < self.joint_motion_end:
                cmd = Float64MultiArray()
                # as requested earlier: value 1.0 in first element
                cmd.data = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.joint_pub.publish(cmd)
            else:
                self.publish_zero_joints()
                self.get_logger().info("15s joint command done → going to W2")
                # record orientation at W2 when we actually reach it
                self.start_hold(0.5, "Move to W2")
                self.phase = 5

        # ---------- Phase 5: Go to W2 ----------
        elif self.phase == 5:
            if self.hold_done():
                if self.goto_pose(self.waypoints[1]):
                    self.get_logger().info("Reached W2 → start fruit processing")
                    # record orientation at W2 (we will keep this orientation when approaching fruits)
                    _, w2_quat = self.waypoints[1]
                    # but better to record actual orientation at the instant we reached W2:
                    if self.current_pose is not None:
                        _, cur_quat = self.current_pose
                        self.pose_at_W2 = (self.waypoints[1][0].copy(), cur_quat.copy())
                    else:
                        self.pose_at_W2 = self.waypoints[1]
                    self.phase = 6
                    self.current_fruit_idx = 0
                    self.processing_fruit = False

        # ---------- Phase 6: Record all fruit TFs once ----------
        elif self.phase == 6:
            self.get_logger().info("Recording all fruit TFs once...")
            self.recorded_fruit_poses = []
            for frame in self.fruit_frames:
                fruit_pose = self.lookup_fruit_pose(frame)
                if fruit_pose is None:
                    self.get_logger().warn(f"TF not available for {frame}, skipping.")
                    continue
                fruit_pos, _ = fruit_pose
                # keep W2 orientation for all fruit approaches
                if self.pose_at_W2 is not None:
                    _, ori_at_W2 = self.pose_at_W2
                else:
                    _, ori_at_W2 = self.waypoints[1]
                self.recorded_fruit_poses.append((fruit_pos, ori_at_W2.copy()))
                self.get_logger().info(f"Recorded {frame} at {fruit_pos.tolist()}")

            if not self.recorded_fruit_poses:
                self.get_logger().warn("No fruit TFs recorded, skipping fruit sequence.")
                self.phase = 10
                return

            self.current_fruit_idx = 0
            self.processing_fruit = False
            self.phase = 6.5

        # ---------- Phase 6.5: Process each recorded fruit ----------
        elif self.phase == 6.5:
            # done all?
            if self.current_fruit_idx >= len(self.recorded_fruit_poses):
                self.get_logger().info("✅ All fruits processed → shutting down.")
                self.publish_zero_twist()
                rclpy.shutdown()
                return

            target_for_fruit = self.recorded_fruit_poses[self.current_fruit_idx]

            if not self.processing_fruit:
                self.get_logger().info(f"🍎 Moving to recorded fruit #{self.current_fruit_idx+1}")
                self.processing_fruit = True
                return

            # move to fruit position (pos only)
            if self.goto_pose_xy_then_z(target_for_fruit, pos_tol=self.POS_TOL):
                self.get_logger().info(f"Reached fruit #{self.current_fruit_idx+1} → attaching bad_fruit")
                self.call_attach_service(model1='bad_fruit')

                # lift up
                lift_end_time = time.time() + 2.0
                while time.time() < lift_end_time and rclpy.ok():
                    tw = Twist()
                    tw.linear.z = 0.12
                    self.twist_pub.publish(tw)
                    time.sleep(0.02)
                self.publish_zero_twist()

                self.start_hold(0.5, "return to W2 after attach")
                self.phase = 7

        # ---------- Phase 7: Return to W2 ----------
        elif self.phase == 7:
            if self.hold_done():
                if self.goto_pose(self.waypoints[1]):
                    self.get_logger().info("Returned to W2 → moving to W3")
                    self.phase = 8

        # ---------- Phase 8: Go to W3 and detach ----------
        elif self.phase == 8:
            if self.goto_pose(self.waypoints[2]):
                self.get_logger().info("Reached W3 → detaching bad_fruit")
                self.call_detach_service(model1='bad_fruit')
                self.start_hold(0.5, "return to W2 after detach")
                self.phase = 9

        # ---------- Phase 9: Return to W2 and next fruit ----------
        elif self.phase == 9:
            if self.hold_done():
                if self.goto_pose(self.waypoints[1]):
                    self.get_logger().info("Returned to W2 → next fruit")
                    self.current_fruit_idx += 1
                    self.processing_fruit = False
                    self.phase = 6.5


    # ------------------------- Helper methods -------------------------
    def goto_pose(self, target, require_orientation=True, pos_tol=None, ori_tol=None):
        """
        Move toward target pose using same Kp-based twist publisher as v6.
        target: (np.array(3), np.array(4)) or special formats.
        If require_orientation is False, only position error is checked and orientation component is ignored.
        """
        if pos_tol is None:
            pos_tol = self.POS_TOL
        if ori_tol is None:
            ori_tol = self.ORI_TOL

        cur_pos, cur_quat = self.current_pose
        target_pos, target_quat = target

        pos_err_vec = target_pos - cur_pos
        pos_err = np.linalg.norm(pos_err_vec)

        if require_orientation:
            ori_err_vec = small_angle_from_quat_error(target_quat, cur_quat)
            ori_err = np.linalg.norm(ori_err_vec)
        else:
            ori_err_vec = np.array([0.0, 0.0, 0.0])
            ori_err = 0.0

        # reached?
        if pos_err < pos_tol and ori_err < ori_tol:
            self.publish_zero_twist()
            return True

        # compute commanded twist
        lin = self.Kp_pos * pos_err_vec
        ang = self.Kp_ori * ori_err_vec

        lin_norm = np.linalg.norm(lin)
        ang_norm = np.linalg.norm(ang)
        if lin_norm > self.MAX_VEL:
            lin *= self.MAX_VEL / lin_norm
        if ang_norm > self.MAX_ANG:
            ang *= self.MAX_ANG / ang_norm

        tw = Twist()
        tw.linear.x, tw.linear.y, tw.linear.z = lin.tolist()
        tw.angular.x, tw.angular.y, tw.angular.z = ang.tolist()
        self.twist_pub.publish(tw)
        return False

    # ------------------------- Services -------------------------
    def call_attach_service(self, model1='fertiliser_can'):
        if not self.attach_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Attach service not available!")
            return
        req = AttachLink.Request()
        req.model1_name = model1
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        future = self.attach_cli.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info(f"Attach service called (model1={model1})."))

    def call_detach_service(self, model1='fertiliser_can'):
        if not self.detach_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Detach service not available!")
            return
        req = DetachLink.Request()
        req.model1_name = model1
        req.link1_name = 'body'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        future = self.detach_cli.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info(f"Detach service called (model1={model1})."))

    # ------------------------- TF Helper -------------------------
    def lookup_fruit_pose(self, frame_name, timeout_sec=1.0):
        """
        Lookup transform base_link -> frame_name.
        Returns (pos np.array(3), quat np.array(4)) or None if not available.
        """
        try:
            # use latest transform
            trans = self.tf_buffer.lookup_transform('base_link', frame_name, rclpy.time.Time())
            t = trans.transform.translation
            r = trans.transform.rotation
            pos = np.array([t.x, t.y, t.z], dtype=float)
            quat = np.array([r.x, r.y, r.z, r.w], dtype=float)
            return (pos, quat)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # not available yet
            return None
        except Exception as e:
            self.get_logger().error(f"Unexpected TF lookup error for {frame_name}: {e}")
            return None

    # ------------------------- Misc -------------------------
    def start_hold(self, sec, msg):
        self.hold_until = time.time() + sec
        self.get_logger().info(f"Holding {sec}s → {msg}")

    def hold_done(self):
        return self.hold_until is None or time.time() >= self.hold_until

    def publish_zero_twist(self):
        self.twist_pub.publish(Twist())

    def publish_zero_joints(self):
        cmd = Float64MultiArray()
        cmd.data = [0.0] * 6
        self.joint_pub.publish(cmd)

# ----------------------------- Main --------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = HybridWaypointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt → shutting down.")
    finally:
        # try to zero commands
        try:
            node.publish_zero_twist()
            node.publish_zero_joints()
        except Exception:
            pass
        # destroy and shutdown if not already
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()