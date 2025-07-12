import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class NavigatorNode(Node):
    MIN_BLOB_WIDTH = 5  # pixels
    BRANCH_CX_TOL = 25  # pixels
    BRANCH_WINDOW = 40  # pixels (tune if needed)

    def __init__(self):
        super().__init__('navigator_node')

        # --- Parameters ---
        # Normalized y-positions of scan lines (0.0 = top, 1.0 = bottom)
        # Closer scan lines have higher priority
        self.scan_lines = [
            0.625, 0.666, 0.708, 0.75, 0.791, 0.833
        ]
        # Weights biased toward the lower part of the image (sum to 1.0)
        self.weights = [
            0.1, 0.1, 0.15, 0.2, 0.25, 0.2
        ]
        self.operation_gain = 0.005            # Gain for deviation-to-angular conversion
        # Motor max speed: 185 RPM ±15% -> ~212.8 RPM at no load
        # With 28 mm wheel radius, the theoretical max is about 0.62 m/s
        self.min_linear = 0.1                  # Min linear velocity [m/s]
        self.max_linear = 0.54                 # Max linear velocity [m/s]
        self.max_angular = 0.6                 # Max angular velocity [rad/s]
        self.alpha = 0.7                       # Low-pass filter coefficient
        self.prev_linear = 0.1                 # Previous linear velocity
        self.prev_cx: float | None = None      # Previous line center

        self.bridge = CvBridge()

        # --- Scan line state parameters ---
        self.BLUE_PIXEL_THRESHOLD = 20
        self.BLUE_RATIO_THRESHOLD = 0.10
        self.blue_lower = np.array([90, 100, 50], dtype=np.uint8)
        self.blue_upper = np.array([130, 255, 255], dtype=np.uint8)
        self.sl_state = ["normal"] * len(self.scan_lines)
        # Counter to lock branch selection until all scan-lines have decided
        self.pending_branch = len(self.scan_lines)

        # Debug parameter to enable OpenCV visualization
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').value

        # --- Publisher and Subscriber ---
        self.sub = self.create_subscription(
            Image, '/camera_top/camera_top/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("NavigatorNode started. Waiting for camera images...")

    def image_callback(self, msg):
        """Process incoming camera image and publish velocity commands."""
        cv_image, binary, hsv = self._preprocess(msg)

        cx_list, debug_info, branch_cx = self._process_scan_lines(binary, hsv)
        if not cx_list:
            return self._handle_no_line(cv_image, debug_info)

        deviation, confidence, averaged_cx = self._compute_velocity_params(
            cx_list, binary.shape[1]
        )
        self._update_prev_cx(branch_cx, averaged_cx)

        linear, angular = self._compute_command(deviation)
        self._publish_cmd(linear, angular)

        # log per-scanline blob info: start,width,chosen_center,state
        entries: list[str] = []
        for y, candidates, chosen, state in debug_info:
            if candidates and chosen is not None:
                for start, w, cx in candidates:
                    if cx == chosen:
                        entries.append(f"{start},{w},{cx},{state}")
                        break
                else:
                    entries.append(f"-,-,{chosen},{state}")
            else:
                entries.append(f"-,-,{chosen},{state}")
        self.get_logger().info(" ".join(entries))

        if self.debug:
            self._show_debug(
                cv_image,
                debug_info,
                deviation=deviation,
                angular=angular,
                confidence=confidence,
            )

    def _preprocess(self, msg):
        """Convert ROS Image message to OpenCV binary and HSV images."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        return cv_image, binary, hsv

    def _process_scan_lines(self, binary, hsv):
        """Detect line positions on scan lines, handling branch logic."""
        height, width = binary.shape[:2]
        cx_list = []
        debug_info = []
        base_cx = self.prev_cx if self.prev_cx is not None else width // 2
        branch_cx = None
        for i, (ratio, weight) in enumerate(zip(self.scan_lines, self.weights)):
            # Lock branch center during selection phase
            if self.pending_branch > 0 and branch_cx is not None:
                self.prev_cx = branch_cx

            branch_cx = self._process_scan_line(
                i, ratio, weight, binary, hsv, base_cx, branch_cx,
                cx_list, debug_info
            )

        return cx_list, debug_info, branch_cx

    def _process_scan_line(
        self, i, ratio, weight,
        binary, hsv,
        base_cx, branch_cx,
        cx_list, debug_info
    ):
        """Process a single scan line and update cx_list, debug_info, and branch selection."""
        # prepare scan line
        y, row, hsv_row = self._compute_scanline_data(ratio, binary, hsv)
        blue_count, blue_ratio, blue_present = self._analyze_blue(
            hsv_row, row.size)
        indices = np.where(row == 255)[0]
        state = self._update_state(
            self.sl_state[i], blue_present, bool(indices.size))

        if indices.size:
            candidates = self._detect_blob_centers(indices)
            chosen_cx, branch_cx, state = self._handle_branch(
                candidates, state, base_cx, branch_cx, cx_list, debug_info,
                y, weight, blue_count, blue_ratio
            )
            cx_list.append((chosen_cx, weight))
        else:
            candidates: list[tuple[int, int, int]] = []
            chosen_cx = branch_cx
            debug_info.append((y, candidates, chosen_cx, state))

        self.sl_state[i] = state
        return branch_cx

    def _compute_scanline_data(self, ratio, binary, hsv):
        """Return y-coordinate, binary row, and HSV row for a given scanline ratio."""
        height, width = binary.shape[:2]
        y = int(ratio * height)
        return y, binary[y, :], hsv[y: y + 1, :, :]

    def _detect_blob_centers(self, indices: np.ndarray) -> list[tuple[int, int, int]]:
        """Split indices into blobs and return their start x, width, and center x."""
        splits = np.where(np.diff(indices) > 1)[0] + 1
        blobs = np.split(indices, splits)
        result: list[tuple[int, int, int]] = []
        for blob in blobs:
            start = int(blob[0])
            w = int(blob[-1] - blob[0] + 1)
            cx = int(np.mean(blob))
            result.append((start, w, cx))
        return result

    def _analyze_blue(self, hsv_row: np.ndarray, width: int) -> tuple[int, float, bool]:
        """Count blue pixels on a scanline and determine presence of blue region."""
        mask = cv2.inRange(hsv_row, self.blue_lower, self.blue_upper)
        count = int(cv2.countNonZero(mask))
        ratio = count / width
        present = count > self.BLUE_PIXEL_THRESHOLD or ratio >= self.BLUE_RATIO_THRESHOLD
        return count, ratio, present

    def _update_state(self, state: str, blue: bool, black: bool) -> str:
        """Advance scan-line state machine based on color detection."""
        if state == "normal" and blue:
            return "blue_detected"
        if state == "blue_detected" and not blue and black:
            return "blue_to_black"
        return state

    def _handle_branch(
        self,
        candidates: list[tuple[int, int]],
        state: str,
        base_cx: float,
        branch_cx: int | None,
        cx_list: list[tuple[int, float]],
        debug_info: list[tuple[int, int | None, str, int, float]],
        y: int,
        weight: float,
        blue_count: int,
        blue_ratio: float,
    ) -> tuple[int, int | None, str]:
        """Select blob center and handle branching logic, updating lists as needed."""
        target = branch_cx if branch_cx is not None else base_cx
        # choose initial blob candidate center
        _, _, chosen_cx = min(candidates, key=lambda c: abs(c[2] - target))

        if state == "blue_to_black" and branch_cx is None:
            near = [
                c for c in candidates
                if abs(c[2] - base_cx) <= self.BRANCH_WINDOW
                and c[1] >= self.MIN_BLOB_WIDTH
            ]
            if len(near) >= 2:
                # choose the rightmost valid branch
                _, _, cx2 = max(near, key=lambda c: c[2])
                branch_cx = cx2
                # retroactively update previous entries with preserved candidates
                cx_list[:] = [(branch_cx, w_prev) for (_, w_prev) in cx_list]
                debug_info[:] = [
                    (y0, candidates0, branch_cx, state0)
                    for (y0, candidates0, _, state0) in debug_info
                ]
                state = "normal"
                self.pending_branch -= 1

        debug_info.append((y, candidates, chosen_cx, state))
        return chosen_cx, branch_cx, state

    def _compute_velocity_params(self, cx_list, width):
        """Compute deviation, confidence, and averaged center from cx_list."""
        confidence = len(cx_list) / len(self.scan_lines)
        total_weight = sum(w for _, w in cx_list)
        deviation = sum((cx - width // 2) * w for cx,
                        w in cx_list) / total_weight
        averaged_cx = sum(cx * w for cx, w in cx_list) / total_weight
        return deviation, confidence, averaged_cx

    def _update_prev_cx(self, branch_cx, averaged_cx):
        """Update prev_cx and reset pending_branch when needed."""
        if branch_cx is None:
            self.prev_cx = averaged_cx
        else:
            if self.pending_branch == 0:
                self.prev_cx = branch_cx
                self.pending_branch = len(self.scan_lines)
            else:
                self.prev_cx = branch_cx

    def _compute_command(self, deviation):
        """Compute linear and angular velocities from deviation."""
        angular = np.clip(
            -deviation * self.operation_gain,
            -self.max_angular,
            self.max_angular,
        )
        norm_ang = min(abs(angular) / self.max_angular, 1.0)
        new_linear = self.max_linear - (
            self.max_linear - self.min_linear
        ) * norm_ang
        linear = self.alpha * self.prev_linear + (
            1.0 - self.alpha
        ) * new_linear
        self.prev_linear = linear
        return linear, angular

    def _publish_cmd(self, linear, angular):
        """Publish Twist command and log it."""
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.pub.publish(cmd)
        # self.get_logger().info(
        #     f"cmd_vel: linear={linear:.4f}, angular={angular:.4f}"
        # )

    def _handle_no_line(self, cv_image, debug_info):
        """Handle case when no line is detected."""
        if self.debug:
            self._show_debug(cv_image, debug_info, message="No line detected")
        self.get_logger().warn("No line detected.")
        return

    def _show_debug(
        self,
        image: np.ndarray,
        debug_info,
        deviation: float | None = None,
        angular: float | None = None,
        confidence: float | None = None,
        message: str | None = None,
    ) -> None:
        """Draw debug information on the image and display it.

        ``debug_info`` is a list of tuples ``(y, cx_or_none, state, blue_count, blue_ratio)``.
        """
        debug_image = image.copy()
        width = image.shape[1]

        for y, candidates, chosen_cx, state in debug_info:
            # draw scan line
            cv2.line(debug_image, (0, y), (width - 1, y), (255, 0, 0), 1)
            # draw blob candidates
            for start, w, cx in candidates:
                color = (0, 255, 0)
                cv2.drawMarker(
                    debug_image,
                    (cx, y),
                    color,
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=8,
                    thickness=1,
                )
            # highlight chosen blob
            if chosen_cx is not None:
                cv2.circle(debug_image, (chosen_cx, y), 4, (0, 255, 0), -1)
            # draw state label
            cv2.putText(
                debug_image,
                state,
                (10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if message:
            cv2.putText(
                debug_image,
                message,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                debug_image,
                f"Deviation: {deviation:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f"Angular: {angular:.2f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f"Confidence: {confidence:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("debug", debug_image)
        cv2.waitKey(1)


def main(args=None):
    """Entry point for the runner node."""
    rclpy.init(args=args)
    runner = NavigatorNode()
    try:
        rclpy.spin(runner)
    except KeyboardInterrupt:
        pass
    runner.destroy_node()
    rclpy.try_shutdown()
    if runner.debug:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
