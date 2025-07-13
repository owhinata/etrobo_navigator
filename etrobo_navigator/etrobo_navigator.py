from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


_STATE_ABBR = {
    "normal": "norm",
    "blue_detected": "bdet",
    "blue_to_black": "btbl",
}


@dataclass
class ScanLineContext:
    y: int
    weight: float
    state: str
    base_cx: float
    branch_cx: Optional[int]
    blue_count: int
    blue_ratio: float


@dataclass
class BlobCandidate:
    start: int
    width: int
    cx: int


@dataclass
class DebugEntry:
    y: int
    candidates: list[BlobCandidate]
    chosen_cx: Optional[int]
    state: str


@dataclass
class BranchResult:
    chosen_cx: int
    branch_cx: Optional[int]
    state: str


@dataclass
class ScanLineResult:
    chosen_cx: Optional[int]
    branch_cx: Optional[int]
    state: str
    weight: float
    debug_entry: DebugEntry


@dataclass
class WeightedCenter:
    cx: int
    weight: float


@dataclass
class ProcessingState:
    branch_cx: Optional[int]
    pending_branch: int

    def detect_branch(self, new_branch_cx: int) -> None:
        """Detect a new branch and update state accordingly."""
        self.branch_cx = new_branch_cx
        self.pending_branch -= 1

    def should_lock_branch(self) -> bool:
        """Check if branch should be locked during selection phase."""
        return self.pending_branch > 0 and self.branch_cx is not None

    def is_branch_selection_complete(self) -> bool:
        """Check if branch selection phase is complete."""
        return self.pending_branch == 0


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
        # Global counter for completed blue events (for alternating branch direction)
        self.branch_count = 0

        # Debug parameter to enable OpenCV visualization
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').value

        # --- Publisher and Subscriber ---
        self.sub = self.create_subscription(
            Image, '/camera_top/camera_top/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("NavigatorNode started. Waiting for camera images...")

    def _format_debug_log_entry(self, debug_entry: DebugEntry) -> str:
        """Format a single debug entry for logging."""
        abbr = _STATE_ABBR.get(debug_entry.state, debug_entry.state)
        if debug_entry.candidates:
            parts = [abbr]
            for candidate in debug_entry.candidates:
                if candidate.cx == debug_entry.chosen_cx:
                    parts.append(f"{candidate.start},{candidate.width}*")
                else:
                    parts.append(f"{candidate.start},{candidate.width}")
            return ",".join(parts)
        else:
            return f"{abbr},-,-"

    def _generate_debug_log(self, debug_entries: list[DebugEntry]) -> None:
        """Generate and log debug information for all scan lines."""
        entries = [self._format_debug_log_entry(
            entry) for entry in debug_entries]
        self.get_logger().info("\n" + "\n".join(entries))

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

        # log per-scanline blob info: state and blob candidates (start,width; chosen marked with *)
        self._generate_debug_log(debug_info)

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

        # per-line previous center initialization (first frame)
        if not hasattr(self, 'prev_cx_per_line'):
            self.prev_cx_per_line = [None] * len(self.scan_lines)
        processing_state = ProcessingState(
            branch_cx=None,
            pending_branch=self.pending_branch
        )

        for i, (ratio, weight) in enumerate(zip(self.scan_lines, self.weights)):
            # Lock branch center during selection phase
            if processing_state.should_lock_branch():
                self.prev_cx = processing_state.branch_cx

            result = self._process_scan_line(
                i, ratio, weight, binary, hsv, base_cx, processing_state
            )

            # Extract results
            if result.chosen_cx is not None:
                cx_list.append(WeightedCenter(result.chosen_cx, result.weight))

            debug_info.append(result.debug_entry)

            # Centralized state management
            if result.branch_cx != processing_state.branch_cx and result.branch_cx is not None:
                # Handle retroactive update of cx_list when branch is detected
                cx_list[:] = [WeightedCenter(
                    result.branch_cx, wc.weight) for wc in cx_list]
                processing_state.detect_branch(result.branch_cx)
            else:
                processing_state.branch_cx = result.branch_cx

        # Update instance state after processing all scan lines
        self.pending_branch = processing_state.pending_branch
        return cx_list, debug_info, processing_state.branch_cx

    def _process_scan_line(
        self, i: int, ratio: float, weight: float,
        binary: np.ndarray, hsv: np.ndarray,
        base_cx: float, processing_state: ProcessingState
    ) -> ScanLineResult:
        """Process a single scan line and return structured result."""
        # prepare scan line
        y, row, hsv_row = self._compute_scanline_data(ratio, binary, hsv)
        blue_count, blue_ratio, blue_present = self._analyze_blue(
            hsv_row, row.size)
        indices = np.where(row == 255)[0]
        state = self._update_state(
            self.sl_state[i], blue_present, bool(indices.size))

        if indices.size:
            candidates = self._detect_blob_centers(indices)
            context = ScanLineContext(
                y=y, weight=weight, state=state, base_cx=base_cx,
                branch_cx=processing_state.branch_cx, blue_count=blue_count, blue_ratio=blue_ratio
            )

            result, debug_entry = self._handle_branch_with_context(
                i, candidates, context)
            chosen_cx = result.chosen_cx
            branch_cx = result.branch_cx
            state = result.state
        else:
            candidates: list[BlobCandidate] = []
            chosen_cx = processing_state.branch_cx
            branch_cx = processing_state.branch_cx
            debug_entry = DebugEntry(y, candidates, chosen_cx, state)

        self.sl_state[i] = state
        return ScanLineResult(
            chosen_cx=chosen_cx,
            branch_cx=branch_cx,
            state=state,
            weight=weight,
            debug_entry=debug_entry
        )

    def _compute_scanline_data(self, ratio, binary, hsv):
        """Return y-coordinate, binary row, and HSV row for a given scanline ratio."""
        height, width = binary.shape[:2]
        y = int(ratio * height)
        return y, binary[y, :], hsv[y: y + 1, :, :]

    def _detect_blob_centers(self, indices: np.ndarray) -> list[BlobCandidate]:
        """Split indices into blobs and return their start x, width, and center x."""
        splits = np.where(np.diff(indices) > 1)[0] + 1
        blobs = np.split(indices, splits)
        result: list[BlobCandidate] = []
        for blob in blobs:
            start = int(blob[0])
            width = int(blob[-1] - blob[0] + 1)
            cx = int(np.mean(blob))
            result.append(BlobCandidate(start, width, cx))
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

    def _log_blob_selection_debug(self, candidates: list[BlobCandidate], target_cx: float, chosen_cx: int) -> None:
        """Log debug information for blob selection."""
        cxs = [c.cx for c in candidates]
        distances = [abs(cx - target_cx) for cx in cxs]
        self.get_logger().debug(
            f"candidates cx={cxs}, "
            f"target={target_cx}, "
            f"distances={distances}, "
            f"chosen_cx={chosen_cx}"
        )

    def _select_blob_center(self, candidates: list[BlobCandidate], target_cx: float) -> int:
        """Select the best blob center from candidates based on target."""
        chosen_cx = min(candidates, key=lambda c: abs(c.cx - target_cx)).cx
        self._log_blob_selection_debug(candidates, target_cx, chosen_cx)
        return chosen_cx

    def _detect_branch_from_candidates(self, candidates: list[BlobCandidate], prev_cx: float) -> Optional[int]:
        """Detect branch with alternating left/right direction based on branch_count."""
        if len(candidates) < 2:
            return None

        # Build near list (same filtering as original implementation)
        near = [
            c for c in candidates
            if abs(c.cx - prev_cx) <= self.BRANCH_WINDOW
            and c.width >= self.MIN_BLOB_WIDTH
        ]

        if len(near) < 2:
            return None

        # Determine direction: 0-based count means 1st event (count=0) goes right
        want_right = (self.branch_count % 2 == 0)

        if want_right:
            chosen_cx = max(near, key=lambda c: c.cx).cx  # rightmost
        else:
            chosen_cx = min(near, key=lambda c: c.cx).cx  # leftmost

        # Debug logging for branch direction
        self.get_logger().info(f"Branch #{self.branch_count + 1}: "
                               f"{'right' if want_right else 'left'} chosen (cx={chosen_cx})")

        return chosen_cx

    def _select_blob_fallback(self, context: ScanLineContext, prev_cx: float) -> int:
        """Select fallback blob center when no candidates are available."""
        return context.branch_cx or int(prev_cx)

    def _handle_branch_with_context(
        self,
        idx: int,
        candidates: list[BlobCandidate],
        context: ScanLineContext,
    ) -> tuple[BranchResult, DebugEntry]:
        """Select blob center based on previous per-line center and branch logic."""
        prev = self.prev_cx_per_line[idx] or context.base_cx

        # Try branch detection first if in blue_to_black state
        if context.state == "blue_to_black":
            branch_center = self._detect_branch_from_candidates(
                candidates, prev)
            if branch_center is not None:
                chosen = branch_center
                result_state = "normal"
                branch_cx = chosen
                self.prev_cx_per_line[idx] = chosen
                debug_entry = DebugEntry(
                    context.y, candidates, chosen, result_state)
                return BranchResult(chosen, branch_cx, result_state), debug_entry

        # Normal blob selection or fallback
        if candidates:
            chosen = self._select_blob_center(candidates, prev)
        else:
            chosen = self._select_blob_fallback(context, prev)

        result_state = context.state
        branch_cx = context.branch_cx

        # update per-line previous center
        self.prev_cx_per_line[idx] = chosen

        debug_entry = DebugEntry(context.y, candidates, chosen, result_state)
        return BranchResult(chosen, branch_cx, result_state), debug_entry

    def _compute_velocity_params(self, cx_list: list[WeightedCenter], width):
        """Compute deviation, confidence, and averaged center from cx_list."""
        confidence = len(cx_list) / len(self.scan_lines)
        total_weight = sum(wc.weight for wc in cx_list)
        deviation = sum((wc.cx - width // 2) *
                        wc.weight for wc in cx_list) / total_weight
        averaged_cx = sum(wc.cx * wc.weight for wc in cx_list) / total_weight
        return deviation, confidence, averaged_cx

    def _update_prev_cx(self, branch_cx, averaged_cx):
        """Update prev_cx and reset pending_branch when needed."""
        if branch_cx is None:
            self.prev_cx = averaged_cx
        else:
            if self.pending_branch == 0:
                # Blue event completion: increment global counter
                self.branch_count += 1
                self.get_logger().info(
                    f"Blue event #{self.branch_count} completed")
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

        # Debug logging for velocity control
        self.get_logger().info(
            f"deviation={deviation:.2f}, angular={angular:.4f}, linear={linear:.4f}"
        )

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
        debug_info: list[DebugEntry],
        deviation: float | None = None,
        angular: float | None = None,
        confidence: float | None = None,
        message: str | None = None,
    ) -> None:
        """Draw debug information on the image and display it."""
        debug_image = image.copy()
        width = image.shape[1]

        for debug_entry in debug_info:
            # draw scan line
            cv2.line(debug_image, (0, debug_entry.y),
                     (width - 1, debug_entry.y), (255, 0, 0), 1)
            # draw blob candidates (non-selected)
            for candidate in debug_entry.candidates:
                if candidate.cx == debug_entry.chosen_cx:
                    continue
                cv2.drawMarker(
                    debug_image,
                    (candidate.cx, debug_entry.y),
                    (0, 255, 0),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=8,
                    thickness=1,
                )
            # highlight chosen blob with filled circle
            if debug_entry.chosen_cx is not None:
                cv2.circle(debug_image, (debug_entry.chosen_cx,
                                         debug_entry.y), 4, (0, 255, 0), -1)
            # draw state label
            cv2.putText(
                debug_image,
                _STATE_ABBR.get(debug_entry.state, debug_entry.state),
                (10, debug_entry.y + 15),
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
            # Add linear velocity display
            linear_vel = self.prev_linear
            cv2.putText(
                debug_image,
                f"Linear: {linear_vel:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_image,
                f"Confidence: {confidence:.2f}",
                (10, 90),
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
