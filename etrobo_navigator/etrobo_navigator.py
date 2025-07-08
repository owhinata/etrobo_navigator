import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class NavigatorNode(Node):
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

        # Debug parameter to enable OpenCV visualization
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').value

        # Parameters for blue line detection and switching
        self.declare_parameter('blue_lower', [100, 100, 50])
        self.declare_parameter('blue_upper', [130, 255, 255])
        self.declare_parameter('blue_scanlines_required', 2)
        self.declare_parameter('blue_detect_frames_threshold', 3)
        self.declare_parameter('main_line_lock_duration', 10)
        self.declare_parameter('multi_blob_scanlines_required', 2)
        self.blue_lower = tuple(
            int(v) for v in self.get_parameter('blue_lower').value
        )
        self.blue_upper = tuple(
            int(v) for v in self.get_parameter('blue_upper').value
        )
        self.blue_scanlines_required = self.get_parameter(
            'blue_scanlines_required'
        ).value
        self.blue_detect_frames_threshold = self.get_parameter(
            'blue_detect_frames_threshold'
        ).value
        self.main_line_lock_duration = self.get_parameter(
            'main_line_lock_duration'
        ).value
        self.multi_blob_scanlines_required = self.get_parameter(
            'multi_blob_scanlines_required'
        ).value

        # State variables for branching logic
        self.main_line_side = 'left'
        self.blue_detect_counter = 0
        self.lock_counter = 0
        self.branching_state = 'idle'

        # --- Publisher and Subscriber ---
        self.sub = self.create_subscription(
            Image, '/camera_top/camera_top/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("NavigatorNode started. Waiting for camera images...")

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Thresholding to extract dark line (invert: dark becomes white)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        height, width = binary.shape[:2]

        # Process each scan line and collect debug info
        cx_list = []
        debug_info = []  # (y position, center x or None)
        target_cx = self.prev_cx if self.prev_cx is not None else width // 2
        blue_scan_count = 0
        multi_blob_scan_count = 0

        for ratio, w in zip(self.scan_lines, self.weights):
            y = int(ratio * height)
            row = binary[y, :]
            hsv_row = hsv[y:y + 1, :]
            if np.any(cv2.inRange(hsv_row, self.blue_lower, self.blue_upper)):
                blue_scan_count += 1

            indices = np.where(row == 255)[0]
            if len(indices) > 0:
                # Split indices into connected components
                splits = np.where(np.diff(indices) > 1)[0] + 1
                blobs = np.split(indices, splits)

                if len(blobs) > 1:
                    multi_blob_scan_count += 1

                # Find center of each blob
                candidates = []
                for blob in blobs:
                    cx = int(np.mean(blob))
                    width_blob = blob[-1] - blob[0] + 1
                    candidates.append((cx, width_blob))

                if self.main_line_side == 'left':
                    chosen_cx, _ = min(candidates, key=lambda c: c[0])
                elif self.main_line_side == 'right':
                    chosen_cx, _ = max(candidates, key=lambda c: c[0])
                else:
                    chosen_cx, _ = min(
                        candidates, key=lambda c: abs(c[0] - target_cx)
                    )
                cx_list.append((chosen_cx, w))
                debug_info.append((y, chosen_cx))
            else:
                debug_info.append((y, None))

        confidence = len(cx_list) / len(self.scan_lines)

        if blue_scan_count >= self.blue_scanlines_required:
            self.blue_detect_counter += 1
        else:
            self.blue_detect_counter = 0

        if self.lock_counter > 0:
            self.lock_counter -= 1

        if (
            self.branching_state == 'idle'
            and self.blue_detect_counter >= self.blue_detect_frames_threshold
            and self.lock_counter == 0
        ):
            self.branching_state = 'blue_detected_pending'
            self.get_logger().info('Blue detected, waiting for branch')
            self.blue_detect_counter = 0
        elif self.branching_state == 'blue_detected_pending':
            if multi_blob_scan_count >= self.multi_blob_scanlines_required:
                self.main_line_side = (
                    'right' if self.main_line_side == 'left' else 'left'
                )
                self.lock_counter = self.main_line_lock_duration
                self.branching_state = 'idle'
                self.get_logger().info(
                    f'Branch detected, switched main line to {self.main_line_side}'
                )

        extra_text = [
            f"Blue lines: {blue_scan_count} ({self.blue_detect_counter})",
            f"State: {self.branching_state}",
            f"Main line: {self.main_line_side}, lock: {self.lock_counter}",
        ]

        if len(cx_list) == 0:
            if self.debug:
                self._show_debug(
                    cv_image,
                    debug_info,
                    message="No line detected",
                    extra_text=extra_text,
                )
            self.get_logger().warn("No line detected.")
            return

        # Compute weighted deviation from image center
        deviation_sum = sum((cx - width // 2)
                            * w for cx, w in cx_list)
        total_weight = sum(w for _, w in cx_list)
        deviation = deviation_sum / total_weight

        # Update the stored center using a weighted average
        self.prev_cx = sum(cx * w for cx, w in cx_list) / total_weight

        # Compute angular velocity using proportional control
        angular = np.clip(
            -deviation * self.operation_gain, -self.max_angular, self.max_angular
        )

        # Determine linear velocity based on angular velocity
        norm_ang = min(abs(angular) / self.max_angular, 1.0)
        new_linear = self.max_linear - (
            self.max_linear - self.min_linear
        ) * norm_ang

        # Apply low-pass filter for smoother speed changes
        linear = self.alpha * self.prev_linear + (1.0 - self.alpha) * new_linear
        self.prev_linear = linear

        # Create and publish Twist message
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.pub.publish(cmd)

        self.get_logger().info(
            f"cmd_vel: linear={linear:.4f}, angular={angular:.4f}")

        if self.debug:
            self._show_debug(
                cv_image,
                debug_info,
                deviation=deviation,
                angular=angular,
                confidence=confidence,
                extra_text=extra_text,
            )

    def _show_debug(
        self,
        image: np.ndarray,
        debug_info,
        deviation: float | None = None,
        angular: float | None = None,
        confidence: float | None = None,
        message: str | None = None,
        extra_text: list[str] | None = None,
    ) -> None:
        """Draw debug information on the image and display it."""
        debug_image = image.copy()
        width = image.shape[1]

        for y, cx in debug_info:
            cv2.line(debug_image, (0, y), (width - 1, y), (255, 0, 0), 1)
            if cx is not None:
                cv2.circle(debug_image, (cx, y), 4, (0, 255, 0), -1)
            else:
                cv2.line(
                    debug_image,
                    (width // 2 - 5, y - 5),
                    (width // 2 + 5, y + 5),
                    (0, 0, 255),
                    1,
                )
                cv2.line(
                    debug_image,
                    (width // 2 - 5, y + 5),
                    (width // 2 + 5, y - 5),
                    (0, 0, 255),
                    1,
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
            if extra_text:
                base_y = 90
                for line in extra_text:
                    cv2.putText(
                        debug_image,
                        line,
                        (10, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    base_y += 20

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
