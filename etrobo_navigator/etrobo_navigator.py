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
        self.max_linear = 0.4                  # Max linear velocity [m/s]
        self.max_angular = 0.6                 # Max angular velocity [rad/s]
        self.alpha = 0.7                       # Low-pass filter coefficient
        self.prev_linear = 0.1                 # Previous linear velocity

        self.bridge = CvBridge()

        # Debug parameter to enable OpenCV visualization
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').value

        # --- Publisher and Subscriber ---
        self.sub = self.create_subscription(
            Image, '/camera_top/camera_top/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("NavigatorNode started. Waiting for camera images...")

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Thresholding to extract dark line (invert: dark becomes white)
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        height, width = binary.shape[:2]

        # Process each scan line and collect debug info
        cx_list = []
        debug_info = []  # (y position, center x or None)
        for ratio, w in zip(self.scan_lines, self.weights):
            y = int(ratio * height)
            row = binary[y, :]
            indices = np.where(row == 255)[0]
            if len(indices) > 0:
                cx = int(np.mean(indices))
                cx_list.append((cx, w))
                debug_info.append((y, cx))
            else:
                debug_info.append((y, None))

        confidence = len(cx_list) / len(self.scan_lines)

        if len(cx_list) == 0:
            if self.debug:
                self._show_debug(
                    cv_image,
                    debug_info,
                    message="No line detected",
                )
            self.get_logger().warn("No line detected.")
            return

        # Compute weighted deviation from image center
        deviation_sum = sum((cx - width // 2)
                            * w for cx, w in cx_list)
        total_weight = sum(w for _, w in cx_list)
        deviation = deviation_sum / total_weight

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
            )

    def _show_debug(
        self,
        image: np.ndarray,
        debug_info,
        deviation: float | None = None,
        angular: float | None = None,
        confidence: float | None = None,
        message: str | None = None,
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
