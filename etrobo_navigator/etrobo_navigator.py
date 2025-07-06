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
        # Y-coordinates of scan lines distributed vertically
        self.scan_lines = [  # closer lines have higher priority
            300, 320, 340, 360, 380, 400
        ]
        # Weights biased toward the lower part of the image (sum to 1.0)
        self.weights = [
            0.1, 0.1, 0.15, 0.2, 0.25, 0.2
        ]
        self.image_width = 640                 # Width of the image
        self.operation_gain = 0.005            # Gain for deviation-to-angular conversion
        # Motor max speed: 185 RPM ±15% -> ~212.8 RPM at no load
        # With 28 mm wheel radius, the theoretical max is about 0.62 m/s
        self.min_linear = 0.1                  # Min linear velocity [m/s]
        self.max_linear = 0.4                  # Max linear velocity [m/s]
        self.max_angular = 0.6                 # Max angular velocity [rad/s]
        self.alpha = 0.7                       # Low-pass filter coefficient
        self.prev_linear = 0.1                 # Previous linear velocity

        self.bridge = CvBridge()

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

        # Process each scan line
        cx_list = []
        for y, w in zip(self.scan_lines, self.weights):
            row = binary[y, :]
            indices = np.where(row == 255)[0]
            if len(indices) > 0:
                cx = int(np.mean(indices))
                cx_list.append((cx, w))

        if len(cx_list) == 0:
            self.get_logger().warn("No line detected.")
            return

        # Compute weighted deviation from image center
        deviation_sum = sum((cx - self.image_width // 2)
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


def main(args=None):
    """Entry point for the runner node."""
    rclpy.init()
    runner = NavigatorNode()
    try:
        rclpy.spin(runner)
    except KeyboardInterrupt:
        pass
    runner.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
