# Design

The package exposes a single node `NavigatorNode` that converts camera images into velocity commands. It is intended for the ET Robocon robot and assumes a camera mounted at the front of the vehicle.

## Node overview
- **Subscription**: `/camera_top/camera_top/image_raw` (`sensor_msgs/Image`)
- **Publication**: `/cmd_vel` (`geometry_msgs/Twist`)

## Algorithm
1. Convert the input image to grayscale and apply a binary threshold so the dark line appears white.
2. For each predefined scan line in the image, find the white pixels and compute their average x position.
3. Calculate the weighted deviation of these center positions from the image center.
4. Convert this deviation into an angular velocity using a proportional gain.
   The linear velocity is scaled using the current angular velocity and
   smoothed by a low-pass filter.
5. Publish the resulting `Twist`.

## Parameters
- `scan_lines`: list of y coordinates used for line detection.
- `weights`: weight value for each scan line.
- `image_width`: width of the camera image.
- `operation_gain`: gain to transform the deviation into an angular velocity.
- `min_linear`: minimum linear velocity.
- `max_linear`: maximum linear velocity.
- `max_angular`: maximum angular velocity.
- `alpha`: coefficient for the low-pass filter used on linear velocity.

