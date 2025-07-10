# Design

The package exposes a single node `NavigatorNode` that converts camera images into velocity commands. It is intended for the ET Robocon robot and assumes a camera mounted at the front of the vehicle.

## Node overview
- **Subscription**: `/camera_top/camera_top/image_raw` (`sensor_msgs/Image`)
- **Publication**: `/cmd_vel` (`geometry_msgs/Twist`)

## Algorithm
1. Convert the input image to grayscale and apply a binary threshold so the dark line appears white.
2. For each predefined scan line, detect connected white pixel blobs.
   Select the blob whose center is closest to the previously detected line
   position (or the image center if none is available).
   When the state machine transitions from `blue_detected` to `blue_to_black`,
   blobs narrower than `MIN_BLOB_WIDTH` (5 px) are ignored and the remaining
   blobs are ranked by distance to the previous center (ties prefer the right
   blob). If a valid blob is chosen, the scan line immediately returns to
   `normal` and the selected center overrides all scan lines for that frame.
   Lines whose detected blob is farther than `BRANCH_CX_TOL` (25 px) adopt this
   branch center. Each scan line tracks a small state machine (`normal`,
   `blue_detected`, `blue_to_black`) to report if a blue area temporarily
   occludes the line.
3. Calculate the weighted deviation of these center positions from the image center.
4. Convert this deviation into an angular velocity using a proportional gain.
   The linear velocity is scaled using the current angular velocity and
   smoothed by a low-pass filter.
5. Publish the resulting `Twist`.

## Parameters
- `scan_lines`: list of normalized y positions (ratio of image height) used for line detection.
- `weights`: weight value for each scan line.
- `operation_gain`: gain to transform the deviation into an angular velocity.
- `min_linear`: minimum linear velocity.
- `max_linear`: maximum linear velocity.
- `max_angular`: maximum angular velocity.
- `alpha`: coefficient for the low-pass filter used on linear velocity.
- `debug`: enable OpenCV visualization when set to `true`.

The node retrieves the image width and height from each received frame, so it can adapt to different camera resolutions.

