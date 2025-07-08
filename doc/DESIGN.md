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
3. Each scan line is also checked for blue pixels using HSV thresholding.
   When enough lines contain blue over consecutive frames, the node enters a
   "blue detected" state.
   It waits until multiple blobs appear on the black line before switching the
   main line and locking this choice for several frames.
4. Calculate the weighted deviation of these center positions from the image center.
5. Convert this deviation into an angular velocity using a proportional gain.
   The linear velocity is scaled using the current angular velocity and
   smoothed by a low-pass filter.
6. Publish the resulting `Twist`.

## Parameters
- `scan_lines`: list of normalized y positions (ratio of image height) used for line detection.
- `weights`: weight value for each scan line.
- `operation_gain`: gain to transform the deviation into an angular velocity.
- `min_linear`: minimum linear velocity.
- `max_linear`: maximum linear velocity.
- `max_angular`: maximum angular velocity.
- `alpha`: coefficient for the low-pass filter used on linear velocity.
- `debug`: enable OpenCV visualization when set to `true`.
- `blue_lower` / `blue_upper`: HSV range used for blue detection.
- `blue_scanlines_required`: number of scan lines that must detect blue in one frame.
- `blue_detect_frames_threshold`: consecutive frame count to confirm a branch.
- `main_line_lock_duration`: frames to lock the selected line after switching.
- `multi_blob_scanlines_required`: scan lines with multiple blobs needed to
  confirm a branch.

The node retrieves the image width and height from each received frame, so it can adapt to different camera resolutions.

