# etrobo_navigator

A ROS 2 package that provides a camera based line following node for the ET Robocon robot. The node uses multiple scan lines from the front camera image to compute a navigation command.

## Requirements
- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- `cv_bridge` and OpenCV

For simulation you can use [etrobo_simulator](https://github.com/owhinata/etrobo_simulator).

## Build
```bash
colcon build --symlink-install --packages-select etrobo_navigator
```

## Run
```bash
ros2 run etrobo_navigator etrobo_navigator
```

To enable debug visualization:

```bash
ros2 run etrobo_navigator etrobo_navigator --ros-args -p debug:=true
```

