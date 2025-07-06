# Repository Guidelines

This repository provides a scan line follower node for **ET Robocon**.

## Environment
- Ubuntu 22.04  
- ROS 2 Humble  
- Gazebo 11.10.2  
- Python 3.10  

For simulation:  
- [etrobo_simulator](https://github.com/owhinata/etrobo_simulator)

## Development
- Language: Python  
- Follow the PEP 8 coding style.  
- When adding new features, make only the minimum necessary changes.  
  Refactoring (e.g., variable renaming or other code improvements) is **not** allowed unless explicitly requested.
- Include a “Requirement” section in the pull-request description.

## Documentation
- Write all documentation, code comments, and branch names in **English**.  
- If the design policy changes, **update `doc/DESIGN.md` as well**.

## Testing
Run the following after making changes:

```bash
colcon test --packages-select etrobo_simulator

