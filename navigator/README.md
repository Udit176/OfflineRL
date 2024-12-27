# AutonomousNavigator

AutonomousNavigator is a ROS2 node that integrates CSV-based scenarios with A* navigation and path following for robots. It processes scenarios from a CSV file, plans optimal paths using A*, follows predefined paths, and records trajectory data.

## Features
- **CSV Scenario Execution:** Reads and executes successful scenarios from a CSV file.
- **A* Path Planning:** Computes optimal paths between start and goal positions.
- **Path Following:** Follows predefined paths from the CSV.
- **Trajectory Recording:** Records positions, distances, and durations.
- **Map & IMU Integration:** Handles map updates and integrates IMU data for accurate pose estimation.

## Usage

1. **Prerequisites:**
   - ROS2 installed.
   - Python 3 environment.
   - Required ROS2 packages (`nav_msgs`, `geometry_msgs`, `sensor_msgs`).

2. **Setup:**
   - Place your CSV file at the path specified in the script (`/path/to/rl_csv_data.csv`).

3. **Build and Run the Node:**
   ```bash
   colcon build --packages-select package_name
   ros2 run package_name navigator_csv
   ```
