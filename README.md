# 🤖 MCL + A\* Autonomous Navigation — PuzzleBot

> Monte Carlo Localization with EKF backup and A\* path planning for a differential drive robot on ROS 2.

---

## Overview

Autonomous navigation system for the **PuzzleBot** (differential drive robot) combining:

- **MCL (Monte Carlo Localization)** with an adaptive particle filter (KLD-sampling) for pose estimation on a pre-built map.
- **EKF backup** that maintains position estimation when LiDAR confidence drops (unmapped obstacles, feature-sparse zones).
- **MCL ↔ EKF state machine** with smooth transitions and a synchronization mode.
- **Adaptive modules**: RAM (Robust Adaptive Motion Model) for motion noise and a Bandit (UCB) sensor model configuration selector.
- **A\* planner** on an inflated map with reactive LiDAR-based obstacle avoidance.

---

## Architecture

```
/odom  ──────────────────────────────────────────────────────┐
                                                              ▼
                                                    ┌─────────────────┐
                                                    │  odom_callback  │
                                                    │  RAM motion     │
                                                    │  EKF predict    │
                                                    └────────┬────────┘
                                                             │ propagated particles
/scan  ──────────────────────────────────────────────────────┤
                                                             ▼
                                                   ┌──────────────────┐
                                                   │  scan_callback   │
                                                   │  Ray casting     │
                                                   │  Beam model      │
                                                   │  Bandit selector │
                                                   │  FSM MCL↔EKF     │
                                                   └────────┬─────────┘
                                                            │
                                               /mcl_pose ───┘
                                                            │
/goal_pose ──────────────────────────────────────────────── ▼
                                                  ┌──────────────────┐
                                                  │   astar_node     │
                                                  │   A* + inflation │
                                                  │   P controller   │
                                                  │   LiDAR avoidance│
                                                  └────────┬─────────┘
                                                           │
                                               /cmd_vel ───┘
```

---

## Nodes

### `mcl_node.py`

Robot pose estimation via particle filter with EKF backup.

| Subscribes | Type | Description |
|---|---|---|
| `/odom` | `nav_msgs/Odometry` | Odometry for particle propagation and EKF prediction |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR measurements for weight update |

| Publishes | Type | Description |
|---|---|---|
| `/mcl_pose` | `geometry_msgs/PoseStamped` | Estimated pose (MCL, EKF, or fused) |
| `/map` | `nav_msgs/OccupancyGrid` | Map loaded from `.npy` |

**State machine:**

| State | Entry condition | Pose source |
|---|---|---|
| `MCL` | Convergence > `CONV_HIGH` | Particle median |
| `EKF` | Convergence < `CONV_LOW` for ≥ N scans | EKF prediction (odometry only) |
| `SYNC` | MCL recovering | Weighted fusion MCL + EKF |

---

### `astar_node.py`

Path planning and motion control.

| Subscribes | Type | Description |
|---|---|---|
| `/mcl_pose` | `geometry_msgs/PoseStamped` | Current robot pose |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Goal published from RViz |
| `/scan` | `sensor_msgs/LaserScan` | Reactive obstacle avoidance |

| Publishes | Type | Description |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands |
| `/planned_path` | `nav_msgs/Path` | Full path for RViz |
| `/current_waypoint` | `geometry_msgs/PoseStamped` | Active waypoint |

---

## Key Parameters

### Map

| Parameter | Value | Description |
|---|---|---|
| `MAP_X_MIN/MAX` | -4.0 / 5.0 m | Map extent in X |
| `MAP_Y_MIN/MAX` | -3.0 / 6.0 m | Map extent in Y |
| `RESOLUTION` | 0.05 m/px | Map resolution |

### MCL

| Parameter | Value | Description |
|---|---|---|
| `N` | 200 | Maximum number of particles |
| `N_MIN` | 30 | Minimum with KLD-sampling |
| `P_NOISE` | 0.8 | Gaussian model weight in beam model |
| `CONV_LOW` | 0.25 | Low convergence threshold → failover to EKF |
| `CONV_HIGH` | 0.55 | High convergence threshold → MCL regains control |
| `SCANS_TO_FAILOVER` | 5 | Consecutive low-confidence scans before EKF takeover |

### A\*

| Parameter | Value | Description |
|---|---|---|
| `INFLATE_RADIUS` | 6 px (0.3 m) | Obstacle inflation radius |
| `WAYPOINT_TOL` | 0.25 m | Tolerance to advance to next waypoint |
| `GOAL_TOL` | 0.35 m | Tolerance to declare goal reached |
| `LIN_VEL` | 0.07 m/s | Nominal linear velocity |
| `ANG_VEL` | 0.07 rad/s | Maximum angular velocity |

---

## Requirements

- ROS 2 Humble / Jazzy
- Python ≥ 3.10
- `numpy`, `opencv-python`, `scipy`
- RPLIDAR A1 (or Gazebo simulation)
- Pre-built map in `.npy` format (output from SLAM node)

---

## Usage

### 1. Place the map

```bash
# Map must be located at:
ros2_ws/src/mcl_robot/maps/slam_map.npy
```

### 2. Launch the nodes

#### 🖥️ Simulation (Gazebo)

```bash
# Connect to Puzzlebot
ssh puzzlebot@xxx.xxx.xxx.xxx

# Set environment variables
export ROS_DOMAIN_ID=0
export ROS_IP=192.168.137.xxx

# Terminal 1 - Launch micro-ROS agent
ros2 launch puzzlebot_ros micro_ros_agent.launch.py

# Terminal 2 - Launch Lidar
ros2 run ros2_lidar_ws ...

# Terminal 3 - Teleop for manual control
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Terminal 4 - PuzzleBot simulation
ros2 launch puzzlebot_gazebo puzzle_sim_pb.launch.py

# Terminal 5 — MCL + EKF localization
ros2 run mcl_robot mcl_node

# Terminal 6 — A* planner + control
ros2 run mcl_robot astar_node
```

#### 🤖 Real robot

```bash
# Terminal 1 — Odometry
ros2 run mcl_robot odom_node

# Terminal 2 — MCL + EKF localization
ros2 run mcl_robot mcl_node

# Terminal 3 — A* planner + control
ros2 run mcl_robot astar_node
```

### 3. Visualize in RViz

Add the following topics:
- `/map` → OccupancyGrid
- `/mcl_pose` → PoseStamped
- `/planned_path` → Path
- `/current_waypoint` → PoseStamped

Use **2D Goal Pose** to send a navigation goal.

---

## Internal Modules

### EKFBackup

3-state EKF `[x, y, θ]` with a non-linear differential drive kinematic model. Analytical Jacobian. Process noise proportional to motion (`Q_trans`, `Q_rot`). Adaptive observation noise scaled by MCL convergence.

### RobustAdaptiveMotionModel (RAM)

Adapts the motion noise covariance online using a Robbins-Monro scheme. Target acceptance rate: 35%.

### BanditSensorSelector

Selects the optimal beam model configuration (σ_hit, ray skip) using variance-adjusted UCB. Automatically balances speed and precision of the sensor model.

---

## Notes

- In environments with dynamic obstacles or feature-sparse zones, the EKF preserves the relative trajectory while MCL recovers.
- Particles are redistributed around the EKF estimate during the `EKF` state, speeding up reconvergence.
- Vectorized ray casting operates in pixel space for maximum NumPy performance.

---

## Author

**Felipe Garcia** — Robotics & Digital Systems Engineering, Tecnológico de Monterrey  
Developed as part of course *TE3003B — Robotics Integration*.
