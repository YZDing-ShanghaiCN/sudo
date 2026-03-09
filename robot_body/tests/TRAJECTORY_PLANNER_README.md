# Joint Trajectory Planner and Controller

## Overview

This document describes the trajectory planning and control system implemented in `test_move_right_arm.py`.

## Components

### 1. JointTrajectory (Dataclass)

A dataclass that stores a complete joint trajectory with time and position sequences.

**Properties:**
- `timestamps`: Array of time points [t0, t1, t2, ...]
- `positions`: Array of joint positions at each time point [n_points x n_joints]
- `velocities`: Array of joint velocities at each time point [n_points x n_joints]
- `duration`: Total duration of the trajectory
- `n_points`: Number of waypoints in the trajectory
- `n_joints`: Number of joints

**Example:**
```python
trajectory = JointTrajectory(
    timestamps=np.array([0.0, 0.004, 0.008, ...]),
    positions=np.array([[0.0, 0.0, ...], [0.01, 0.01, ...], ...]),
    velocities=np.array([[0.0, 0.0, ...], [0.5, 0.5, ...], ...])
)
```

### 2. JointPlanner

Generates smooth joint trajectories from start to target positions using quintic (5th order) polynomial interpolation.

**Features:**
- Smooth position, velocity, and acceleration profiles
- Zero velocity and acceleration at start and end points
- Configurable control frequency (default: 250 Hz to match robot)
- Time-parameterized trajectory generation

**Method:**
```python
plan_trajectory(start_positions, target_positions, duration) -> JointTrajectory
```

**Parameters:**
- `start_positions`: Starting joint positions [rad]
- `target_positions`: Target joint positions [rad]
- `duration`: Time to complete the motion [seconds]

**Returns:** `JointTrajectory` object

**Example:**
```python
planner = JointPlanner(control_frequency=250.0)
trajectory = planner.plan_trajectory(
    start_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    target_positions=[-2.2, 0.77, 2.27, 1.66, -0.20, 0.57, 2.23],
    duration=5.0
)
```

### 3. TrajectoryController

Executes a planned trajectory on the robot hardware.

**Features:**
- Real-time trajectory execution at ~250 Hz
- Progress monitoring
- Safe interruption handling
- Automatic waypoint interpolation

**Method:**
```python
execute_trajectory(trajectory, monitor=True) -> bool
```

**Parameters:**
- `trajectory`: JointTrajectory to execute
- `monitor`: If True, print progress during execution

**Returns:** `True` if completed successfully, `False` if interrupted

**Example:**
```python
controller = TrajectoryController(robot, hardware_name="right_arm")
success = controller.execute_trajectory(trajectory, monitor=True)
```

### 4. Safety Features

**Automatic Shutdown:**
- Signal handler for Ctrl+C
- `atexit` registration ensures cleanup on any exit
- Try-finally block in main ensures shutdown
- Global `control_running` flag for safe interruption

**Shutdown Sequence:**
1. Ctrl+C pressed or program ends
2. `control_running` set to False
3. Trajectory execution stops
4. `robot.shutdown()` called
5. Robot safely powered down

## Usage Example

```python
# Initialize robot
robot = RobotWrapper()

# Get current positions
states = robot._robot.get_states(["right_arm"])
current_positions = states["right_arm"]["position"]

# Define target
target_positions = [-2.2, 0.77, 2.27, 1.66, -0.20, 0.57, 2.23]

# Create planner and controller
planner = JointPlanner(control_frequency=250.0)
controller = TrajectoryController(robot, hardware_name="right_arm")

# Plan trajectory
trajectory = planner.plan_trajectory(
    start_positions=current_positions,
    target_positions=target_positions,
    duration=5.0
)

# Execute
success = controller.execute_trajectory(trajectory, monitor=True)

# Always shutdown
robot.shutdown()
```

## Mathematical Details

### Quintic Polynomial

The trajectory for each joint uses a quintic (5th order) polynomial:

```
q(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵
```

With boundary conditions:
- q(0) = q_start, q(T) = q_end
- q'(0) = 0, q'(T) = 0 (zero velocity)
- q''(0) = 0, q''(T) = 0 (zero acceleration)

Coefficients:
```
a₀ = q_start
a₁ = 0
a₂ = 0
a₃ = 10(q_end - q_start) / T³
a₄ = -15(q_end - q_start) / T⁴
a₅ = 6(q_end - q_start) / T⁵
```

This ensures smooth, continuous motion with no jerky transitions.

## Control Frequency

The system operates at 250 Hz (4ms per cycle), matching the robot's native control frequency for optimal performance.

## Improvements Over Original Code

1. **Structured trajectory planning** instead of single position commands
2. **Smooth motion** with continuous velocity profiles
3. **Time-parameterized control** for predictable behavior
4. **Progress monitoring** for debugging and visualization
5. **Guaranteed shutdown** with multiple safety mechanisms
6. **Dataclass-based architecture** for clean code organization
7. **Reusable components** for other robot control tasks

## Future Enhancements

- Add velocity and acceleration limits checking
- Support for via-points (multi-segment trajectories)
- Cartesian space planning (not just joint space)
- Collision checking
- Real-time trajectory modification
- Trajectory recording and playback
