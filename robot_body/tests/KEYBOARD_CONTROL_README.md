# Keyboard Joint Jogging Controller

## Overview

The `keyboard_move.py` script provides real-time keyboard-based control for individual robot joints. This allows you to manually jog each joint using simple keyboard inputs at a constant velocity of 0.2 rad/s.

## Features

- **Real-time keyboard input** using threading for non-blocking key capture
- **Incremental joint position control** at 0.2 rad/s velocity
- **250 Hz control frequency** matching the robot's native control rate
- **Safe shutdown** with multiple safety mechanisms (Ctrl+C, atexit, try-finally)
- **Simple key mapping** for intuitive control

## Key Mapping

### Positive Direction (Increase joint angle)
- `q` - Joint 0 +
- `w` - Joint 1 +
- `e` - Joint 2 +
- `r` - Joint 3 +
- `t` - Joint 4 +
- `y` - Joint 5 +
- `u` - Joint 6 +

### Negative Direction (Decrease joint angle)
- `a` - Joint 0 -
- `s` - Joint 1 -
- `d` - Joint 2 -
- `f` - Joint 3 -
- `g` - Joint 4 -
- `h` - Joint 5 -
- `j` - Joint 6 -

### Exit
- `Ctrl+C` - Stop control and safely shutdown robot

## Usage

```bash
cd /home/hillbot/sudo/betav1_0_pick_place_1/robot_body/tests
python keyboard_move.py
```

The script will:
1. Initialize the CAN bus (using `init_socket_can.sh`)
2. Start logging
3. Initialize the robot
4. Read current joint positions
5. Display control instructions
6. Enter keyboard control mode

## How It Works

### Architecture

1. **KeyboardController Class**
   - Manages keyboard input via a separate thread
   - Updates target positions incrementally based on key presses
   - Sends position commands to robot at 250 Hz

2. **Keyboard Listener Thread**
   - Runs in background (daemon thread)
   - Captures raw keyboard input using `termios` and `tty`
   - Non-blocking to allow smooth robot control

3. **Control Loop**
   - Runs at 250 Hz (4ms per cycle)
   - Checks for new key presses
   - Calculates position increment: `delta = direction × speed × dt`
   - Sends updated position command to robot

### Motion Profile

- **Speed**: 0.2 rad/s (configurable)
- **Control Frequency**: 250 Hz
- **Position Increment per Cycle**: 0.2 × (1/250) = 0.0008 rad per 4ms
- **Motion Type**: Constant velocity while key is held

### Safety Features

1. **Signal Handler**: Catches Ctrl+C interrupt
2. **atexit Registration**: Ensures cleanup on any exit
3. **Try-Finally Block**: Guarantees robot shutdown
4. **Global Control Flag**: Allows safe stopping from any thread
5. **Thread Cleanup**: Properly joins keyboard listener thread

## Example Session

```
Running script: /home/hillbot/sudo/betav1_0_pick_place_1/robot_body/init_socket_can.sh
Script executed successfully.
Logging initialized
new a robot
Service up

Starting keyboard control...

Initializing keyboard control...
Initial positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Keyboard Control Active (Speed: 0.2 rad/s)
============================================================
Controls:
  q/a - Joint 0 (+/-)  |  w/s - Joint 1 (+/-)  |  e/d - Joint 2 (+/-)
  r/f - Joint 3 (+/-)  |  t/g - Joint 4 (+/-)  |  y/h - Joint 5 (+/-)
  u/j - Joint 6 (+/-)

  Ctrl+C - Exit
============================================================
Joint 0: +0.234 rad

[Press Ctrl+C to exit]

You pressed Ctrl+C!

Keyboard interrupt in control loop

Final positions: [0.234, 0.145, -0.089, 0.456, -0.123, 0.678, 0.234]

Cleaning up...
Shutting down robot...
Robot shutdown completed
Program finished
```

## Implementation Details

### Position Update Logic

```python
if key in self.key_map:
    joint_idx, direction = self.key_map[key]
    delta = direction * self.speed * self.dt
    self.target_positions[joint_idx] += delta
```

- Each key press increments/decrements the target position
- The increment is: `±0.2 rad/s × 0.004s = ±0.0008 rad`
- Holding a key continuously updates the position

### Threading Model

```python
# Main thread: Control loop at 250 Hz
while control_running:
    if self.key_pressed is not None:
        # Process key
        # Update target position
    
    # Send position command
    robot.set_actions(...)
    time.sleep(self.dt)

# Background thread: Keyboard listener
while self.reading_keyboard:
    key = self.get_key()
    self.key_pressed = key
```

### Terminal Settings

The controller uses raw terminal mode to capture individual keypresses:

```python
tty.setraw(sys.stdin.fileno())  # Disable line buffering
ch = sys.stdin.read(1)           # Read single character
termios.tcsetattr(...)           # Restore settings
```

## Advantages

1. **Smooth Motion**: High control frequency (250 Hz) provides smooth joint movement
2. **Responsive**: Separate keyboard thread ensures no input lag
3. **Safe**: Multiple layers of safety ensure robot always shuts down
4. **Simple**: Intuitive key mapping for easy use
5. **Real-time**: Immediate response to key presses

## Limitations

1. **Single Key at a Time**: Only one joint can move at a time
2. **Constant Velocity**: No acceleration/deceleration profiles
3. **No Position Limits**: User must be aware of joint limits
4. **Terminal-Based**: Requires direct terminal access (no remote via some SSH clients)

## Comparison to Trajectory Control

| Feature | Keyboard Control | Trajectory Control |
|---------|------------------|-------------------|
| Planning | None (real-time) | Pre-computed |
| Motion | Constant velocity | Smooth (quintic) |
| Joints | One at a time | All simultaneously |
| Use Case | Manual teaching | Automated tasks |
| Precision | User-dependent | Repeatable |

## Future Enhancements

- Multi-key support for simultaneous joint control
- Adjustable speed (e.g., Shift for faster, Alt for slower)
- Position display/monitoring in real-time
- Joint limit checking and warnings
- Save/load positions for teaching
- Cartesian space control (XYZ movement)
