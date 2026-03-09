import rb_python
import time
import signal
import hblog
import subprocess
import os
import sys
import termios
import tty
import threading
import atexit


class KeyboardController:
    """Controller for keyboard-based joint jogging"""
    
    def __init__(self, robot_wrapper, hardware_name: str = "right_arm", gripper_name: str = "left_gripper", 
                 rail_name: str = "rail", waist_name: str = "waist",
                 speed: float = 0.2, gripper_speed: float = 0.05, 
                 rail_speed: float = 0.1, waist_speed: float = 0.1):
        """
        Initialize keyboard controller
        
        Args:
            robot_wrapper: RobotWrapper instance
            hardware_name: Name of the hardware to control
            gripper_name: Name of the gripper to control
            rail_name: Name of the rail hardware to control
            waist_name: Name of the waist hardware to control
            speed: Joint velocity in rad/s (default 0.2)
            gripper_speed: Gripper velocity in rad/s (default 0.05)
            rail_speed: Rail velocity in m/s (default 0.1)
            waist_speed: Waist velocity in rad/s (default 0.1)
        """
        self.robot_wrapper = robot_wrapper
        self.hardware_name = hardware_name
        self.gripper_name = gripper_name
        self.rail_name = rail_name
        self.waist_name = waist_name
        self.speed = speed  # rad/s
        self.gripper_speed = gripper_speed  # rad/s
        self.rail_speed = rail_speed  # m/s
        self.waist_speed = waist_speed  # rad/s
        self.control_freq = 250.0  # Hz
        self.dt = 1.0 / self.control_freq
        
        # Current target positions (will be updated incrementally)
        self.target_positions = None
        self.gripper_position = None
        self.rail_position = None
        self.waist_position = None
        
        # Key mapping: key -> (joint_index, direction)
        # q,w,e,r,t,y,u -> joints 0-6 positive direction
        # a,s,d,f,g,h,j -> joints 0-6 negative direction
        self.key_map = {
            'q': (0, 1),   # Joint 0 +
            'a': (0, -1),  # Joint 0 -
            'w': (1, 1),   # Joint 1 +
            's': (1, -1),  # Joint 1 -
            'e': (2, 1),   # Joint 2 +
            'd': (2, -1),  # Joint 2 -
            'r': (3, 1),   # Joint 3 +
            'f': (3, -1),  # Joint 3 -
            't': (4, 1),   # Joint 4 +
            'g': (4, -1),  # Joint 4 -
            'y': (5, 1),   # Joint 5 +
            'h': (5, -1),  # Joint 5 -
            'u': (6, 1),   # Joint 6 +
            'j': (6, -1),  # Joint 6 -
        }
        
        # Gripper control keys
        self.gripper_keys = {
            '[': 1,   # Open gripper (positive direction)
            ']': -1,  # Close gripper (negative direction)
        }
        
        # Rail and waist control keys (arrow keys)
        # Arrow keys come as escape sequences: \x1b[A (up), \x1b[B (down), \x1b[C (right), \x1b[D (left)
        self.rail_waist_keys = {
            '\x1b[C': ('rail', 1),    # Right arrow - rail forward
            '\x1b[D': ('rail', -1),   # Left arrow - rail backward
            '\x1b[A': ('waist', 1),   # Up arrow - waist positive
            '\x1b[B': ('waist', -1),  # Down arrow - waist negative
        }
        
        # Keyboard input thread
        self.key_pressed = None
        self.keyboard_thread = None
        self.reading_keyboard = False
    
    def get_key(self):
        """Get a single keypress from terminal (non-blocking after first call)"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            # Check for escape sequences (arrow keys)
            if ch == '\x1b':
                ch += sys.stdin.read(2)  # Read the next 2 characters for arrow keys
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def keyboard_listener(self):
        """Thread function to continuously listen for keyboard input"""
        global control_running
        while self.reading_keyboard and control_running:
            try:
                key = self.get_key()
                if key == '\x03':  # Ctrl+C
                    control_running = False
                    break
                self.key_pressed = key
            except:
                break
    
    def start_keyboard_listener(self):
        """Start the keyboard listening thread"""
        self.reading_keyboard = True
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
    
    def stop_keyboard_listener(self):
        """Stop the keyboard listening thread"""
        self.reading_keyboard = False
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=1.0)
    
    def run(self):
        """Main control loop for keyboard jogging"""
        # Get initial positions
        print("\nInitializing keyboard control...")
        time.sleep(0.1)
        
        # Wait for valid state data for arm
        max_attempts = 20
        for attempt in range(max_attempts):
            states = self.robot_wrapper._robot.get_states([self.hardware_name])
            current_state = states.get(self.hardware_name, None)
            if current_state is not None:
                current_positions = current_state.get("position", None)
                if current_positions is not None:
                    self.target_positions = list(current_positions)
                    break
            print(f"Waiting for valid arm state data... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(0.1)
        
        if self.target_positions is None:
            print("Error: Could not read initial arm positions after multiple attempts!")
            return
        
        # Initialize gripper position
        for attempt in range(max_attempts):
            gripper_states = self.robot_wrapper._robot.get_states([self.gripper_name])
            gripper_state = gripper_states.get(self.gripper_name, None)
            if gripper_state is not None:
                gripper_pos = gripper_state.get("position", None)
                if gripper_pos is not None:
                    self.gripper_position = list(gripper_pos)
                    break
            print(f"Waiting for valid gripper state data... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(0.1)
        
        if self.gripper_position is None:
            print("Warning: Could not read gripper position. Gripper control disabled.")
        
        # Initialize rail position
        for attempt in range(max_attempts):
            rail_states = self.robot_wrapper._robot.get_states([self.rail_name])
            rail_state = rail_states.get(self.rail_name, None)
            if rail_state is not None:
                rail_pos = rail_state.get("position", None)
                if rail_pos is not None:
                    self.rail_position = list(rail_pos)
                    break
            print(f"Waiting for valid rail state data... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(0.1)
        
        if self.rail_position is None:
            print("Warning: Could not read rail position. Rail control disabled.")
        
        # Initialize waist position
        for attempt in range(max_attempts):
            waist_states = self.robot_wrapper._robot.get_states([self.waist_name])
            waist_state = waist_states.get(self.waist_name, None)
            if waist_state is not None:
                waist_pos = waist_state.get("position", None)
                if waist_pos is not None:
                    self.waist_position = list(waist_pos)
                    break
            print(f"Waiting for valid waist state data... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(0.1)
        
        if self.waist_position is None:
            print("Warning: Could not read waist position. Waist control disabled.")
        
        print(f"Initial arm positions: {self.target_positions}")
        if self.gripper_position is not None:
            print(f"Initial gripper position: {self.gripper_position}")
        if self.rail_position is not None:
            print(f"Initial rail position: {self.rail_position}")
        if self.waist_position is not None:
            print(f"Initial waist position: {self.waist_position}")
        print(f"\nKeyboard Control Active (Speed: {self.speed} rad/s, Gripper: {self.gripper_speed} rad/s)")
        print(f"Rail Speed: {self.rail_speed} m/s, Waist Speed: {self.waist_speed} rad/s")
        print("=" * 60)
        print("Controls:")
        print("  q/a - Joint 0 (+/-)  |  w/s - Joint 1 (+/-)  |  e/d - Joint 2 (+/-)")
        print("  r/f - Joint 3 (+/-)  |  t/g - Joint 4 (+/-)  |  y/h - Joint 5 (+/-)")
        print("  u/j - Joint 6 (+/-)")
        if self.gripper_position is not None:
            print("  [ - Open Gripper    |  ] - Close Gripper")
        print("\n  Arrow Keys:")
        if self.rail_position is not None:
            print("  → Right Arrow - Rail Forward   |  ← Left Arrow - Rail Backward")
        if self.waist_position is not None:
            print("  ↑ Up Arrow - Waist Up          |  ↓ Down Arrow - Waist Down")
        print("\n  Ctrl+C - Exit")
        print("=" * 60)
        
        # Start keyboard listener thread
        self.start_keyboard_listener()
        
        # Control loop
        loop_count = 0
        try:
            while control_running:
                # Check for key press
                if self.key_pressed is not None:
                    key = self.key_pressed
                    self.key_pressed = None  # Clear the key
                    
                    # Handle arm joint control
                    if key in self.key_map:
                        joint_idx, direction = self.key_map[key]
                        # Calculate position increment based on speed and dt
                        delta = direction * self.speed * self.dt
                        self.target_positions[joint_idx] += delta
                        
                        # Print status every 50 iterations (~0.2 seconds)
                        if loop_count % 50 == 0:
                            print(f"Joint {joint_idx}: {self.target_positions[joint_idx]:+.3f} rad", end='\r')
                    
                    # Handle gripper control
                    elif key in self.gripper_keys and self.gripper_position is not None:
                        direction = self.gripper_keys[key]
                        # Calculate position increment for gripper
                        delta = direction * self.gripper_speed * self.dt
                        self.gripper_position[0] += delta
                        
                        # Print gripper status
                        if loop_count % 50 == 0:
                            print(f"Gripper: {self.gripper_position[0]:+.3f} rad", end='\r')
                    
                    # Handle rail and waist control (arrow keys)
                    elif key in self.rail_waist_keys:
                        hardware, direction = self.rail_waist_keys[key]
                        
                        if hardware == 'rail' and self.rail_position is not None:
                            # Calculate position increment for rail (in meters)
                            delta = direction * self.rail_speed * self.dt
                            self.rail_position[0] += delta
                            
                            # Print rail status
                            if loop_count % 50 == 0:
                                print(f"Rail: {self.rail_position[0]:+.3f} m", end='\r')
                        
                        elif hardware == 'waist' and self.waist_position is not None:
                            # Calculate position increment for waist (in radians)
                            delta = direction * self.waist_speed * self.dt
                            self.waist_position[0] += delta
                            
                            # Print waist status
                            if loop_count % 50 == 0:
                                print(f"Waist: {self.waist_position[0]:+.3f} rad", end='\r')
                
                # Send position command for arm
                self.robot_wrapper._robot.set_actions({
                    self.hardware_name: {
                        "type": "position",
                        "position": self.target_positions
                    }
                })
                # print("position: {}".format(self.target_positions))
                # Send position command for gripper if available
                if self.gripper_position is not None:
                    self.robot_wrapper._robot.set_actions({
                        self.gripper_name: {
                            "type": "position",
                            "position": self.gripper_position
                        }
                    })
                
                # Send position command for rail if available
                if self.rail_position is not None:
                    self.robot_wrapper._robot.set_actions({
                        self.rail_name: {
                            "type": "position",
                            "position": self.rail_position
                        }
                    })
                
                # Send position command for waist if available
                if self.waist_position is not None:
                    self.robot_wrapper._robot.set_actions({
                        self.waist_name: {
                            "type": "position",
                            "position": self.waist_position
                        }
                    })
                
                # Sleep to maintain control frequency
                time.sleep(self.dt)
                loop_count += 1
                
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt in control loop")
        finally:
            self.stop_keyboard_listener()
            print(f"\n\nFinal arm positions: {self.target_positions}")
            if self.gripper_position is not None:
                print(f"Final gripper position: {self.gripper_position}")
            if self.rail_position is not None:
                print(f"Final rail position: {self.rail_position}")
            if self.waist_position is not None:
                print(f"Final waist position: {self.waist_position}")



class RobotWrapper:
    def __init__(self):
        print(f"new a robot")
        cfg_str = {
            "hardware": {  # 硬件列表，想要启动多少个硬件，从此处配置，以下为模板配置项
                "right_arm": { # Found, can't reset
                    "type": "eyou",
                    "ids": [20, 21, 22, 23, 24, 25, 26],
                    "length_per_radian": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "invert_directions": [True, False, True, False, True, False, True],
                    "control_freq": 250,
                    "interpolation_points": 13,
                    "max_velocity": 3.0,
                    "gravity_compensation_tolerance": 0.0,
                    "friction_compensation_scale": 0.0,
                    "friction_compensation_stiffness": 10,
                    "external_protections": [],
                    "offset_at_hardware_zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "joint_names": [
                        "right-joint_arm_1",
                        "right-joint_arm_2",
                        "right-joint_arm_3",
                        "right-joint_arm_4",
                        "right-joint_arm_5",
                        "right-joint_arm_6",
                        "right-joint_arm_7",
                    ],
                    "max_torque": [1800, 1800, 2400, 2400, 2000, 2000, 2000],
                    "protection_rebound": 0.0,
                    "admittance_config": {
                        "param_mass": [1.8, 1.8, 1.8, 0.03, 0.03, 0.03],
                        "param_stiff": [180.0, 180.0, 180.0, 3.0, 3.0, 3.0],
                        "param_damp": [18.0, 18.0, 18.0, 0.3, 0.3, 0.3],
                        "param_wrench_zero": [
                            0.11954392,
                            0.78386304,
                            -12.38093143,
                            -0.17465492,
                            -0.1360826,
                            -0.06371948,
                        ],
                        "param_gravity": [-0.04990359, 0.56608497, -12.3256402],
                        "param_mass_pos": [-0.01228323, 0.01042718, 0.06536248],
                        "force_threshold": 5.0,
                        "deadband": 3.0,
                    },
                    "force_sensor_name": "",
                    "expected_urdf_link_name": "left-link_ee_ft_sensor",
                #    "waist_angle": 0.0,  ## 需要仔细核对再放开注释
                },
                "left_arm": { # Found, can't reset
                    "type": "eyou",
                    "ids": [10, 11, 12, 13, 14, 15, 16],
                    "length_per_radian": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "invert_directions": [False, True, False, True, False, True, False],
                    "control_freq": 250,
                    "interpolation_points": 13,
                    "max_velocity": 3.0,
                    "gravity_compensation_tolerance": 0.0,
                    "friction_compensation_scale": 0.0,
                    "friction_compensation_stiffness": 10,
                    "external_protections": [],
                    "offset_at_hardware_zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "joint_names": [
                        "left-joint_arm_1",
                        "left-joint_arm_2",
                        "left-joint_arm_3",
                        "left-joint_arm_4",
                        "left-joint_arm_5",
                        "left-joint_arm_6",
                        "left-joint_arm_7",
                    ],
                    "max_torque": [1800, 1800, 2400, 2400, 2000, 2000, 2000],
                    "protection_rebound": 0.0,
                    "admittance_config": {
                        "param_mass": [1.8, 1.8, 1.8, 0.03, 0.03, 0.03],
                        "param_stiff": [180.0, 180.0, 180.0, 3.0, 3.0, 3.0],
                        "param_damp": [18.0, 18.0, 18.0, 0.3, 0.3, 0.3],
                        "param_wrench_zero": [
                            0.11954392,
                            0.78386304,
                            -12.38093143,
                            -0.17465492,
                            -0.1360826,
                            -0.06371948,
                        ],
                        "param_gravity": [-0.04990359, 0.56608497, -12.3256402],
                        "param_mass_pos": [-0.01228323, 0.01042718, 0.06536248],
                        "force_threshold": 5.0,
                        "deadband": 3.0,
                    },
                    "force_sensor_name": "",
                    "expected_urdf_link_name": "left-link_ee_ft_sensor",
                    "waist_angle": 0.0,  ## 需要仔细核对再放开注释
                },

                "left_gripper": {  # Found, can't reset
                    "type": "eyou",
                    "ids": [30],
                    "length_per_radian": [0.0092115],
                    "invert_directions": [False],
                    "control_freq": 250,
                    "interpolation_points": 13,
                    "max_velocity": 0.05,
                    "gravity_compensation_tolerance": 0.0,
                    "friction_compensation_scale": 0.0,
                    "friction_compensation_stiffness": 0.0,
                    "external_protections": [],
                    "offset_at_hardware_zero": [-0.02151744830904212],
                    "joint_names": ["left-joint_gripper_finger_1"],
                    "max_torque": [1000],
                    "protection_rebound": 0.0,
                },
                "right_gripper": {
                    "type": "eyou",
                    "ids": [50],
                    "length_per_radian": [0.0092115],
                    "invert_directions": [False],
                    "control_freq": 250,
                    "interpolation_points": 13,
                    "max_velocity": 0.05,
                    "gravity_compensation_tolerance": 0.0,
                    "friction_compensation_scale": 0.0,
                    "friction_compensation_stiffness": 0.0,
                    "external_protections": [],
                    "offset_at_hardware_zero": [0.0],
                    "joint_names": ["right-joint_gripper_finger_1"],
                    "max_torque": [2000],
                    "protection_rebound": 0.0,
                },
                "rail": { # Ready
                    "type": "kinco",
                    "ids": [75],
                    "encoder_id": None,
                    "din1": "limit_low",
                    "din1_invert": False,
                    "din2": "limit_high",
                    "din2_invert": False,
                    "din3": "none",
                    "din3_invert": True,
                    "din4": "none",
                    "din4_invert": True,
                    "invert_directions": [True],
                    "joint_names": ["joint_torso_1"],
                    "inc_per_rev": 65535.0,
                    "rev_per_meter": 100.0,
                    "dec_per_rpm": 17895.424,
                    "dec_per_rps2": 1073.709,
                    "arms_per_dec": 2048.0 / (100.0 / 1.414),
                    "nm_per_arms": 2.39 / 19.2,
                    "encoder_inc_per_rev": 0.0,
                    "encoder_max_revs": 0.0,
                    "control_freq": 250.0,
                    "interpolation_points": 13,
                    "max_velocity": 0.5,
                    "max_acc": 0.5,
                    "max_dec": 0.5,
                },
                "waist": { # Ready
                    "type": "zeroerr",
                    "ids": [9],
                    "joint_names": ["joint_torso_2"],
                    "inc_per_rev": 524288,
                    "invert_directions": [False],
                    "min_position_rad": [0],
                    "max_position_rad": [0.785],
                    "control_freq": 250.0,
                    "interpolation_points": 13,
                    "max_velocity": 0.3,
                    "max_acc": 0.3,
                    "max_dec": 0.3,
                },
            },
            "planner": None, # 可先不管
            "robot_model": "", # 可先不管
        }

        self._robot = rb_python.robot.Robot(cfg_str) # 初始化机器人实例
        time.sleep(1)
    
    def shutdown(self):
        """Safely shutdown the robot"""
        try:
            print("Shutting down robot...")
            self._robot.shutdown()
            print("Robot shutdown completed")
        except Exception as e:
            print(f"Error during shutdown: {e}")


robot = None
control_running = False

def cleanup():
    """Cleanup function to ensure robot always shuts down"""
    global robot
    if robot is not None:
        robot.shutdown()

def signal_handler(sig, frame): # 简单信号处理，用于安全退出驱动
    print("\nYou pressed Ctrl+C!")
    global control_running
    control_running = False

# 注册信号处理函数
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    # Register cleanup to ensure robot always shuts down
    atexit.register(cleanup)

    # 调用 sh 脚本
    # 获取当前脚本的绝对路径
    script_path = os.path.dirname(os.path.abspath(__file__))
    # 构建 sh 脚本的绝对路径，这个路径是以本脚本目录为根基的相对路径
    shell_script_path = os.path.join(script_path, "..", "init_socket_can.sh")
    try:
        # 使用 sudo 运行脚本，并检查返回码
        print(f"Running script: {shell_script_path}")
        result = subprocess.run(["sudo", "bash", shell_script_path], check=True)
        print("Script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script: {e}")
    except FileNotFoundError:
        print(f"Error: The script {shell_script_path} was not found.")

    # Create log directory if it doesn't exist
    log_dir = os.path.join(script_path, "..", "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "file.log")
    
    config = { # 日志库配置，直接使用即可
        "refresh_rate": "30 seconds",
        "appenders": {
            "stderr": {
                "kind": "console",
                "target": "stderr",
                "encoder": {
                    "pattern": "{h({d(%Y-%m-%d %H:%M:%S.%3f)} [{t}] {l} {m})}{n}"
                },
            },
            "file": {
                "kind": "file",
                "path": log_file_path,
                "encoder": {
                    "pattern": "{h({d(%Y-%m-%d %H:%M:%S.%3f)} [{t}] {l} {m})}{n}"
                },
            },
        },
        "root": {"level": "debug", "appenders": ["stderr"]}, # 若想屏蔽信息，修改此处的log level即可
        "loggers": {},
    }
    hblog.start(config)
    print("Logging initialized")

    try:
        robot = RobotWrapper()
        print(f"Service up")
        control_running = True
        
        # Create and run keyboard controller
        print("\nStarting keyboard control...")
        controller = KeyboardController(
            robot, 
            hardware_name="left_arm", 
            gripper_name="right_gripper",
            rail_name="rail",
            waist_name="waist",
            speed=0.6,
            gripper_speed=0.05,
            rail_speed=0.1,
            waist_speed=0.1
        )
        controller.run()
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure robot shuts down properly
        print("\nCleaning up...")
        if robot is not None:
            robot.shutdown()
        print("Program finished")

