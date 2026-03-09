import rb_python
import time
import signal
import hblog
import subprocess
import os
import numpy as np
from dataclasses import dataclass
from typing import List
import atexit

@dataclass
class JointTrajectory:
    """Dataclass to store a joint trajectory with time and position sequences"""
    timestamps: np.ndarray  # Array of time points [t0, t1, t2, ...]
    positions: np.ndarray   # Array of joint positions at each time point [n_points x n_joints]
    velocities: np.ndarray  # Array of joint velocities at each time point [n_points x n_joints]
    
    def __post_init__(self):
        """Validate trajectory data"""
        if len(self.timestamps) != len(self.positions):
            raise ValueError("Timestamps and positions must have same length")
        if len(self.timestamps) != len(self.velocities):
            raise ValueError("Timestamps and velocities must have same length")
    
    @property
    def duration(self) -> float:
        """Total duration of the trajectory"""
        return self.timestamps[-1] - self.timestamps[0]
    
    @property
    def n_points(self) -> int:
        """Number of waypoints in the trajectory"""
        return len(self.timestamps)
    
    @property
    def n_joints(self) -> int:
        """Number of joints"""
        return self.positions.shape[1] if len(self.positions.shape) > 1 else 1


class JointPlanner:
    """Simple joint space planner that generates smooth trajectories"""
    
    def __init__(self, control_frequency: float = 20.0):
        """
        Initialize joint planner
        
        Args:
            control_frequency: Control frequency in Hz (default 20Hz to match robot)
        """
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
    
    def plan_trajectory(self, 
                       start_positions: List[float], 
                       target_positions: List[float], 
                       duration: float) -> JointTrajectory:
        """
        Plan a smooth trajectory from start to target using quintic polynomial
        
        Args:
            start_positions: Starting joint positions [rad]
            target_positions: Target joint positions [rad]
            duration: Time to complete the motion [seconds]
        
        Returns:
            JointTrajectory object with smooth position and velocity profiles
        """
        start_positions = np.array(start_positions)
        target_positions = np.array(target_positions)
        n_joints = len(start_positions)
        
        if len(target_positions) != n_joints:
            raise ValueError("Start and target positions must have same length")
        
        # Generate time sequence
        n_points = int(duration * self.control_frequency) + 1
        timestamps = np.linspace(0, duration, n_points)
        
        # Initialize output arrays
        positions = np.zeros((n_points, n_joints))
        velocities = np.zeros((n_points, n_joints))
        
        # Generate smooth trajectory for each joint using quintic polynomial
        # Quintic (5th order) ensures smooth position, velocity, and acceleration
        for i in range(n_joints):
            q_start = start_positions[i]
            q_end = target_positions[i]
            
            # Quintic polynomial coefficients (assuming zero velocity and acceleration at endpoints)
            # q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
            a0 = q_start
            a1 = 0  # start velocity = 0
            a2 = 0  # start acceleration = 0
            a3 = 10 * (q_end - q_start) / (duration ** 3)
            a4 = -15 * (q_end - q_start) / (duration ** 4)
            a5 = 6 * (q_end - q_start) / (duration ** 5)
            
            for j, t in enumerate(timestamps):
                # Position: q(t)
                positions[j, i] = (a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5)
                
                # Velocity: dq/dt
                velocities[j, i] = (a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4)
        
        return JointTrajectory(timestamps=timestamps, positions=positions, velocities=velocities)


class TrajectoryController:
    """Controller to execute a joint trajectory on the robot"""
    
    def __init__(self, robot_wrapper, hardware_name: str = "right_arm"):
        """
        Initialize trajectory controller
        
        Args:
            robot_wrapper: RobotWrapper instance
            hardware_name: Name of the hardware to control
        """
        self.robot_wrapper = robot_wrapper
        self.hardware_name = hardware_name
    
    def execute_trajectory(self, trajectory: JointTrajectory, monitor: bool = True) -> bool:
        """
        Execute a joint trajectory
        
        Args:
            trajectory: JointTrajectory to execute
            monitor: If True, print progress during execution
        
        Returns:
            True if trajectory completed successfully, False if interrupted
        """
        start_time = time.time()
        trajectory_start_time = trajectory.timestamps[0]
        trajectory_end_time = trajectory.timestamps[-1]
        
        print(f"Executing trajectory with {trajectory.n_points} waypoints over {trajectory.duration:.2f}s")
        
        waypoint_idx = 0
        
        while control_running:
            current_time = time.time() - start_time
            
            # Check if trajectory is complete
            if current_time >= trajectory_end_time:
                # Send final position
                final_position = trajectory.positions[-1, :].tolist()
                self.robot_wrapper._robot.set_actions({
                    self.hardware_name: {
                        "type": "position",
                        "position": final_position
                    }
                })
                print(f"Trajectory completed successfully!")
                return True
            
            # Find the appropriate waypoint for current time
            while waypoint_idx < trajectory.n_points - 1 and current_time >= trajectory.timestamps[waypoint_idx + 1]:
                waypoint_idx += 1
            
            # Get current target position
            target_position = trajectory.positions[waypoint_idx, :].tolist()
            
            # Send position command to robot
            self.robot_wrapper._robot.set_actions({
                self.hardware_name: {
                    "type": "position",
                    "position": target_position
                }
            })
            # print(target_position)

            # Monitor progress
            if monitor and waypoint_idx % 50 == 0:  # Print every 50 waypoints
                progress = (current_time / trajectory.duration) * 100
                print(f"Progress: {progress:.1f}% | Time: {current_time:.2f}s/{trajectory.duration:.2f}s")
            
            # Sleep to maintain control frequency
            time.sleep(0.004)  # ~250Hz control loop
        
        print("Trajectory execution interrupted!")
        return False


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
        
        # Get current joint positions
        print("\nReading current joint positions...")
        time.sleep(0.05)
        states = robot._robot.get_states(["right_arm"])
        current_state = states.get("right_arm", None)
        current_positions = current_state.get("position", None)
        while current_positions is None:
            print("Waiting for valid state...")
            time.sleep(0.05)
            states = robot._robot.get_states(["right_arm"])
            current_state = states.get("right_arm", None)
            current_positions = current_state.get("position", None)
        print(f"Current positions: {current_positions}")
        
        # Define target positions
        target_positions = [-2.1977976844056335, 0.7659509700797422, 1.5, 1.6625067351122131, -0.2034166738727585, 0.5, 2.226552230214193]
        print(f"Target positions: {target_positions}")
        
        # Create planner and controller
        planner = JointPlanner(control_frequency=20.0)
        controller = TrajectoryController(robot, hardware_name="right_arm")
        
        # Plan trajectory with 5 second duration
        print("\nPlanning trajectory...")
        trajectory = planner.plan_trajectory(
            start_positions=current_positions,
            target_positions=target_positions,
            duration=5.0
        )
        
        print(f"Trajectory planned: {trajectory.n_points} waypoints, "
              f"{trajectory.duration:.2f}s duration, {trajectory.n_joints} joints")
        print(f"Start: {trajectory.positions[0, :]}")
        print(f"End: {trajectory.positions[-1, :]}")
        print(f"start 10: {trajectory.positions[10, :]}")
        
        # Execute trajectory
        print("\nExecuting trajectory...")
        success = controller.execute_trajectory(trajectory, monitor=True)
        
        if success:
            print("\n✓ Trajectory execution completed successfully!")
            
            # Hold final position for a moment
            print("Holding final position for 2 seconds...")
            time.sleep(2.0)
        else:
            print("\n✗ Trajectory execution was interrupted")
            
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

