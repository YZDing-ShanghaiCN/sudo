import rb_python
import time
import signal
import hblog
import subprocess
import os

class RobotWrapper:
    def __init__(self):
        print(f"new a robot")
        cfg_str = {
            "hardware": {  # 硬件列表，想要启动多少个硬件，从此处配置，以下为模板配置项
                # "left_arm": { # Found, can't reset
                #     "type": "eyou",
                #     "ids": [10, 11, 12, 13, 14, 15, 16],
                #     "length_per_radian": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                #     "invert_directions": [False, True, False, True, False, True, False],
                #     "control_freq": 250,
                #     "interpolation_points": 13,
                #     "max_velocity": 3.0,
                #     "gravity_compensation_tolerance": 0.0,
                #     "friction_compensation_scale": 0.0,
                #     "friction_compensation_stiffness": 10,
                #     "external_protections": [],
                #     "offset_at_hardware_zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                #     "joint_names": [
                #         "left-joint_arm_1",
                #         "left-joint_arm_2",
                #         "left-joint_arm_3",
                #         "left-joint_arm_4",
                #         "left-joint_arm_5",
                #         "left-joint_arm_6",
                #         "left-joint_arm_7",
                #     ],
                #     "max_torque": [1800, 1800, 2400, 2400, 2000, 2000, 2000],
                #     "protection_rebound": 0.0,
                #     "admittance_config": {
                #         "param_mass": [1.8, 1.8, 1.8, 0.03, 0.03, 0.03],
                #         "param_stiff": [180.0, 180.0, 180.0, 3.0, 3.0, 3.0],
                #         "param_damp": [18.0, 18.0, 18.0, 0.3, 0.3, 0.3],
                #         "param_wrench_zero": [
                #             0.11954392,
                #             0.78386304,
                #             -12.38093143,
                #             -0.17465492,
                #             -0.1360826,
                #             -0.06371948,
                #         ],
                #         "param_gravity": [-0.04990359, 0.56608497, -12.3256402],
                #         "param_mass_pos": [-0.01228323, 0.01042718, 0.06536248],
                #         "force_threshold": 5.0,
                #         "deadband": 3.0,
                #     },
                #     "force_sensor_name": "",
                #     "expected_urdf_link_name": "left-link_ee_ft_sensor",
                #     "waist_angle": 0.0,  ## 需要仔细核对再放开注释
                # },
                # "right_arm": { # Found, can't reset
                #     "type": "eyou",
                #     "ids": [20, 21, 22, 23, 24, 25, 26],
                #     "length_per_radian": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                #     "invert_directions": [True, False, True, False, True, False, True],
                #     "control_freq": 250,
                #     "interpolation_points": 13,
                #     "max_velocity": 3.0,
                #     "gravity_compensation_tolerance": 0.0,
                #     "friction_compensation_scale": 0.0,
                #     "friction_compensation_stiffness": 10,
                #     "external_protections": [],
                #     "offset_at_hardware_zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                #     "joint_names": [
                #         "right-joint_arm_1",
                #         "right-joint_arm_2",
                #         "right-joint_arm_3",
                #         "right-joint_arm_4",
                #         "right-joint_arm_5",
                #         "right-joint_arm_6",
                #         "right-joint_arm_7",
                #     ],
                #     "max_torque": [1800, 1800, 2400, 2400, 2000, 2000, 2000],
                #     "protection_rebound": 0.0,
                #     "admittance_config": {
                #         "param_mass": [1.8, 1.8, 1.8, 0.03, 0.03, 0.03],
                #         "param_stiff": [180.0, 180.0, 180.0, 3.0, 3.0, 3.0],
                #         "param_damp": [18.0, 18.0, 18.0, 0.3, 0.3, 0.3],
                #         "param_wrench_zero": [
                #             0.11954392,
                #             0.78386304,
                #             -12.38093143,
                #             -0.17465492,
                #             -0.1360826,
                #             -0.06371948,
                #         ],
                #         "param_gravity": [-0.04990359, 0.56608497, -12.3256402],
                #         "param_mass_pos": [-0.01228323, 0.01042718, 0.06536248],
                #         "force_threshold": 5.0,
                #         "deadband": 3.0,
                #     },
                #     "force_sensor_name": "",
                #     "expected_urdf_link_name": "left-link_ee_ft_sensor",
                #    "waist_angle": 0.0,  ## 需要仔细核对再放开注释
                # },
                # "left_gripper": {  # Found, can't reset
                #     "type": "eyou",
                #     "ids": [30],
                #     "length_per_radian": [0.0092115],
                #     "invert_directions": [False],
                #     "control_freq": 250,
                #     "interpolation_points": 13,
                #     "max_velocity": 0.05,
                #     "gravity_compensation_tolerance": 0.0,
                #     "friction_compensation_scale": 0.0,
                #     "friction_compensation_stiffness": 0.0,
                #     "external_protections": [],
                #     "offset_at_hardware_zero": [-0.02151744830904212],
                #     "joint_names": ["left-joint_gripper_finger_1"],
                #     "max_torque": [1000],
                #     "protection_rebound": 0.0,
                # },
                # "right_gripper": {
                #     "type": "eyou",
                #     "ids": [50],
                #     "length_per_radian": [0.0092115],
                #     "invert_directions": [False],
                #     "control_freq": 250,
                #     "interpolation_points": 13,
                #     "max_velocity": 0.05,
                #     "gravity_compensation_tolerance": 0.0,
                #     "friction_compensation_scale": 0.0,
                #     "friction_compensation_stiffness": 0.0,
                #     "external_protections": [],
                #     "offset_at_hardware_zero": [0.0],
                #     "joint_names": ["right-joint_gripper_finger_1"],
                #     "max_torque": [2000],
                #     "protection_rebound": 0.0,
                # },
                # "left_gripper": { 
                #     "type": "robstride",
                #     "ids": [30],
                #     "length_per_radian": [0.010197],
                #     "invert_directions": [False],
                #     "position_limit": [[0.01, 0.05]],
                #     "control_freq": 250,
                #     "interpolation_points": 13,
                #     "max_velocity": 1,
                #     "gravity_compensation_tolerance": 0.0,
                #     "friction_compensation_scale": 0.0,
                #     "friction_compensation_stiffness": 0.0,
                #     "external_protections": [],
                #     "offset_at_hardware_zero": [0.0],
                #     "joint_names": ["joint_right_gripper"],
                #     "max_torque": [1.0],
                #     "wrap_around_zero": [False],
                #     "protection_rebound": 0.0,
                # },
                # "right_gripper": { # Not found
                #     "type": "robstride",
                #     "ids": [50],
                #     "length_per_radian": [0.010197],
                #     "invert_directions": [False],
                #     "position_limit": [[-0.2, 0.2]],
                #     "control_freq": 250,
                #     "interpolation_points": 13,
                #     "max_velocity": 1,
                #     "gravity_compensation_tolerance": 0.0,
                #     "friction_compensation_scale": 0.0,
                #     "friction_compensation_stiffness": 0.0,
                #     "external_protections": [],
                #     "offset_at_hardware_zero": [0.0],
                #     "joint_names": ["joint_right_gripper"],
                #     "max_torque": [1.0],
                #     "wrap_around_zero": [False],
                #     "protection_rebound": 0.0,
                # },
                # "left_wrench": { # exit xjc
                #     "type": "xjc",
                #     "usb_bus": 1,
                #     "usb_ports": [1, 4],
                #     "usb_index": 0,
                #     "id": 5,
                #     "use_can_version": True,
                #     "freq": 200,
                #     "legacy_mode": False,
                #     "link_name": "",
                #     "calibration": None,
                # },
                # "right_wrench": { # exit xjc
                #     "type": "xjc",
                #     "usb_bus": 1,
                #     "usb_ports": [1, 4],
                #     "usb_index": 0,
                #     "id": 6,
                #     "use_can_version": True,
                #     "freq": 200,
                #     "legacy_mode": False,
                #     "link_name": "",
                #     "calibration": None,
                # },
                # "left_tactile": {
                #     "type": "dayang",
                #     "usb_bus": 1,
                #     "usb_ports": [3, 2, 1, 4],
                #     "usb_index": 0,
                #     "link_name": "",
                # },
                # "right_tactile": {
                #     "type": "dayang",
                #     "usb_bus": 1,
                #     "usb_ports": [4, 3, 1, 4],
                #     "usb_index": 0,
                #     "link_name": "",
                # },
                # "head": {
                #     "type": "eyou",
                #     "ids": [70],
                #     "length_per_radian": [1.0],
                #     "invert_directions": [True],
                #     "control_freq": 250,
                #     "interpolation_points": 13,
                #     "max_velocity": 3.0,
                #     "gravity_compensation_tolerance": 0.0,
                #     "friction_compensation_scale": 0.0,
                #     "friction_compensation_stiffness": 0.0,
                #     "external_protections": [],
                #     "offset_at_hardware_zero": [0.0],
                #     "joint_names": ["joint_head_camera"],
                #     "max_torque": [1000],
                #     "protection_rebound": 0.0,
                # },
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
                # "base": {
                #     "type": "base_b1p0",
                #     "steering_ids": [5, 6, 7],
                #     "steering_joint_names": [
                #         "joint_wheel_2_1",
                #         "joint_wheel_3_1",
                #         "joint_wheel_4_1",
                #     ],
                #     "steering_motor_acc_dec_vel": [
                #         [6.28, 6.28, 3.14],
                #         [6.28, 6.28, 3.14],
                #         [6.28, 6.28, 3.14],
                #     ],
                #     "driving_ids": [1, 2, 3],
                #     "driving_joint_names": [
                #         "joint_wheel_2_3",
                #         "joint_wheel_3_3",
                #         "joint_wheel_4_3",
                #     ],
                #     "wheel_placement": [[0.265, 0.215], [-0.265, 0], [0.265, -0.215]],
                #     "wheel_radius": 0.0825,
                #     "wheel_direction": [-1.0, -1.0, 1.0],
                #     "steering_motor_shaft_ratio": 1.0,
                #     "steering_threshold": 0.1,
                #     "speed_threshold": 0.1,
                #     "max_linear_velocity": 0.3,
                #     "max_angular_velocity": 30.0,
                #     "control_freq": 250,
                #     "start_with_calibration_mode": False,
                # },
                # "battery": {
                #     "type": "battery",
                #     "usb_bus": 1,
                #     "usb_ports": [1, 2, 1],
                #     "usb_index": 0,
                #     "check_interval_sec": 2,
                # },
                # "io": {
                #     "type": "zhongsheng",
                #     "ip_and_port": "192.168.100.7:8234",
                #     "check_interval_ms": 1200,
                #     "address_map": {
                #         "map": {
                #             "Hub power": 4,
                #             "Camera power": 5,
                #             "Motor power": 6,
                #         }
                #     },
                # },
                # "lidar360-up": {
                #     "type": "livox",
                #     "device_ip": "192.168.100.21",
                #     "queue_size": 20000,
                #     "imu_cycle_ms": 100,
                #     "use_ptp": False,
                #     "time_stamp": "first",
                # },
                # "lidar360-left-front": {
                #     "type": "livox",
                #     "device_ip": "192.168.100.22",
                #     "queue_size": 20000,
                #     "imu_cycle_ms": 100,
                #     "use_ptp": False,
                #     "time_stamp": "first",
                # },
                # "lidar360-right-back": {
                #     "type": "livox",
                #     "device_ip": "192.168.100.23",
                #     "queue_size": 20000,
                #     "imu_cycle_ms": 100,
                #     "use_ptp": False,
                #     "time_stamp": "first",
                # },
                # "lidar-front": {
                #     "type": "lidar_ls",
                #     "usb_bus": 1,
                #     "usb_ports": [1, 4, 3],
                #     "usb_index": 0,
                #     "link_name": "",
                #     "use_dual_wave": False,
                # },            
            },
            "planner": None, # 可先不管
            "robot_model": "", # 可先不管
        }

        self._robot = rb_python.robot.Robot(cfg_str) # 初始化机器人实例
        time.sleep(1)

robot = 0
control_running = False

def signal_handler(sig, frame): # 简单信号处理，用于安全退出驱动
    print("You pressed Ctrl+C!")
    global control_running
    control_running = False
    robot._robot.shutdown()
    print("Robot shutdown finish")
    exit()

# 注册信号处理函数
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

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

    robot = RobotWrapper()
    print(f"Service up")
    control_running = True
    # robot._robot.shutdown()
    # exit()
    # 主逻辑循环
    while control_running:
        time.sleep(0.05)
        # 获取名称为"your_sensor"的硬件的状态
        states = robot._robot.get_states(["waist", "rail"]) 
        # print(states)
        my_motor_1_state = states.get("waist", None)
        # 获取位置列表，得到的数据是关节的当前位置的列表，单位根据cfg_str中配置决定
        my_motor_1_position = my_motor_1_state.get("position", None)
        # 获取速度列表，得到的数据是关节的当前速度的列表，单位根据cfg_str中配置决定
        my_motor_1_velocity = my_motor_1_state.get("velocity", None)
        # 获取力矩列表，得到的数据是关节的当前力矩的列表，单位一般是Nm
        my_motor_1_torque = my_motor_1_state.get("torque", None)
        print(f"waist position: {my_motor_1_position}, velocity: {my_motor_1_velocity}, torque: {my_motor_1_torque}")
        # 对名称为"motor_name"的硬件进行类型为position的控制，控制位两个关节，角度为1.0弧度以及0.0弧度
        # robot._robot.set_actions({"motor_name": {"type": "position", "position": [1.0, 0.0]}}) 
        # robot._robot.set_actions({"right_arm": {"type": "position", "position": [-2.2, 0.7685329179405398, 2.267851055794973, 1.6621441233170569, -0.2034138261361473, 0.5714913770946625, 2.226551280968656]}})
        # robot._robot.set_actions({"left_gripper": {"type": "position", "position": [0.0]}})  
        # 对名称为test_base的底盘硬件进行速度控制，控制输入为twist [v_x（m/s）, v_y(m/s), v_w(rad/s)]
        # robot._robot.set_actions({"test_base": {"type": "velocity", "velocity": [1.0, 0.0, 0.0]}}) 
    robot._robot.shutdown()
    exit()
