"""模拟机器人状态和场景配置。

这里没有任何 ROS、硬件 SDK 或网络服务。所有“机器人动作”都通过修改
SimulatedRobotState 的字段来模拟，便于初学者把注意力放在行为树本身。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import py_trees


ScenarioName = Literal[
    "normal",
    "recovery",
    "low-battery",
    "grasp-failure",
    "object-dropped",
]


@dataclass(slots=True)
class ScenarioConfig:
    """控制不同演示场景的确定性开关。"""

    name: ScenarioName
    initial_battery: float = 90.0
    sensors_ready: bool = True
    target_known: bool = False
    target_location_confidence: float = 0.0
    fail_first_target_navigation: bool = False
    grasp_failures_before_success: int = 1
    drop_object_during_delivery: bool = False
    delivery_battery_drain_per_tick: float = 1.5


@dataclass(slots=True)
class SimulatedRobotState:
    """用普通变量模拟移动机器人执行任务时的内部状态。"""

    battery_level: float = 90.0
    sensors_ready: bool = True
    target_known: bool = False
    target_location: str | None = None
    target_location_confidence: float = 0.0
    navigation_progress: int = 0
    user_navigation_progress: int = 0
    navigation_attempts: int = 0
    normal_navigation_failed: bool = False
    path_cleared: bool = False
    path_replanned: bool = False
    object_found: bool = False
    object_grasped: bool = False
    object_dropped: bool = False
    delivery_completed: bool = False
    search_ticks: int = 0
    grasp_attempts: int = 0
    deliver_ticks: int = 0
    tick_count: int = 0

    def as_dict(self) -> dict[str, object]:
        """用于终端打印，避免直接暴露 dataclass repr 造成信息过多。"""

        return {
            "battery_level": round(self.battery_level, 1),
            "sensors_ready": self.sensors_ready,
            "target_known": self.target_known,
            "target_location": self.target_location,
            "target_location_confidence": round(self.target_location_confidence, 2),
            "navigation_progress": self.navigation_progress,
            "user_navigation_progress": self.user_navigation_progress,
            "navigation_attempts": self.navigation_attempts,
            "path_cleared": self.path_cleared,
            "path_replanned": self.path_replanned,
            "object_found": self.object_found,
            "object_grasped": self.object_grasped,
            "object_dropped": self.object_dropped,
            "delivery_completed": self.delivery_completed,
        }


def config_for_scenario(name: ScenarioName) -> ScenarioConfig:
    """返回可复现的场景配置。"""

    configs: dict[ScenarioName, ScenarioConfig] = {
        "normal": ScenarioConfig(
            name="normal",
            target_known=True,
            target_location_confidence=0.92,
            grasp_failures_before_success=1,
        ),
        "recovery": ScenarioConfig(
            name="recovery",
            fail_first_target_navigation=True,
            grasp_failures_before_success=1,
        ),
        "low-battery": ScenarioConfig(
            name="low-battery",
            initial_battery=12.0,
            grasp_failures_before_success=1,
        ),
        "grasp-failure": ScenarioConfig(
            name="grasp-failure",
            target_known=True,
            target_location_confidence=0.9,
            grasp_failures_before_success=99,
        ),
        "object-dropped": ScenarioConfig(
            name="object-dropped",
            target_known=True,
            target_location_confidence=0.9,
            grasp_failures_before_success=1,
            drop_object_during_delivery=True,
        ),
    }
    return configs[name]


def create_state(config: ScenarioConfig) -> SimulatedRobotState:
    """根据场景创建初始状态。"""

    return SimulatedRobotState(
        battery_level=config.initial_battery,
        sensors_ready=config.sensors_ready,
        target_known=config.target_known,
        target_location="kitchen_table" if config.target_known else None,
        target_location_confidence=config.target_location_confidence,
    )


def reset_blackboard() -> py_trees.blackboard.Client:
    """清空并初始化 Blackboard。

    Blackboard 类似行为树中的共享任务状态，不同节点可以通过它交换信息。
    这里统一初始化，保证每个场景和每个测试都是干净起步。
    """

    py_trees.blackboard.Blackboard.clear()
    client = py_trees.blackboard.Client(name="blackboard_initializer")
    for key in (
        "target_location",
        "target_confidence",
        "navigation_failure_reason",
        "recovery_count",
        "object_grasped",
        "task_progress",
    ):
        client.register_key(key=key, access=py_trees.common.Access.WRITE)

    client.target_location = None
    client.target_confidence = 0.0
    client.navigation_failure_reason = None
    client.recovery_count = 0
    client.object_grasped = False
    client.task_progress = "not_started"
    return client
