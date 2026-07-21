"""条件节点：只检查状态，不执行耗时动作。"""

from __future__ import annotations

import py_trees

from behaviours.base import RobotBehaviour
from robot_state import ScenarioConfig, SimulatedRobotState


class CheckBattery(RobotBehaviour):
    """检查电量是否足够。"""

    def __init__(
        self,
        name: str,
        state: SimulatedRobotState,
        config: ScenarioConfig,
        minimum_level: float = 20.0,
    ) -> None:
        super().__init__(name, state, config)
        self.minimum_level = minimum_level
        self.blackboard.register_key("task_progress", py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        if self.state.battery_level >= self.minimum_level:
            self.feedback_message = (
                f"battery {self.state.battery_level:.1f}% >= {self.minimum_level:.1f}%"
            )
            self.blackboard.task_progress = "battery_ok"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = (
            f"battery {self.state.battery_level:.1f}% < {self.minimum_level:.1f}%"
        )
        self.blackboard.task_progress = "battery_low"
        return py_trees.common.Status.FAILURE


class CheckSensors(RobotBehaviour):
    """检查传感器是否可用。"""

    def __init__(self, name: str, state: SimulatedRobotState, config: ScenarioConfig) -> None:
        super().__init__(name, state, config)
        self.blackboard.register_key("task_progress", py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        if self.state.sensors_ready:
            self.feedback_message = "all simulated sensors are ready"
            self.blackboard.task_progress = "sensors_ok"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = "simulated sensors are not ready"
        self.blackboard.task_progress = "sensors_failed"
        return py_trees.common.Status.FAILURE


class CheckTargetMemory(RobotBehaviour):
    """检查记忆或黑板中是否已有目标位置。"""

    def __init__(self, name: str, state: SimulatedRobotState, config: ScenarioConfig) -> None:
        super().__init__(name, state, config)
        self.blackboard.register_key("target_location", py_trees.common.Access.READ)
        self.blackboard.register_key("target_confidence", py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        target_location = self.blackboard.target_location
        target_confidence = self.blackboard.target_confidence
        if target_location and target_confidence > 0.0:
            self.feedback_message = (
                f"remembered target={target_location}, confidence={target_confidence:.2f}"
            )
            return py_trees.common.Status.SUCCESS

        self.feedback_message = "no target location in memory"
        return py_trees.common.Status.FAILURE


class ValidateTargetLocation(RobotBehaviour):
    """验证目标位置置信度是否足够可靠。"""

    def __init__(
        self,
        name: str,
        state: SimulatedRobotState,
        config: ScenarioConfig,
        minimum_confidence: float = 0.75,
    ) -> None:
        super().__init__(name, state, config)
        self.minimum_confidence = minimum_confidence
        self.blackboard.register_key("target_location", py_trees.common.Access.READ)
        self.blackboard.register_key("target_confidence", py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        confidence = self.blackboard.target_confidence
        if self.blackboard.target_location and confidence >= self.minimum_confidence:
            self.feedback_message = f"target confidence {confidence:.2f} is reliable"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = f"target confidence {confidence:.2f} is unreliable"
        return py_trees.common.Status.FAILURE


class VerifyGrasp(RobotBehaviour):
    """验证物体是否已经被抓住。"""

    def __init__(self, name: str, state: SimulatedRobotState, config: ScenarioConfig) -> None:
        super().__init__(name, state, config)
        self.blackboard.register_key("object_grasped", py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        if self.blackboard.object_grasped and self.state.object_grasped:
            self.feedback_message = "cup is confirmed in gripper"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = "cup is not in gripper"
        return py_trees.common.Status.FAILURE


class VerifyDelivery(RobotBehaviour):
    """验证任务是否完成。"""

    def __init__(self, name: str, state: SimulatedRobotState, config: ScenarioConfig) -> None:
        super().__init__(name, state, config)
        self.blackboard.register_key("task_progress", py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        if self.state.delivery_completed and not self.state.object_grasped:
            self.feedback_message = "delivery is complete"
            self.blackboard.task_progress = "complete"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = "delivery is not complete"
        self.blackboard.task_progress = "delivery_verification_failed"
        return py_trees.common.Status.FAILURE
