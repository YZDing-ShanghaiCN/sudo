"""监控节点：适合放在 Parallel 中与主动作同步运行。"""

from __future__ import annotations

import py_trees

from behaviours.base import RobotBehaviour
from robot_state import ScenarioConfig, SimulatedRobotState


class MonitorBattery(RobotBehaviour):
    """递送过程中持续监控电量。"""

    def __init__(
        self,
        name: str,
        state: SimulatedRobotState,
        config: ScenarioConfig,
        minimum_level: float = 15.0,
    ) -> None:
        super().__init__(name, state, config)
        self.minimum_level = minimum_level

    def update(self) -> py_trees.common.Status:
        if self.state.battery_level >= self.minimum_level:
            self.feedback_message = f"battery safe: {self.state.battery_level:.1f}%"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = f"battery unsafe: {self.state.battery_level:.1f}%"
        return py_trees.common.Status.FAILURE


class MonitorObject(RobotBehaviour):
    """递送过程中持续监控水杯是否仍被抓住。"""

    def __init__(self, name: str, state: SimulatedRobotState, config: ScenarioConfig) -> None:
        super().__init__(name, state, config)
        self.blackboard.register_key("object_grasped", py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        if self.state.object_dropped:
            self.feedback_message = "cup dropped during delivery"
            return py_trees.common.Status.FAILURE

        if self.state.object_grasped and self.blackboard.object_grasped:
            self.feedback_message = "cup is still grasped"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = "cup is missing from gripper"
        return py_trees.common.Status.FAILURE
