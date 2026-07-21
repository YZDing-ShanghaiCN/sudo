"""抓取和交付相关动作节点。"""

from __future__ import annotations

import py_trees

from behaviours.base import RobotBehaviour
from robot_state import ScenarioConfig, SimulatedRobotState


class GraspObject(RobotBehaviour):
    """模拟抓取水杯。

    抓取结果由场景配置决定，便于演示 Retry Decorator。
    """

    def __init__(self, name: str, state: SimulatedRobotState, config: ScenarioConfig) -> None:
        super().__init__(name, state, config)
        self.blackboard.register_key("object_grasped", py_trees.common.Access.WRITE)
        self.blackboard.register_key("task_progress", py_trees.common.Access.WRITE)

    def initialise(self) -> None:
        super().initialise()
        self.state.grasp_attempts += 1

    def update(self) -> py_trees.common.Status:
        self.state.battery_level -= 0.8
        if self.state.grasp_attempts <= self.config.grasp_failures_before_success:
            self.state.object_grasped = False
            self.blackboard.object_grasped = False
            self.blackboard.task_progress = "grasp_failed"
            self.feedback_message = f"grasp attempt {self.state.grasp_attempts} failed"
            return py_trees.common.Status.FAILURE

        self.state.object_grasped = True
        self.blackboard.object_grasped = True
        self.blackboard.task_progress = "grasped"
        self.feedback_message = f"grasp attempt {self.state.grasp_attempts} succeeded"
        return py_trees.common.Status.SUCCESS


class DeliverObject(RobotBehaviour):
    """模拟把水杯交给用户或放到用户面前。"""

    def __init__(
        self,
        name: str,
        state: SimulatedRobotState,
        config: ScenarioConfig,
        ticks_to_deliver: int = 2,
    ) -> None:
        super().__init__(name, state, config)
        self.ticks_to_deliver = ticks_to_deliver
        self.local_ticks = 0
        self.blackboard.register_key("object_grasped", py_trees.common.Access.WRITE)
        self.blackboard.register_key("task_progress", py_trees.common.Access.WRITE)

    def initialise(self) -> None:
        super().initialise()
        self.local_ticks = 0

    def update(self) -> py_trees.common.Status:
        if not self.state.object_grasped:
            self.feedback_message = "cannot deliver because cup is not grasped"
            self.blackboard.task_progress = "delivery_failed"
            return py_trees.common.Status.FAILURE

        self.local_ticks += 1
        if self.local_ticks < self.ticks_to_deliver:
            self.feedback_message = f"placing cup ({self.local_ticks}/{self.ticks_to_deliver})"
            self.blackboard.task_progress = "delivering"
            return py_trees.common.Status.RUNNING

        self.state.object_grasped = False
        self.state.delivery_completed = True
        self.blackboard.object_grasped = False
        self.blackboard.task_progress = "delivered"
        self.feedback_message = "cup delivered to user"
        return py_trees.common.Status.SUCCESS
