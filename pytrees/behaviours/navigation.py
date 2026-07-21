"""导航和搜索相关动作节点。"""

from __future__ import annotations

import py_trees

from behaviours.base import RobotBehaviour
from robot_state import ScenarioConfig, SimulatedRobotState


class SearchForObject(RobotBehaviour):
    """模拟搜索水杯，经过若干 Tick 后写入 Blackboard。"""

    def __init__(
        self,
        name: str,
        state: SimulatedRobotState,
        config: ScenarioConfig,
        ticks_to_find: int = 3,
    ) -> None:
        super().__init__(name, state, config)
        self.ticks_to_find = ticks_to_find
        self.local_ticks = 0
        self.blackboard.register_key("target_location", py_trees.common.Access.WRITE)
        self.blackboard.register_key("target_confidence", py_trees.common.Access.WRITE)
        self.blackboard.register_key("task_progress", py_trees.common.Access.WRITE)

    def initialise(self) -> None:
        super().initialise()
        self.local_ticks = 0

    def update(self) -> py_trees.common.Status:
        self.local_ticks += 1
        self.state.search_ticks += 1
        self.state.battery_level -= 0.5
        if self.local_ticks < self.ticks_to_find:
            self.feedback_message = f"searching for cup ({self.local_ticks}/{self.ticks_to_find})"
            self.blackboard.task_progress = "searching"
            return py_trees.common.Status.RUNNING

        self.state.object_found = True
        self.state.target_known = True
        self.state.target_location = "kitchen_table"
        self.state.target_location_confidence = 0.88
        self.blackboard.target_location = self.state.target_location
        self.blackboard.target_confidence = self.state.target_location_confidence
        self.blackboard.task_progress = "target_found"
        self.feedback_message = "found cup at kitchen_table, wrote target to blackboard"
        return py_trees.common.Status.SUCCESS


class NavigateToTarget(RobotBehaviour):
    """模拟导航到水杯附近。

    第一次正常导航可按场景配置故意失败，恢复分支重新规划后再次导航成功。
    """

    def __init__(
        self,
        name: str,
        state: SimulatedRobotState,
        config: ScenarioConfig,
        *,
        recovered: bool = False,
        ticks_to_arrive: int = 3,
    ) -> None:
        super().__init__(name, state, config)
        self.recovered = recovered
        self.ticks_to_arrive = ticks_to_arrive
        self.blackboard.register_key("target_location", py_trees.common.Access.READ)
        self.blackboard.register_key("navigation_failure_reason", py_trees.common.Access.WRITE)
        self.blackboard.register_key("task_progress", py_trees.common.Access.WRITE)

    def initialise(self) -> None:
        super().initialise()
        self.state.navigation_progress = 0
        self.state.navigation_attempts += 1

    def update(self) -> py_trees.common.Status:
        target_location = self.blackboard.target_location
        if not target_location:
            self.feedback_message = "cannot navigate because target location is missing"
            self.blackboard.navigation_failure_reason = "missing_target"
            return py_trees.common.Status.FAILURE

        self.state.navigation_progress += 1
        self.state.battery_level -= 1.0
        self.blackboard.task_progress = "navigating_to_target"
        if self.state.navigation_progress < self.ticks_to_arrive:
            mode = "replanned path" if self.recovered else "normal path"
            self.feedback_message = (
                f"{mode} to {target_location}: "
                f"{self.state.navigation_progress}/{self.ticks_to_arrive}"
            )
            return py_trees.common.Status.RUNNING

        if (
            self.config.fail_first_target_navigation
            and not self.recovered
            and not self.state.normal_navigation_failed
        ):
            self.state.normal_navigation_failed = True
            self.blackboard.navigation_failure_reason = "path_blocked"
            self.feedback_message = "normal navigation failed: path_blocked"
            return py_trees.common.Status.FAILURE

        self.blackboard.navigation_failure_reason = None
        mode = "after recovery" if self.recovered else "normally"
        self.feedback_message = f"arrived near cup {mode}"
        return py_trees.common.Status.SUCCESS


class ClearOldPath(RobotBehaviour):
    """清除旧路径，为恢复流程做准备。"""

    def __init__(self, name: str, state: SimulatedRobotState, config: ScenarioConfig) -> None:
        super().__init__(name, state, config)
        self.blackboard.register_key("task_progress", py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        self.state.path_cleared = True
        self.blackboard.task_progress = "old_path_cleared"
        self.feedback_message = "old path cleared"
        return py_trees.common.Status.SUCCESS


class ReplanPath(RobotBehaviour):
    """模拟重新规划，经过两个 Tick 后完成。"""

    def __init__(self, name: str, state: SimulatedRobotState, config: ScenarioConfig) -> None:
        super().__init__(name, state, config)
        self.local_ticks = 0
        self.blackboard.register_key("recovery_count", py_trees.common.Access.WRITE)
        self.blackboard.register_key("task_progress", py_trees.common.Access.WRITE)

    def initialise(self) -> None:
        super().initialise()
        self.local_ticks = 0

    def update(self) -> py_trees.common.Status:
        self.local_ticks += 1
        self.state.battery_level -= 0.5
        if self.local_ticks < 2:
            self.feedback_message = "replanning path (1/2)"
            self.blackboard.task_progress = "replanning"
            return py_trees.common.Status.RUNNING

        self.state.path_replanned = True
        self.blackboard.recovery_count = self.blackboard.recovery_count + 1
        self.blackboard.task_progress = "path_replanned"
        self.feedback_message = f"path replanned, recovery_count={self.blackboard.recovery_count}"
        return py_trees.common.Status.SUCCESS


class NavigateToUser(RobotBehaviour):
    """模拟携带水杯导航到用户位置。"""

    def __init__(
        self,
        name: str,
        state: SimulatedRobotState,
        config: ScenarioConfig,
        ticks_to_arrive: int = 4,
    ) -> None:
        super().__init__(name, state, config)
        self.ticks_to_arrive = ticks_to_arrive
        self.blackboard.register_key("object_grasped", py_trees.common.Access.WRITE)
        self.blackboard.register_key("task_progress", py_trees.common.Access.WRITE)

    def initialise(self) -> None:
        super().initialise()
        self.state.user_navigation_progress = 0

    def update(self) -> py_trees.common.Status:
        self.state.user_navigation_progress += 1
        self.state.battery_level -= self.config.delivery_battery_drain_per_tick
        self.blackboard.task_progress = "navigating_to_user"

        if self.config.drop_object_during_delivery and self.state.user_navigation_progress == 2:
            self.state.object_dropped = True
            self.state.object_grasped = False
            self.blackboard.object_grasped = False

        if self.state.user_navigation_progress < self.ticks_to_arrive:
            self.feedback_message = (
                f"moving to user: {self.state.user_navigation_progress}/{self.ticks_to_arrive}"
            )
            return py_trees.common.Status.RUNNING

        self.feedback_message = "arrived at user location"
        return py_trees.common.Status.SUCCESS
