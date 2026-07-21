"""所有自定义行为节点的公共基类。"""

from __future__ import annotations

import py_trees

from robot_state import ScenarioConfig, SimulatedRobotState


class RobotBehaviour(py_trees.behaviour.Behaviour):
    """带生命周期注释的行为节点基类。

    所有具体节点都继承这个类，因此也都是 py_trees.behaviour.Behaviour。
    """

    def __init__(
        self,
        name: str,
        state: SimulatedRobotState,
        config: ScenarioConfig,
    ) -> None:
        super().__init__(name=name)
        self.state = state
        self.config = config
        self.blackboard = py_trees.blackboard.Client(name=name)

    def setup(self, **kwargs: object) -> None:
        """树 setup 时调用一次，用来申请资源或检查依赖。

        在真实机器人中，这里常用于创建 action client、订阅器、硬件句柄等。
        本项目是纯模拟，所以只记录 feedback。
        """

        self.feedback_message = "setup complete"

    def initialise(self) -> None:
        """节点从非 RUNNING 状态进入运行时调用。

        如果一个动作要跨多个 Tick 执行，通常在这里重置本次动作的局部计数器。
        """

        self.feedback_message = "initialised"

    def update(self) -> py_trees.common.Status:
        """每次 Tick 访问到该节点都会调用。

        条件节点通常立即返回 SUCCESS/FAILURE；动作节点可以在未完成时返回 RUNNING。
        """

        raise NotImplementedError

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """节点结束或被中断时调用。

        new_status 可能是 SUCCESS、FAILURE，也可能是 INVALID。INVALID 常见于父节点
        切换分支时，py_trees 会停止之前仍在 RUNNING 的子节点。
        """

        if new_status == py_trees.common.Status.INVALID:
            self.feedback_message = "interrupted -> INVALID"
