"""组装行为树。"""

from __future__ import annotations

import py_trees

from behaviours.conditions import (
    CheckBattery,
    CheckSensors,
    CheckTargetMemory,
    ValidateTargetLocation,
    VerifyDelivery,
    VerifyGrasp,
)
from behaviours.manipulation import DeliverObject, GraspObject
from behaviours.monitoring import MonitorBattery, MonitorObject
from behaviours.navigation import (
    ClearOldPath,
    NavigateToTarget,
    NavigateToUser,
    ReplanPath,
    SearchForObject,
)
from robot_state import ScenarioConfig, SimulatedRobotState, reset_blackboard


def create_delivery_tree(
    state: SimulatedRobotState,
    config: ScenarioConfig,
) -> py_trees.behaviour.Behaviour:
    """创建“寻找水杯并递送”的完整行为树。"""

    blackboard = reset_blackboard()
    blackboard.target_location = state.target_location
    blackboard.target_confidence = state.target_location_confidence
    blackboard.object_grasped = state.object_grasped

    system_ready = py_trees.composites.Sequence(
        name="系统准备",
        memory=False,
        children=[
            CheckBattery("检查电量", state, config, minimum_level=20.0),
            CheckSensors("检查传感器状态", state, config),
        ],
    )

    use_memory = py_trees.composites.Sequence(
        name="使用已有记忆",
        memory=False,
        children=[
            CheckTargetMemory("检查记忆中是否有目标位置", state, config),
            ValidateTargetLocation("验证目标位置是否可靠", state, config),
        ],
    )
    acquire_target = py_trees.composites.Selector(
        name="获得目标位置",
        memory=True,
        children=[
            use_memory,
            SearchForObject("搜索目标物体", state, config),
        ],
    )

    normal_navigation = NavigateToTarget(
        "正常导航",
        state,
        config,
        recovered=False,
        ticks_to_arrive=3,
    )
    recovery_navigation = py_trees.composites.Sequence(
        name="导航恢复",
        memory=True,
        children=[
            ClearOldPath("清除旧路径", state, config),
            ReplanPath("重新规划", state, config),
            NavigateToTarget("再次导航", state, config, recovered=True, ticks_to_arrive=2),
        ],
    )
    reach_target = py_trees.composites.Selector(
        name="到达目标附近",
        memory=True,
        children=[
            normal_navigation,
            recovery_navigation,
        ],
    )

    grasp_sequence = py_trees.composites.Sequence(
        name="抓取物体",
        memory=True,
        children=[
            GraspObject("执行抓取", state, config),
            VerifyGrasp("验证抓取结果", state, config),
        ],
    )
    retry_grasp = py_trees.decorators.Retry(
        name="有限次数重试抓取",
        child=grasp_sequence,
        num_failures=3,
    )

    delivery_monitor = py_trees.composites.Parallel(
        name="递送过程监控",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False),
        children=[
            NavigateToUser("导航到用户位置", state, config),
            MonitorBattery("监控电量", state, config, minimum_level=15.0),
            MonitorObject("监控物体是否仍被抓住", state, config),
        ],
    )

    complete_delivery = py_trees.composites.Sequence(
        name="完成交付",
        memory=True,
        children=[
            DeliverObject("放置或交付物体", state, config),
            VerifyDelivery("验证任务完成", state, config),
        ],
    )

    return py_trees.composites.Sequence(
        name="递送任务",
        memory=True,
        children=[
            system_ready,
            acquire_target,
            reach_target,
            retry_grasp,
            delivery_monitor,
            complete_delivery,
        ],
    )


def create_behaviour_tree(
    state: SimulatedRobotState,
    config: ScenarioConfig,
) -> py_trees.trees.BehaviourTree:
    """创建 BehaviourTree 包装器，方便 setup 和 tick。"""

    return py_trees.trees.BehaviourTree(create_delivery_tree(state, config))
