"""基础行为测试。

测试只验证行为树逻辑，不等待真实时间，也不连接任何硬件。
"""

from __future__ import annotations

import py_trees

from behaviours.conditions import CheckBattery, VerifyGrasp
from behaviours.manipulation import GraspObject
from behaviours.navigation import NavigateToTarget, SearchForObject
from main import run_scenario
from robot_state import ScenarioConfig, create_state, reset_blackboard


def seed_target(location: str = "kitchen_table", confidence: float = 0.9) -> None:
    client = py_trees.blackboard.Client(name="test_seed")
    client.register_key("target_location", py_trees.common.Access.WRITE)
    client.register_key("target_confidence", py_trees.common.Access.WRITE)
    client.target_location = location
    client.target_confidence = confidence


def test_check_battery_success_when_level_is_enough() -> None:
    config = ScenarioConfig(name="normal")
    state = create_state(config)
    reset_blackboard()
    node = CheckBattery("检查电量", state, config, minimum_level=20.0)

    node.tick_once()

    assert node.status == py_trees.common.Status.SUCCESS


def test_check_battery_failure_when_level_is_low() -> None:
    config = ScenarioConfig(name="low-battery", initial_battery=10.0)
    state = create_state(config)
    reset_blackboard()
    node = CheckBattery("检查电量", state, config, minimum_level=20.0)

    node.tick_once()

    assert node.status == py_trees.common.Status.FAILURE


def test_navigation_returns_running_before_arrival() -> None:
    config = ScenarioConfig(name="normal", target_known=True, target_location_confidence=0.9)
    state = create_state(config)
    reset_blackboard()
    seed_target()
    node = NavigateToTarget("正常导航", state, config, ticks_to_arrive=3)

    node.tick_once()

    assert node.status == py_trees.common.Status.RUNNING


def test_navigation_returns_success_after_arrival() -> None:
    config = ScenarioConfig(name="normal", target_known=True, target_location_confidence=0.9)
    state = create_state(config)
    reset_blackboard()
    seed_target()
    node = NavigateToTarget("正常导航", state, config, ticks_to_arrive=2)

    node.tick_once()
    node.tick_once()

    assert node.status == py_trees.common.Status.SUCCESS


def test_search_writes_target_to_blackboard() -> None:
    config = ScenarioConfig(name="recovery")
    state = create_state(config)
    reset_blackboard()
    node = SearchForObject("搜索目标物体", state, config, ticks_to_find=2)

    node.tick_once()
    node.tick_once()

    reader = py_trees.blackboard.Client(name="test_reader")
    reader.register_key("target_location", py_trees.common.Access.READ)
    reader.register_key("target_confidence", py_trees.common.Access.READ)
    assert node.status == py_trees.common.Status.SUCCESS
    assert reader.target_location == "kitchen_table"
    assert reader.target_confidence >= 0.75


def test_grasp_can_retry_after_first_failure() -> None:
    config = ScenarioConfig(name="normal", grasp_failures_before_success=1)
    state = create_state(config)
    reset_blackboard()
    grasp_sequence = py_trees.composites.Sequence(
        name="抓取物体",
        memory=True,
        children=[
            GraspObject("执行抓取", state, config),
            VerifyGrasp("验证抓取结果", state, config),
        ],
    )
    retry = py_trees.decorators.Retry(
        name="有限次数重试抓取",
        child=grasp_sequence,
        num_failures=3,
    )

    retry.tick_once()
    assert retry.status == py_trees.common.Status.RUNNING
    retry.tick_once()

    assert retry.status == py_trees.common.Status.SUCCESS
    assert state.grasp_attempts == 2


def test_recovery_scenario_eventually_succeeds() -> None:
    final_status, state = run_scenario(
        "recovery",
        max_ticks=40,
        tick_period=0.0,
        print_output=False,
    )

    assert final_status == py_trees.common.Status.SUCCESS
    assert state["delivery_completed"] is True
    assert state["path_replanned"] is True


def test_low_battery_scenario_fails_immediately() -> None:
    final_status, state = run_scenario(
        "low-battery",
        max_ticks=5,
        tick_period=0.0,
        print_output=False,
    )

    assert final_status == py_trees.common.Status.FAILURE
    assert state["delivery_completed"] is False
