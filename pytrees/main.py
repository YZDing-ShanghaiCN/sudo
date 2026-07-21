"""命令行入口：运行不同演示场景并打印每次 Tick 的树状态。"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from pprint import pformat

import py_trees

from robot_state import ScenarioName, config_for_scenario, create_state
from tree_builder import create_behaviour_tree


BLACKBOARD_KEYS = (
    "target_location",
    "target_confidence",
    "navigation_failure_reason",
    "recovery_count",
    "object_grasped",
    "task_progress",
)


def iter_nodes(root: py_trees.behaviour.Behaviour) -> Iterable[py_trees.behaviour.Behaviour]:
    """深度优先遍历行为树节点，用于打印每个节点的 status 和 feedback。"""

    yield root
    for child in getattr(root, "children", []):
        yield from iter_nodes(child)


def read_blackboard_snapshot() -> dict[str, object]:
    """读取关键 Blackboard 数据。"""

    client = py_trees.blackboard.Client(name="printer")
    for key in BLACKBOARD_KEYS:
        client.register_key(key, py_trees.common.Access.READ)

    return {key: getattr(client, key) for key in BLACKBOARD_KEYS}


def print_tick_snapshot(
    tick_number: int,
    tree: py_trees.trees.BehaviourTree,
) -> None:
    """打印一次 Tick 后的所有观察信息。"""

    root = tree.root
    print(f"\n========== TICK {tick_number} ==========\n")
    print(f"Root Status: {root.status.name}")
    print("\nTree:")
    print(py_trees.display.unicode_tree(root, show_status=True))

    print("\nNode Status / Feedback:")
    for node in iter_nodes(root):
        feedback = node.feedback_message or "-"
        print(f"- {node.name}: {node.status.name} | {feedback}")

    print("\nBlackboard:")
    for key, value in read_blackboard_snapshot().items():
        print(f"{key} = {value}")


def run_scenario(
    scenario: ScenarioName,
    *,
    max_ticks: int = 40,
    tick_period: float = 0.5,
    print_output: bool = True,
) -> tuple[py_trees.common.Status, dict[str, object]]:
    """运行一个场景，返回根节点最终状态和机器人状态快照。"""

    config = config_for_scenario(scenario)
    state = create_state(config)
    tree = create_behaviour_tree(state, config)
    tree.setup(timeout=15)

    final_status = py_trees.common.Status.INVALID
    for tick_number in range(1, max_ticks + 1):
        state.tick_count = tick_number
        tree.tick()
        final_status = tree.root.status

        if print_output:
            print_tick_snapshot(tick_number, tree)
            print("\nRobot State:")
            print(pformat(state.as_dict(), sort_dicts=False))

        if final_status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
            break

        if tick_period > 0:
            time.sleep(tick_period)

    if print_output:
        print("\n========== FINAL RESULT ==========")
        print(f"Scenario: {scenario}")
        print(f"Result: {final_status.name}")

    return final_status, state.as_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="纯 Python py_trees 行为树学习项目")
    parser.add_argument(
        "--scenario",
        choices=("normal", "recovery", "low-battery", "grasp-failure", "object-dropped"),
        default="recovery",
        help="选择演示场景",
    )
    parser.add_argument("--max-ticks", type=int, default=40, help="最大 Tick 数")
    parser.add_argument("--tick-period", type=float, default=0.5, help="每次 Tick 间隔秒数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_scenario(
        args.scenario,
        max_ticks=args.max_ticks,
        tick_period=args.tick_period,
        print_output=True,
    )


if __name__ == "__main__":
    main()
