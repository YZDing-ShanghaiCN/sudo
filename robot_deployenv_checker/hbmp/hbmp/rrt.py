import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional, Dict

from typing import List, Tuple, Optional
import numpy as np
from enum import Enum

from rtree import index
from .base import Mop
import ampl


class RRTNode:
    def __init__(self, state: np.ndarray):
        self.state = state
        self.parent: Optional["RRTNode"] = None


class RRTTree:
    """Helper class to manage individual trees and their R-Tree indices."""

    def __init__(self, active_dim: int):
        p = index.Property()
        p.dimension = active_dim
        self.rtree_index = index.Index(properties=p)
        self.nodes: List[RRTNode] = []

    def _state_to_bbox(self, state: np.ndarray) -> tuple:
        return tuple(state) + tuple(state)

    def add_node(self, node: RRTNode):
        node_id = len(self.nodes)
        self.nodes.append(node)
        self.rtree_index.insert(node_id, self._state_to_bbox(node.state))

    def get_nearest(self, target_state: np.ndarray) -> RRTNode:
        nearest_id = list(
            self.rtree_index.nearest(self._state_to_bbox(target_state), 1)
        )[0]
        return self.nodes[nearest_id]


class ExtendStatus(Enum):
    ADVANCED = 1  # Successfully took a step toward the target
    REACHED = 2  # Successfully reached the exact target state
    TRAPPED = 3  # Hit an obstacle and could not move


class MopRRTC(Mop):
    def __init__(
        self, bounds, collision_fn, step_size=0.1, max_extend_dist=0.5, frozen_dofs=None
    ):
        super().__init__(bounds, collision_fn, step_size, frozen_dofs)
        self.max_extend_dist = max_extend_dist
        self.nb_shortcut_subd = 1

    def _extend(
        self, tree: RRTTree, target_state: np.ndarray
    ) -> Tuple[ExtendStatus, RRTNode]:
        """Takes a single step from the nearest node in the tree toward the target."""
        nearest_node = tree.get_nearest(target_state)
        direction = target_state - nearest_node.state
        dist = np.linalg.norm(direction)

        if dist > self.max_extend_dist:
            new_state = nearest_node.state + (direction / dist) * self.max_extend_dist
            status = ExtendStatus.ADVANCED
        else:
            new_state = target_state
            status = ExtendStatus.REACHED

        if self.is_path_valid(nearest_node.state, new_state):
            new_node = RRTNode(new_state)
            new_node.parent = nearest_node
            tree.add_node(new_node)
            return status, new_node

        return ExtendStatus.TRAPPED, nearest_node

    def _connect(
        self, tree: RRTTree, target_state: np.ndarray
    ) -> Tuple[ExtendStatus, RRTNode]:
        """Repeatedly extends the tree toward the target until reached or trapped."""
        while True:
            status, new_node = self._extend(tree, target_state)
            if status != ExtendStatus.ADVANCED:
                return status, new_node

    def _extract_path(
        self, node_a: RRTNode, node_b: RRTNode, is_tree_a_start: bool
    ) -> List[np.ndarray]:
        """Traces back through parents to reconstruct the full path."""

        def trace_to_root(node):
            path = []
            current = node
            while current is not None:
                path.append(current.state)
                current = current.parent
            return path

        path_a = trace_to_root(node_a)
        path_b = trace_to_root(node_b)

        if is_tree_a_start:
            active_path = path_a[::-1] + path_b[1:]
        else:
            active_path = path_b[::-1] + path_a[1:]

        return [self._get_full_state(state) for state in active_path]

    def plan(
        self, full_start: np.ndarray, full_goal: np.ndarray, max_iterations: int = 1000
    ) -> Optional[List[np.ndarray]]:
        start_active = self._extract_active_state(full_start)
        goal_active = self._extract_active_state(full_goal)

        if not self.is_state_valid(start_active) or not self.is_state_valid(
            goal_active
        ):
            print("Start or Goal is in collision!")
            return None
        if self.is_path_valid(start_active, goal_active):
            print("Direct path is free! Skipping tree generation.")
            return [
                self._get_full_state(start_active),
                self._get_full_state(goal_active),
            ]
        tree_a = RRTTree(self.active_dim)
        tree_b = RRTTree(self.active_dim)

        tree_a.add_node(RRTNode(start_active))
        tree_b.add_node(RRTNode(goal_active))

        is_tree_a_start = True

        for _ in range(max_iterations):
            q_rand = self.sample_active_state()

            # 1. Extend Tree A
            status_a, new_node_a = self._extend(tree_a, q_rand)

            if status_a != ExtendStatus.TRAPPED:
                # 2. Try to connect Tree B to the new node in Tree A
                status_b, new_node_b = self._connect(tree_b, new_node_a.state)

                if status_b == ExtendStatus.REACHED:
                    # Trees connected successfully!
                    return self._extract_path(new_node_a, new_node_b, is_tree_a_start)

            # 3. Swap trees
            tree_a, tree_b = tree_b, tree_a
            is_tree_a_start = not is_tree_a_start

        print("Failed to connect trees within max iterations.")
        return None

    def shortcut(self, waypoints_feasible: np.ndarray) -> Optional[List[np.ndarray]]:
        nb_subdivision_internal = self.nb_shortcut_subd

        active_traj = np.array(
            [self._extract_active_state(q) for q in waypoints_feasible]
        )

        graph, graph_waypoints = ampl.graph_from_polyline(
            active_traj, nb_subdivision_internal
        )
        nb_node = len(graph_waypoints)
        for i in range(0, nb_node - (nb_subdivision_internal + 1)):
            for j in range(i + (nb_subdivision_internal + 1), nb_node, 3):
                if self.is_path_valid(
                    graph_waypoints[i].copy(), graph_waypoints[j].copy()
                ):
                    wij = np.linalg.norm(
                        graph_waypoints[j].copy() - graph_waypoints[i].copy()
                    )
                    graph[i].append((j, wij))
        new_path_len, new_path = ampl.graph_dijkstra_single(
            graph, nb_node, 0, nb_node - 1
        )

        active_traj_shortcut = graph_waypoints[new_path].copy()

        print(
            "c-space length before =",
            np.linalg.norm(active_traj[:-1] - active_traj[1:], axis=1).sum(),
        )
        print(
            "c-space length after  =",
            np.linalg.norm(
                active_traj_shortcut[:-1] - active_traj_shortcut[1:], axis=1
            ).sum(),
        )

        return [self._get_full_state(state) for state in active_traj_shortcut]
