"""Trajectory-optimization post-processor for `surface_cover_planner`.

Takes a feasible warm-start q-trajectory (produced by the planner pipeline,
either from disk or directly from a `robot_check.CheckResult`) and refines
it via sequential convex optimization: minimize total time + joint
accel-as-jerk-proxy + cartesian tracking error under hard joint-pos /
joint-vel / cartesian-deviation / self-collision constraints. Default
solver is IPOPT (via `cyipopt`); falls back to scipy's `trust-constr`
when cyipopt isn't installed.

Public API:
    solve            — run the SCO over a WarmStart bundle.
    SolverOpts       — solver method, tol, max_iters, verbose.
    OptWeights       — weighted-sum cost coefficients.
    OptBounds        — per-step trust radius, cart caps, endpoint pin, +
                       embedded collision SDF margins (d_safe / d_warn / top_k).
    SolverResult     — q_arm + dt + achieved pose + cart_dev + solve_stats.
    WarmStart        — input bundle for `solve()`; build via
                       `WarmStart.from_planner_result(...)` (in-memory) or
                       `load_warmstart(<json_path>)` (from disk).
    load_warmstart   — parse a surface_cover_planner output JSON.
    write_optimizer_json — write the optimizer's own output JSON.
    RobotModel / RobotConfig — `ampl_motion`-backed FK + SDF wrapper.
"""

from __future__ import annotations

from .io import WarmStart, load_warmstart, write_optimizer_json
from .local_dp import solve_local_dp
from .robot_model import RobotConfig, RobotModel
from .solver import (
    OptBounds,
    OptWeights,
    SolverOpts,
    SolverResult,
    solve,
)

__all__ = [
    "OptBounds",
    "OptWeights",
    "RobotConfig",
    "RobotModel",
    "SolverOpts",
    "SolverResult",
    "WarmStart",
    "load_warmstart",
    "solve",
    "solve_local_dp",
    "write_optimizer_json",
]
