"""matplotlib-only visualizer for trajopt outputs.

Renders PNGs side-by-side with the trajopt JSON:
  * `*_iter.png`      — cost components vs solver iteration (log-y).
  * `*_qtraj.png`     — q_arm[k] (7 lines) vs t, warm-start vs optimized.
  * `*_cart.png`      — cartesian deviation per step, warm-start vs optimized.
  * `*_dist.png`      — min pair distance vs solver iter (collision audit).
  * `*_vel_accel.png` — joint |q_dot| and |q_ddot| vs time, warm vs optimized.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def render_iter_history_png(trajopt_json: str, out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load(trajopt_json)
    iters = data["solve_stats"]["iter_history"]
    if not iters:
        print(f"[visualize] no iter history in {trajopt_json}, skipping iter PNG")
        return
    keys = list(iters[0]["components"].keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [e["iter"] for e in iters]
    for k in keys:
        ys = [max(e["components"][k], 1e-12) for e in iters]
        ax.plot(xs, ys, label=k)
    total = [max(e["cost"], 1e-12) for e in iters]
    ax.plot(xs, total, "k--", label="total", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("iter")
    ax.set_ylabel("cost (log scale)")
    ax.set_title("trajopt cost components vs iteration")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")


def render_q_traj_png(
    trajopt_json: str,
    q_warm: np.ndarray,
    t_warm: np.ndarray,
    out_png: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load(trajopt_json)
    qt = data["q_trajectory"]
    t_final = np.array([e["t_global"] for e in qt], dtype=np.float64)
    q_final = np.array([e["q_arm"] for e in qt], dtype=np.float64)

    fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
    for j in range(7):
        ax = axes[j]
        ax.plot(t_warm, q_warm[:, j], "C0-", alpha=0.5, label="warm")
        ax.plot(t_final, q_final[:, j], "C3-", linewidth=1.2, label="opt")
        ax.set_ylabel(f"q[{j}]")
        ax.grid(True, alpha=0.3)
    axes[0].set_title("q_arm vs time — warmstart (blue) vs trajopt (red)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")


def render_min_distance_png(trajopt_json: str, out_png: str) -> None:
    """Plot `min_pair_distance` over solver iterations.

    Each iter logs `min_pair_distance` = sphere-BVH SDF minimum across the
    whole trajectory at that iter's `x`. The hard `d_safe` floor is drawn as
    a dotted line; the run is feasible iff every iter's line sits above it.
    `null` entries (no pair within `d_warn`) are plotted as the `d_warn`
    ceiling so the trace stays connected.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load(trajopt_json)
    iters = data["solve_stats"].get("iter_history", []) or []
    if not iters:
        # IPOPT path: no callback table, just print the post-solve audit.
        print(f"[visualize] no iter history in {trajopt_json}, skipping dist PNG")
        return
    d_safe = float(data["solve_stats"].get("d_safe_used", 0.005))
    d_warn = float(data["solve_stats"].get("d_warn_used", 0.050))
    xs = [e["iter"] for e in iters]
    ys = [
        d_warn if e.get("min_pair_distance") is None else float(e["min_pair_distance"])
        for e in iters
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, "C3-", label="min pair distance")
    ax.axhline(d_safe, color="k", linestyle=":", alpha=0.6, label=f"d_safe ({d_safe:.4f} m)")
    ax.axhline(d_warn, color="0.6", linestyle="--", alpha=0.5, label=f"d_warn ({d_warn:.4f} m)")
    final = data["solve_stats"].get("min_pair_distance_after")
    if final is not None:
        ax.axhline(float(final), color="C0", linestyle="-.", alpha=0.5,
                   label=f"min_d post-solve ({float(final):.4f} m)")
    ax.set_xlabel("iter")
    ax.set_ylabel("min pair signed distance (m)")
    ax.set_title("min pair distance vs iter — d_safe is the hard floor")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")


def render_cart_deviation_png(
    trajopt_json: str,
    q_warm: np.ndarray,
    t_warm: np.ndarray,
    target_pos: np.ndarray,
    out_png: str,
) -> None:
    """Plot per-step per-axis cartesian deviation (|Δx|, |Δy|, |Δz|, L2).

    The per-axis box now caps drift on each axis independently, so the
    diagnostic plot leads with the per-axis breakdown — `|Δz|` is the
    safety-critical curve for surface-cover plans. The L2 magnitude is
    plotted in the background for cross-run comparison with the pre-change
    runs that only logged L2.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load(trajopt_json)
    qt = data["q_trajectory"]
    t_final = np.array([e["t_global"] for e in qt], dtype=np.float64)
    cart_final = np.array([e["cartesian_deviation"] for e in qt], dtype=np.float64)
    # Old trajopt JSONs (pre-per-axis change) won't have cartesian_deviation_axes.
    # Fall back to deriving from achieved - target so this viz keeps working
    # against historical outputs.
    if qt and "cartesian_deviation_axes" in qt[0]:
        axes_final = np.array(
            [e["cartesian_deviation_axes"] for e in qt], dtype=np.float64
        )
    else:
        achieved = np.array([e["achieved_position"] for e in qt], dtype=np.float64)
        target = np.array([e["target_position"] for e in qt], dtype=np.float64)
        axes_final = achieved - target
    abs_axes = np.abs(axes_final)

    bounds_md = data.get("metadata", {}).get("bounds", {}) or {}
    cap_xy = bounds_md.get("max_xy_deviation")
    cap_z = bounds_md.get("max_z_deviation")
    # Effective caps used during the solve (may be auto-relaxed above the
    # config value if the warmstart violated them).
    used = data.get("solve_stats", {}).get("cart_cap_used")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t_final, abs_axes[:, 0], "C0-", label="|Δx|", linewidth=1.1)
    ax.plot(t_final, abs_axes[:, 1], "C2-", label="|Δy|", linewidth=1.1)
    ax.plot(t_final, abs_axes[:, 2], "C3-", label="|Δz|", linewidth=1.6)
    ax.plot(t_final, cart_final, "0.4", alpha=0.4, label="L2 norm", linewidth=0.9)
    if cap_xy is not None:
        ax.axhline(float(cap_xy), color="C0", linestyle=":", alpha=0.6,
                   label=f"max_xy_deviation ({float(cap_xy):.4f} m)")
    if cap_z is not None:
        ax.axhline(float(cap_z), color="C3", linestyle=":", alpha=0.6,
                   label=f"max_z_deviation ({float(cap_z):.4f} m)")
    if isinstance(used, dict):
        used_z = float(used.get("z", 0.0))
        cfg_z = float(cap_z) if cap_z is not None else used_z
        if used_z > cfg_z * 1.001:
            ax.axhline(used_z, color="k", linestyle="-.", alpha=0.5,
                       label=f"z-cap used ({used_z:.4f} m, auto-relaxed)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|deviation| (m)")
    ax.set_title("per-step cartesian deviation (per-axis box)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")


def render_vel_accel_png(
    q_warm: np.ndarray,
    dt_warm: np.ndarray,
    q_opt: np.ndarray,
    dt_opt: np.ndarray | float,
    v_max: np.ndarray,
    out_png: str,
) -> None:
    """Plot joint |q_dot| and |q_ddot| vs absolute time, warm vs optimized.

    Top panel: |joint velocity| per DOF with a horizontal `v_max` line
    (dashed). The optimized curve should saturate this line on segments
    where the warmstart had slack; that's the speedup the freed dt_lo
    recovers.
    Bottom panel: |joint acceleration|. No hard cap — informational.

    Both curves are plotted in their own absolute-time frame:
      - warm uses variable `dt_warm[k]`; finite-difference at segment
        midpoints / inner-knot times.
      - opt may be uniform-dt (densified path; pass scalar) or variable-dt
        (sparse SCO output; pass an (N-1,) array).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _vel_accel(q: np.ndarray, dt: np.ndarray):
        """Return (t_vel, qd, t_accel, qdd) for a (N, 7) joint trajectory.

        Forward-diff vel at segment midpoints; central-diff accel using the
        average of adjacent dt as the denominator (correct for variable dt).
        """
        n = q.shape[0]
        t_knot = np.concatenate([[0.0], np.cumsum(dt)])         # (N,)
        qd = (q[1:] - q[:-1]) / dt[:, None]                      # (N-1, 7)
        t_vel = 0.5 * (t_knot[:-1] + t_knot[1:])                 # (N-1,)
        if n >= 3:
            dt_avg = 0.5 * (dt[1:] + dt[:-1])                    # (N-2,)
            qdd = (qd[1:] - qd[:-1]) / dt_avg[:, None]           # (N-2, 7)
            t_accel = t_knot[1:-1]                               # (N-2,)
        else:
            qdd = np.zeros((0, q.shape[1]))
            t_accel = np.zeros(0)
        return t_vel, qd, t_accel, qdd

    dt_warm = np.asarray(dt_warm, dtype=np.float64)
    if np.isscalar(dt_opt):
        dt_opt_arr = np.full(q_opt.shape[0] - 1, float(dt_opt), dtype=np.float64)
    else:
        dt_opt_arr = np.asarray(dt_opt, dtype=np.float64)

    t_v_w, qd_w, t_a_w, qdd_w = _vel_accel(q_warm, dt_warm)
    t_v_o, qd_o, t_a_o, qdd_o = _vel_accel(q_opt, dt_opt_arr)

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for j in range(7):
        w_label = "warm" if j == 0 else None
        o_label = "opt" if j == 0 else None
        ax_v.plot(t_v_w, np.abs(qd_w[:, j]), "C0-", alpha=0.5, linewidth=1.0,
                  label=w_label)
        ax_v.plot(t_v_o, np.abs(qd_o[:, j]), "C3-", linewidth=1.0,
                  label=o_label)
        if t_a_w.size:
            ax_a.plot(t_a_w, np.abs(qdd_w[:, j]), "C0-", alpha=0.5,
                      linewidth=1.0, label="warm" if j == 0 else None)
        if t_a_o.size:
            ax_a.plot(t_a_o, np.abs(qdd_o[:, j]), "C3-", linewidth=1.0,
                      label="opt" if j == 0 else None)

    v_max = np.asarray(v_max, dtype=np.float64).reshape(-1)
    if np.allclose(v_max, v_max[0]):
        ax_v.axhline(float(v_max[0]), color="k", linestyle="--", alpha=0.6,
                     label=f"v_max ({float(v_max[0]):.2f} rad/s)")
    else:
        for j in range(7):
            ax_v.axhline(float(v_max[j]), color=f"C{j}", linestyle="--",
                         alpha=0.3)

    ax_v.set_ylabel("|q_dot[j]| (rad/s)")
    ax_v.set_title("joint velocity / acceleration — warm (blue) vs optimized (red)")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend(loc="upper right", fontsize=8)
    ax_a.set_ylabel("|q_ddot[j]| (rad/s²)")
    ax_a.set_xlabel("t (s)")
    ax_a.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")
