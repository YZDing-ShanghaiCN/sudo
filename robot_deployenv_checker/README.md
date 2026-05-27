# Robot Deploy Environment Checker

Interactive tool for verifying robot task viability before deployment. Checks **reachability** (can the robot reach the workspace?) and **visibility** (can cameras see the workspace?) using a browser-based 3D visualization.

Built on [Viser](https://github.com/nerfstudio-project/viser) with the `hbmp` motion planning library for IK solving and collision detection.

## Features

- Config-driven scene: define **N robots**, objects, cameras, and workspace bounds in a single YAML
- Each robot has a **draggable base gizmo** — move the whole robot in the scene; URDF, EEFs, and camera frustums follow
- Drag EEF gizmos to test reachability via IK tracking (WBC-based)
- Real-time self-collision detection with per-robot visual feedback
- 8 robot-mounted cameras *per robot* with frustum visualization
- On-demand camera rendering from any mounted camera viewpoint
- Workspace bounding box with adjustable wall constraints
- **Trajectory replay**: load a planner JSON (e.g. from `surface_cover_planner`) and play it back in real time on the first robot

## Requirements

- Linux x86_64 (the `ampl` and `pywbc` wheels are platform-specific)
- Python 3.10 (required by pre-built wheels)
- [uv](https://docs.astral.sh/uv/) (recommended for venv setup)

## Setup

```bash
cd robot_deployenv_checker

# Create Python 3.10 venv
uv venv --python 3.10 .venv
source .venv/bin/activate

# Install pre-built motion planning wheels
uv pip install hbmp/packages/ampl-0.0.25-cp310-cp310-linux_x86_64.whl
uv pip install hbmp/packages/wbc_py-0.2.3-cp310-cp310-linux_x86_64.whl

# Install runtime dependencies
uv pip install viser yourdfpy trimesh numpy pyyaml dacite

# (Optional) Install test dependencies
uv pip install pytest
```

## Usage

```bash
source .venv/bin/activate

# Single-robot scene
PYTHONPATH=hbmp:src python -m deployenv_checker --config configs/example_scene.yaml

# Three-robot scene around the zongzhuang object
PYTHONPATH=hbmp:src python -m deployenv_checker --config configs/three_robots_scene.yaml
```

Open http://localhost:8080 in a browser.

### Controls

1. **Base gizmo** (per robot) -- drag the gizmo at each robot's base to reposition/rotate the entire robot. URDF, EEF gizmos, and camera frustums follow.
2. **Go to Target** button (per robot) -- snap that robot's joints to its current EEF gizmo poses via IK.
3. **Wall sliders** (per robot) -- adjust the WBC workspace wall (y-max, z-min/max).
4. **Camera renders** (per robot) -- expand a camera group folder, click "Render" to capture a snapshot from that camera's viewpoint. "Render All Cameras" captures all 8.
5. **Show Frustums** (per robot) -- toggle camera frustum wireframes in the scene.
6. **View mode** (global) -- switch between visual, collision, or both mesh displays for all robots.
7. **Trajectory** (global) -- replay a planner-generated joint trajectory:
   - **Load JSON** -- upload a trajectory file (expects `q_trajectory: [{ t_global, q_full[16], ... }, ...]` in the same DOF layout as the T2DA2 robot, e.g. produced by `tools/surface_cover_planner`). Frames where `q_full` is `null` (failed IK) are skipped.
   - **Play / Pause / Stop** -- play in real time using `t_global`, pause to freeze mid-replay, stop to snap back to frame 0. The trajectory drives the **first robot** in the config.

### Custom scene config

Create a YAML file (see `configs/example_scene.yaml` and `configs/three_robots_scene.yaml`):

```yaml
scene:
  name: "My Workspace Check"

robots:
  - name: "hb11"                # unique per-robot id, used as scene path & GUI label
    type: "t2da2"
    urdf_visual: "./hbmp/assets/hb11/urdf_c.urdf"
    urdf_collision: "./hbmp/assets/hb11/urdf_c.urdf"
    position: [0.0, 0.0, 0.0]   # initial base position; draggable in browser
    wxyz: [1.0, 0.0, 0.0, 0.0]
    scale: 0.25
    initial_q: [0.15, 0.3, 0.8, 0.64, 1.5, -1.65, -0.8, -0.8, 0.6,
                0.8, 0.64, 1.5, -1.65, -0.8, -0.8, 0.6]
    end_effectors:
      - name: "left_tool0"
        frame: "FRAME_TACTILE_L"
        mesh_path: "./hbmp/assets/gripper/meshes/visual/gripper.glb"
        position: [0.707, 0.370, 0.921]
        wxyz: [0.372, 0.573, 0.512, 0.520]
        scale: 0.15
    params:
      wbc_config: "./hbmp/wbc_config_hb.yaml"
      ndof: 16
    cameras:                    # cameras live under each robot
      config_path: "./configs/camera_config.json"
      show_frustums: true
      frustum_scale: 0.1

  # Add more robots by extending the list. Each gets its own base gizmo,
  # GUI folder, and camera frustums.
  # - name: "hb11_b"
  #   ...

objects:
  - name: "workbench"
    mesh_path: "./path/to/mesh.glb"
    position: [0.8, 0.0, 0.4]
    draggable: true

workspace:
  bounds: [0.0, 1.25, -1.0, 1.0, 0.6, 1.8]
  show_bounds: true
```

The legacy single-`robot:` schema (with top-level `cameras:`) is still accepted by the loader for back-compat.

## Tests

```bash
source .venv/bin/activate
PYTHONPATH=hbmp:src python -m pytest tests/ -v
```

## Project structure

```
robot_deployenv_checker/
├── configs/
│   ├── camera_config.json       # 8-camera calibration (intrinsics + extrinsics)
│   └── example_scene.yaml       # Example scene config
├── hbmp/                        # Motion planning library (dependency)
│   ├── hbmp/                    # Python package (Robot_T2DA2, IK, collision)
│   ├── assets/                  # URDF, meshes
│   └── packages/                # Pre-built wheels (ampl, pywbc)
├── src/deployenv_checker/       # Main application
│   ├── __main__.py              # CLI entry point
│   ├── app.py                   # Orchestrator + mode management
│   ├── config.py                # Dataclass config models + loaders
│   ├── scene.py                 # Viser scene manager
│   ├── robot.py                 # Robot wrapper (IK tracking, FK queries)
│   ├── camera.py                # Camera manager (frustums, rendering)
│   └── gui.py                   # GUI panels
├── tests/
│   └── test_hbmp.py             # Unit tests for hbmp
└── pyproject.toml
```
