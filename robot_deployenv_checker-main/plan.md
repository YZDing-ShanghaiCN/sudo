# Robot Deploy Environment Checker — Implementation Plan

## Context

We're building a tool to check if a robot task is workable by verifying **reachability** (can the robot reach the workspace?) and **visibility** (can cameras see the workspace?). The user has an existing `hbmp/` codebase with Viser-based visualization, URDF loading, IK solving (AMPL + WBC), and collision detection. This tool wraps and generalizes that into a config-driven, interactive checker.

**What exists in `hbmp/`:**
- `scene_manager.py` — `ConfigSceneManager` with Viser: object spawning, gizmos, display modes, save/load
- `main_viser.py` — Full integration: Viser server, robot creation, gizmo→IK loop, GUI panels
- `hbmp/` package — `RobotInterface`, `Kin`, `Col`, `Robot_T2DA2`, WBC, RRT, collision primitives
- `assets/` — HB11 URDF, gripper meshes
- `packages/` — Pre-built wheels for `ampl` and `pywbc`
- `configs/camera_config.json` — Real camera calibration data (8 cameras)

**What's missing / needs to change:**
- No camera visibility system (frustums, on-demand rendering)
- Hardcoded to HB11/T2DA2 — needs config-driven robot+scene loading
- No unified config format — scene JSON + WBC YAML + hardcoded params scattered in main
- `main_viser.py` mixes concerns (modes, planning, slerp) — checker tool is simpler and focused

---

## Camera System — Critical Design Note

Cameras are **mounted on robot links**, not fixed in world space. From `configs/camera_config.json`:
- 2 chest cameras → mounted on `link_torso_2`
- 3 left hand cameras → mounted on `left-link_arm_7`
- 3 right hand cameras → mounted on `right-link_arm_7`

Each camera has:
- **`mount`**: Robot link name the camera is attached to
- **`extrinsics`**: 4x4 transform matrix (camera pose relative to mount link)
- **`intrinsics`**: 3x3 K matrix (fx, fy, cx, cy)
- **`width`/`height`**: Image resolution (1280x800)

**Implication**: Camera world poses must be recomputed every time the robot moves (FK of mount link × extrinsics). Frustum visualization must update with the robot. On-demand rendering must use the current camera world pose.

For rendering, Viser's `client.get_render()` uses FoV, not intrinsics. We need to convert:
- `fov_y = 2 * atan(height / (2 * fy))`
- `fov_x = 2 * atan(width / (2 * fx))`
- Use vertical FoV for Viser's render call

Note: `client.get_render()` does not support non-square pixels (fx ≠ fy) or off-center principal points (cx ≠ w/2). For approximate visualization this is fine. For pixel-accurate rendering, a custom OpenGL/offscreen renderer would be needed later.

---

## Architecture

```
robot_deployenv_checker/
├── pyproject.toml
├── plan.md
├── configs/
│   ├── camera_config.json         # Real camera calibration (existing)
│   └── example_scene.yaml         # Full scene config
├── src/
│   └── deployenv_checker/
│       ├── __init__.py
│       ├── __main__.py            # CLI entry: python -m deployenv_checker --config scene.yaml
│       ├── app.py                 # Main orchestrator
│       ├── config.py              # Dataclass models + YAML loader
│       ├── scene.py               # Scene manager (evolved from hbmp/scene_manager.py)
│       ├── robot.py               # Robot wrapper: registry + IK controller
│       ├── camera.py              # Camera manager: frustums + on-demand rendering
│       └── gui.py                 # GUI panel builder (buttons, camera views)
├── hbmp/                          # Existing code (used as dependency, minimal changes)
│   └── ...
└── tests/
    └── ...
```

**Key design decisions:**
- `hbmp/` stays mostly unchanged — we wrap it, not rewrite it
- Config-driven: one YAML defines robot, environment objects; camera config loaded from its JSON
- `app.py` is the orchestrator wiring config → scene → robot → cameras → GUI
- Camera rendering uses Viser's `client.get_render()` for on-demand snapshots
- Camera poses are dynamic — recomputed from FK(mount_link) × extrinsics on each robot state change

---

## Config Schema (YAML)

```yaml
scene:
  name: "HB11 Workspace Check"

robot:
  type: "t2da2"
  urdf_visual: "./assets/hb11/urdf_c.urdf"
  urdf_collision: "./assets/hb11/urdf_c.urdf"
  position: [0.0, 0.0, 0.0]
  wxyz: [1.0, 0.0, 0.0, 0.0]
  scale: 0.25
  initial_q: [0.15, 0.3, 0.8, 0.64, 1.5, -1.65, -0.8, -0.8, 0.6,
              0.8, 0.64, 1.5, -1.65, -0.8, -0.8, 0.6]
  end_effectors:
    - name: "left_tool0"
      frame: "FRAME_TACTILE_L"
      mesh_path: "./assets/gripper/meshes/visual/gripper.glb"
      position: [0.707, 0.370, 0.921]
      wxyz: [0.372, 0.573, 0.512, 0.520]
      scale: 0.15
    - name: "right_tool0"
      frame: "FRAME_TACTILE_R"
      mesh_path: "./assets/gripper/meshes/visual/gripper.glb"
      position: [0.702, -0.358, 0.941]
      wxyz: [0.395, 0.620, 0.542, 0.408]
      scale: 0.15
  params:
    wbc_config: "wbc_config_hb.yaml"
    ndof: 16

cameras:
  config_path: "./configs/camera_config.json"   # Load from existing calibration file
  show_frustums: true                            # Display frustum wireframes in scene
  frustum_scale: 0.1                             # Frustum visual size

objects:
  - name: "workbench"
    mesh_path: "./assets/env/workbench.glb"
    position: [0.8, 0.0, 0.4]
    wxyz: [1.0, 0.0, 0.0, 0.0]
    scale: 1.0
    draggable: true

workspace:
  bounds: [0.0, 1.25, -1.0, 1.0, 0.6, 1.8]
  show_bounds: true
```

---

## Module Details

### 1. `config.py` — Config Models + Loader
- Dataclass models: `SceneConfig`, `RobotConfig`, `EEFConfig`, `ObjectConfig`, `CameraSystemConfig`, `CameraConfig`, `WorkspaceConfig`
- `CameraConfig` holds: name, mount link, extrinsics (4x4), intrinsics (3x3), width, height
- `load_config(path) -> SceneConfig`: YAML loading + camera JSON loading + path resolution
- Helper: `intrinsics_to_fov(K, w, h) -> (fov_x, fov_y)` for Viser rendering

### 2. `scene.py` — Scene Manager
- Evolved from `hbmp/scene_manager.py`
- Manages Viser server, spawns objects with gizmos, display mode toggling
- Key change: driven by `SceneConfig` instead of JSON + hardcoded setup
- Reuses: `ConfigSceneManager` patterns (spawn, pose get/set, display modes)

### 3. `robot.py` — Robot Wrapper
- Registry mapping type string → constructor (e.g., `"t2da2"` → `Robot_T2DA2`)
- IK controller: registers gizmo `on_update` callbacks → calls `robot.track_tcp()` → updates scene
- Exposes `get_link_pose(link_name) -> np.ndarray(4x4)` for camera system to query mount link poses
- Reuses: `hbmp.Robot_T2DA2`, `hbmp.RobotInterface` directly

### 4. `camera.py` — Camera Manager (NEW — most complex new module)

**Core responsibilities:**
- Load camera configs from JSON (intrinsics, extrinsics, mount links)
- Compute camera world poses: `T_world_camera = T_world_link(FK) @ T_link_camera(extrinsics)`
- Add Viser camera frustums (`server.scene.add_camera_frustum`) for each camera
- Update frustum positions when robot moves (called after each IK solve)
- On-demand rendering: `render_camera(client, name) -> np.ndarray`
  - Get current world pose of camera
  - Convert intrinsics to FoV
  - Call `client.get_render(height, width, wxyz, position, fov)`
- `render_all(client) -> dict[str, np.ndarray]`

**Camera grouping for UI:**
- Chest cameras (move with torso)
- Left hand cameras (move with left arm)
- Right hand cameras (move with right arm)

### 5. `gui.py` — GUI Panels
- **Reachability section**: collision status indicator, wall sliders
- **Camera section**:
  - Camera group folders (Chest / Left Hand / Right Hand)
  - Per-camera "Render" button → shows snapshot image in panel
  - "Render All" button → shows all camera views at once
  - Toggle frustum visibility
- **Scene section**: display mode radio, save/load buttons

### 6. `app.py` — Orchestrator
- Load config → create Viser server → build scene → init robot → setup cameras → build GUI
- Main loop:
  1. When gizmo moves → IK solve → update robot viz
  2. After robot update → recompute camera world poses → update frustums
  3. When "Render" clicked → render from current camera pose → display image

### 7. `__main__.py` — CLI Entry
- `python -m deployenv_checker --config configs/example_scene.yaml`
- Argparse with `--config` and optional `--port`

---

## Development Phases

### Phase 1: Foundation
**Tasks (parallelizable):**
- **1a.** `pyproject.toml` setup — dependencies: viser, yourdfpy, trimesh, numpy, pyyaml, dacite
- **1b.** `config.py` — Dataclass models + YAML/JSON loader + intrinsics→FoV helper
- **1c.** Example `configs/example_scene.yaml` for HB11

**Verification:** Load example config, verify all fields parsed, camera intrinsics converted to FoV correctly.

### Phase 2: Scene + Robot
**Tasks (2a and 2b parallelizable):**
- **2a.** `scene.py` — Config-driven scene manager (refactored from `hbmp/scene_manager.py`)
- **2b.** `robot.py` — Robot registry + IK controller + link pose queries
- **2c.** `app.py` (v1) — Wire config → scene → robot, get EEF dragging + IK working

**Verification:** Launch app, drag EEF gizmo, robot follows with IK. Collision feedback visible.

### Phase 3: Camera System
**Tasks:**
- **3a.** `camera.py` — Frustum display + world pose computation + on-demand rendering
- **3b.** `gui.py` — Camera group panels, render buttons, image display
- **3c.** Wire into `app.py` — frustum updates after IK, render on button click

**Verification:** Drag EEF → hand camera frustums move with arm. Click "Render" → see camera snapshot. Move robot → re-render → view changes.

### Phase 4: Polish
**Tasks:**
- **4a.** Workspace bounds visualization
- **4b.** Save/load scene state
- **4c.** `__main__.py` CLI entry point
- **4d.** README

**Verification:** `python -m deployenv_checker --config configs/example_scene.yaml` → full end-to-end works.

---

## Parallel Agent Distribution (for execution)

- **Agent A**: Phase 1 — `pyproject.toml` + `config.py` + example YAML
- **Agent B**: Phase 2a — `scene.py` (needs to read `hbmp/scene_manager.py`)
- **Agent C**: Phase 2b — `robot.py` (needs to read `hbmp/hbmp/base.py`, `impl_t2da2.py`)
- **Agent D**: Phase 3a — `camera.py` (needs `configs/camera_config.json`, Viser frustum API)

After parallel work → integrate in `app.py` + `gui.py` sequentially.

---

## Key Files to Reference

| File | Role |
|------|------|
| `hbmp/scene_manager.py` | Base for `scene.py` refactoring |
| `hbmp/main_viser.py` | Reference for app flow, gizmo→IK wiring, GUI patterns |
| `hbmp/hbmp/base.py` | `RobotInterface` abstract class — integration contract |
| `hbmp/hbmp/impl_t2da2.py` | Concrete robot implementation to wrap |
| `hbmp/scene_config_hb.json` | Reference for scene object data |
| `hbmp/wbc_config_hb.yaml` | WBC config — stays as-is, referenced by robot config |
| `configs/camera_config.json` | Real camera calibration (8 cameras, mounted on links) |

---

## Open Questions

1. **Platform**: `ampl` and `pywbc` wheels are `linux_x86_64` only. Will this tool run on Linux only, or do we need macOS support?
2. **Additional robot types**: Only T2DA2 exists now. Design registry for extensibility, or T2DA2-only for now?
3. **Depth rendering**: RGB snapshot sufficient, or also need depth maps?
4. **Pixel-accurate rendering**: Viser's `get_render()` approximates (no principal point offset, assumes square pixels). Is this good enough, or do we need a custom OpenGL renderer for accurate intrinsics?
