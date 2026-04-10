# sudo： My robot work in Sudo company during 2026 in Shanghai

# 1. pose

This directory is for object pose estimation from RGB image in robot manipulation tasks. 

--  ENVIRONMENT SETUP ------------- RTX 5060 8GB

reference link: [CSDN](https://blog.csdn.net/2504_93649063/article/details/159248787?spm=1001.2014.3001.5501) [CSDN](https://blog.csdn.net/2504_93649063/article/details/158879459?spm=1001.2014.3001.5501)

``` bash
# cuda preperation
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
python - <<'PY'
import torch
print("Torch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0))
print("CUDA Available:", torch.cuda.is_available())
print("GPU sm type:", torch.cuda.get_device_capability(0))
PY

# dependency installation
pip install -r requirements.txt

export TORCH_CUDA_ARCH_LIST="12.0"
pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

export all_proxy=socks5://127.0.0.1:7897
export ALL_PROXY=socks5://127.0.0.1:7897
export PYTHONPATH=$PYTHONPATH:/home/user/Desktop/main/posemain/segment_anything

python scripts/project.py
```

# 2
# Robot Deploy Environment Checker

Interactive tool for verifying robot task viability before deployment. Checks **reachability** (can the robot reach the workspace?) and **visibility** (can cameras see the workspace?) using a browser-based 3D visualization.

Built on [Viser](https://github.com/nerfstudio-project/viser) with the `hbmp` motion planning library for IK solving and collision detection.

## Features

- Config-driven scene: define robot, objects, cameras, and workspace bounds in a single YAML
- Drag EEF gizmos to test reachability via IK tracking (WBC-based)
- Real-time self-collision detection with visual feedback
- 8 robot-mounted cameras with frustum visualization
- On-demand camera rendering from any mounted camera viewpoint
- Workspace bounding box with adjustable wall constraints

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
PYTHONPATH=hbmp:src python -m deployenv_checker --config configs/example_scene.yaml
```

Open http://localhost:8080 in a browser.

### Controls

1. **Track** button -- enables EEF tracking. Drag the left/right tool gizmos and the robot follows via IK.
2. **Wall sliders** -- adjust workspace bounds (y-max, z-min/max). The robot's collision checker uses these.
3. **Camera renders** -- expand a camera group folder, click "Render" to capture a snapshot from that camera's viewpoint. "Render All" captures all 8 cameras.
4. **Show Frustums** -- toggle camera frustum wireframes in the scene.
5. **View mode** -- switch between visual, collision, or both mesh displays.

### Custom scene config

Create a YAML file (see `configs/example_scene.yaml` for reference):

```yaml
scene:
  name: "My Workspace Check"

robot:
  type: "t2da2"
  urdf_visual: "./hbmp/assets/hb11/urdf_c.urdf"
  urdf_collision: "./hbmp/assets/hb11/urdf_c.urdf"
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

cameras:
  config_path: "./configs/camera_config.json"
  show_frustums: true
  frustum_scale: 0.1

objects:
  - name: "workbench"
    mesh_path: "./path/to/mesh.glb"
    position: [0.8, 0.0, 0.4]
    draggable: true

workspace:
  bounds: [0.0, 1.25, -1.0, 1.0, 0.6, 1.8]
  show_bounds: true
```

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
