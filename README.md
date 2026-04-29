# My Robot Work - Sudo Company (2026 Shanghai)

Comprehensive robotics projects including AGV control, object pose estimation, and deployment environment validation.

## 📁 Project Structure

```
main/
├── agv-roboshoppro-control/    # AGV Robot Control System
├── main2/                       # Stereo Depth Inference Module
├── posemain/                    # Object Pose Estimation Module
├── robot_deployenv_checker-main/ # Deployment Environment Checker
└── README.md
```

---

## 1. AGV Robot Control System (`agv-roboshoppro-control/`)

Control and calibration system for AGV robots with camera integration and motion control.

### Contents
- **calibrate/** - Camera calibration tools (ArUco, chessboard detection)
- **config/** - Configuration files (camera, robot, control profiles)
- **main/** - Main control loop and processing
- **scripts/** - Utility scripts (control, navigation, testing)
- **src/** - Source code (client, controller, types)
- **output/** - Generated position data and capture results
- **tests/** - Unit tests for motion controller

### Key Features
- Multi-camera calibration (ArUco and chessboard patterns)
- V4L2 camera capture support
- Robot motion control and navigation
- Real-time position tracking (2D/3D)

### Setup
```bash
cd agv-roboshoppro-control
pip install -r requirements.txt
python main/main.py
```

---

## 2. Object Pose Estimation (`posemain/`)

RGB-based object pose estimation for robot manipulation tasks.

### Hardware Requirements
- **GPU**: RTX 5060 8GB or better
- **OS**: Linux x86_64
- **Python**: 3.10+

### Environment Setup

Reference: [CSDN Blog 1](https://blog.csdn.net/2504_93649063/article/details/159248787?spm=1001.2014.3001.5501) | [CSDN Blog 2](https://blog.csdn.net/2504_93649063/article/details/158879459?spm=1001.2014.3001.5501)

```bash
# CUDA preparation
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify CUDA setup
python - <<'PY'
import torch
print("Torch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0))
print("CUDA Available:", torch.cuda.is_available())
print("GPU sm type:", torch.cuda.get_device_capability(0))
PY

# Install dependencies
pip install -r requirements.txt

# Install specialized packages
export TORCH_CUDA_ARCH_LIST="12.0"
pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Configure environment paths
export all_proxy=socks5://127.0.0.1:7897
export ALL_PROXY=socks5://127.0.0.1:7897
export PYTHONPATH=$PYTHONPATH:/home/user/Desktop/main/posemain/segment_anything

python scripts/project.py
```

### Key Modules
- **BundleDF** - Pose estimation using bundle DF
- **MiDaS** - Monocular depth estimation
- **Depth Anything V2** - Advanced depth perception
- **DINO v2** - Vision features
- **Segment Anything** - Instance segmentation

---

## 3. Robot Deploy Environment Checker (`robot_deployenv_checker-main/`)

Interactive tool for verifying robot task viability before deployment. Checks **reachability** (can the robot reach the workspace?) and **visibility** (can cameras see the workspace?) using a browser-based 3D visualization.

**Built on**: [Viser](https://github.com/nerfstudio-project/viser) + `hbmp` motion planning library (IK solving and collision detection)

### Features
- ✅ Config-driven scene setup (robots, objects, cameras, workspace bounds in YAML)
- ✅ Drag EEF gizmos to test reachability via IK tracking (WBC-based)
- ✅ Real-time self-collision detection with visual feedback
- ✅ 8 robot-mounted cameras with frustum visualization
- ✅ On-demand camera rendering from any mounted camera viewpoint
- ✅ Workspace bounding box with adjustable wall constraints

### Requirements
- **OS**: Linux x86_64 (pre-built wheels are platform-specific)
- **Python**: 3.10 (required by pre-built wheels)
- Tool: [uv](https://docs.astral.sh/uv/) (recommended)

### Setup

```bash
cd robot_deployenv_checker-main

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

### Usage

```bash
source .venv/bin/activate
PYTHONPATH=hbmp:src python -m deployenv_checker --config configs/example_scene.yaml
```

Open **http://localhost:8080** in a browser.

### GUI Controls

| Control | Function |
|---------|----------|
| **Track** button | Enable EEF tracking; drag left and right tool gizmos for IK following |
| **Wall sliders** | Adjust workspace bounds (y-max, z-min/max) |
| **Camera renders** | Capture snapshots from camera viewpoints; "Render All" for all 8 cameras |
| **Show Frustums** | Toggle camera frustum wireframes |
| **View mode** | Switch between visual, collision, or both mesh displays |

### Custom Scene Config

Create a YAML file (example below and in `configs/example_scene.yaml`):

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

### Testing

```bash
source .venv/bin/activate
PYTHONPATH=hbmp:src python -m pytest tests/ -v
```

### Project Structure

```
robot_deployenv_checker-main/
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

---

## 🛠️ Quick Start Guide

### For AGV Control
```bash
cd agv-roboshoppro-control
pip install -r requirements.txt
python main/main.py
```

### For Pose Estimation (CUDA-enabled)
```bash
cd posemain
# Follow environment setup above with CUDA
python run_demo.py
```

### For Deployment Checking
```bash
cd robot_deployenv_checker-main
source .venv/bin/activate
PYTHONPATH=hbmp:src python -m deployenv_checker --config configs/example_scene.yaml
# Open http://localhost:8080
```

---

## 4. Stereo Depth Inference Module (`main2/`)

Module for stereo depth inference and depth map (EXR) processing.

### Key Features
- Reading, writing, and processing `.exr` depth maps
- Camera math operations (Plücker rays calculation, spatial transforms via `calibur`)
- Preprocessing utilities: Image cropping, intrinsic camera matrix updating (`stereo_depth_inference_middleburry.py`)
- Camera configurations and specific pose presets (`aililight_cameras/`, `near_pose`, `far_pose`, `wait_pose`)

### Requirements
- **Python libraries**: `torch`, `OpenEXR`, `numpy`, `opencv-python`, `Pillow`, `calibur`, `Imath`, `pyyaml`

---

## 📝 Notes

- Each project has independent dependencies in respective `requirements.txt`
- CUDA and GPU setup is critical for `posemain`
- Ensure Python 3.10 for `robot_deployenv_checker-main` (pre-built wheels requirement)
- See individual project directories for detailed documentation

---

## ✨ Last Updated
2026年4月29日