# Calculation of Yoshikawa's Manipulability (Pinocchio Only)

Yoshikawa's manipulability index $w$ quantifies how far a robot configuration is from a singularity.

$$w(q) = \sqrt{\det(J(q) J^T(q))}$$

## Implementation Plan

Following your request to **only use Pinocchio**, the implementation extracts the arm chain from the whole-body URDF to calculate scores specifically for the manipulator.

### 1. Correct Installation
The default `pip install pinocchio` installs a testing plugin. You MUST use the robotics version:

```bash
# Correct installation
pip uninstall pinocchio
pip install pin
# OR
conda install pinocchio -c conda-forge
```

### 2. Implementation Steps

1.  **Parse URDF**: Load the robot model from [beta1.0_v2_20251011(1).urdf](motion_plan/yoshikawa_score/beta1.0_v2_20251011(1).urdf).
2.  **Extract Arm Chain**:
    - Select joints: `left-joint_arm_1` through `left-joint_arm_7`.
    - Lock all other joints (torso, wheels, root) using `pin.buildReducedModel`.
3.  **Compute Jacobian**:
    - Compute the $6 \times 7$ Jacobian for the `left-link_tcp` frame.
4.  **Calculate Score**:
    - Compute $w = \sqrt{\det(J J^T)}$.

### 3. Usage & Visualization ([yoshikawa_score.py](motion_plan/yoshikawa_score/yoshikawa_score.py))

The script samples random joint configurations and visualizes the end-effector (EE) positions in 3D.
- **Color Mapping**: Yoshikawa manipulability score (Yellow for high, Purple for low).
- **Filtering**: Automatically filters out "TCP pointing up" configurations (where tool Z points towards the sky).
- **Base Frame**: Shows the arm base coordinate axes for reference.

```bash
python3.11 motion_plan/yoshikawa_score/yoshikawa_score.py
```

## Considerations for Hillbot Beta
- **Redundancy**: Since the arm has 7-DOF ($n=7$) and task space has 6-DOF ($m=6$), the arm is redundant. The manipulability score remains non-zero unless the arm reaches a singular configuration.
- **Joint Limits**: Sampled configurations respect the limits defined in the URDF.
- **Color Mapping**: The 'viridis' color map is used to show the gradient of the manipulability index across the workspace.
