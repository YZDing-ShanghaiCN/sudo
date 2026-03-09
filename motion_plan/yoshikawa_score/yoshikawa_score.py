import numpy as np
import os
import pinocchio as pin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_axes(ax, T, length=0.1, label=None):
    """
    Plots 3D axes from a 4x4 transformation matrix.
    Red = X, Green = Y, Blue = Z
    """
    origin = T[:3, 3]
    R = T[:3, :3]
    
    colors = ['r', 'g', 'b']
    for i in range(3):
        axis = R[:, i] * length
        ax.quiver(origin[0], origin[1], origin[2], axis[0], axis[1], axis[2], 
                  color=colors[i], pivot='tail', linewidth=2)
    
    if label:
        ax.text(origin[0], origin[1], origin[2], label, color='black', fontweight='bold')

class ArmManipulability:
    def __init__(self, urdf_path, arm_prefix="left"):
        """
        Calculates manipulability for a Hillbot arm using Pinocchio from URDF.
        Extracts the arm chain from the whole-body model.
        
        Args:
            urdf_path (str): Path to the URDF file.
            arm_prefix (str): "left" or "right" to select which arm to process.
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        # 1. Load the full model
        # Note: If this fails with AttributeError, you have the wrong pinocchio package installed.
        # Run: pip uninstall pinocchio && pip install pin
        full_model = pin.buildModelFromUrdf(urdf_path)
        
        # 2. Identify arm joints (7-DOF Hillbot arm)
        # Joints are named like: left-joint_arm_1, left-joint_arm_2, ...
        self.arm_prefix = arm_prefix
        self.joint_names = [f"{arm_prefix}-joint_arm_{i}" for i in range(1, 8)]
        
        # 3. Identify joints to lock (all non-arm joints)
        all_joint_names = list(full_model.names)
        # Skip 'universe' (index 0)
        joints_to_lock = [
            name for name in all_joint_names 
            if name != "universe" and name not in self.joint_names
        ]
        # print(f"Joints to lock: {joints_to_lock}")
        # breakpoint()
        lock_ids = [full_model.getJointId(name) for name in joints_to_lock]
        
        # 4. Build reduced model (locks other joints at their neutral position)
        q_neutral = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, lock_ids, q_neutral)
        self.data = self.model.createData()
        
        # 5. Identify TCP frame in the reduced model
        self.tcp_frame_name = f"{arm_prefix}-link_tcp"
        if not self.model.existFrame(self.tcp_frame_name):
            # Fallback to the last arm link if TCP frame isn't found
            self.tcp_frame_name = f"{arm_prefix}-link_arm_7"
            
        self.tcp_frame_id = self.model.getFrameId(self.tcp_frame_name)
        
        # Identify Arm Base frame
        self.base_frame_name = f"{arm_prefix}-link_arm_base"
        if not self.model.existFrame(self.base_frame_name):
            # Fallback to absolute base link if specific arm base not found
            self.base_frame_name = "base_link"
        self.base_frame_id = self.model.getFrameId(self.base_frame_name)
        
        print(f"Initialized {arm_prefix} arm model with {self.model.nq} degrees of freedom.")

    def calculate_score(self, q):
        """
        Calculates Yoshikawa's manipulability index: sqrt(det(J * J^T))
        
        Args:
            q (np.ndarray): Arm joint angles (size self.model.nq).
            
        Returns:
            float: Manipulability index.
        """
        if len(q) != self.model.nq:
            raise ValueError(f"Expected q of size {self.model.nq}, got {len(q)}")
            
        # Compute Jacobian at the TCP frame
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get Jacobian in Local World Aligned frame (spatial velocity)
        # This provides a 6xN matrix
        J = pin.getFrameJacobian(self.model, self.data, self.tcp_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        # Yoshikawa index: sqrt(det(J * J.T))
        # For redundant manipulators, J*J.T is 6x6.
        w = np.sqrt(np.linalg.det(J @ J.T))
        return w

    def calculate_translation_score(self, q):
        """Calculates manipulability focusing only on translation (linear part, 3xN Jacobian)."""
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(self.model, self.data, self.tcp_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        J_lin = J[:3, :]
        w = np.sqrt(np.linalg.det(J_lin @ J_lin.T))
        return w

    def get_ee_transform(self, q):
        """Returns the 4x4 transformation matrix of the end-effector."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tcp_frame_id].homogeneous

    def get_base_transform(self):
        """Returns the 4x4 transformation matrix of the arm base."""
        # For reduced model, base position is constant relative to universe
        q_neutral = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q_neutral)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.base_frame_id].homogeneous

    def sample_configurations(self, num_samples=1000, filter_pointing_up=True):
        """
        Samples joint 1, joint 2 and joint 3 on a grid, keeping others at neutral.
        
        Args:
            num_samples: Total approximate number of points (used to determine grid resolution).
            filter_pointing_up: If True, filters out configurations where EE Z points up.
        
        Returns:
            dict: { 'q': (N, 7), 'tf': (N, 4, 4), 'score': (N,) }
        """
        qs = []
        tfs = []
        scores = []
        
        # Calculate grid resolution based on total desired samples (cubic root for 3 joints)
        res = int(np.cbrt(num_samples))
        q1_range = np.linspace(-np.pi, np.pi, res)
        q2_range = np.linspace(-np.pi, np.pi, res)
        q3_range = np.linspace(-np.pi, np.pi, res)
        
        # Get neutral configuration for the reduced model
        q_neutral = pin.neutral(self.model)
        
        print(f"Sampling {res}x{res}x{res} grid for {self.arm_prefix} Arm (Joint 1, 2, 3)...")
        
        for q1 in q1_range:
            for q2 in q2_range:
                for q3 in q3_range:
                    # Create configuration: set J1, J2, J3, others neutral
                    q = q_neutral.copy()
                    q[0] = q1
                    q[1] = q2
                    q[2] = q3
                    
                    # Compute Forward Kinematics and Jacobian
                    pin.forwardKinematics(self.model, self.data, q)
                    pin.computeJointJacobians(self.model, self.data, q)
                    pin.updateFramePlacements(self.model, self.data)
                    
                    # EE transform
                    tf = self.data.oMf[self.tcp_frame_id].homogeneous
                    
                    # Filter: Check if TCP Z-axis points Up (Z component of Z axis > 0.8)
                    # if filter_pointing_up and tf[2, 2] > 0.8:
                    #     continue
                    
                    # Manipulability score (using translation part for visual workspace analysis)
                    J = pin.getFrameJacobian(self.model, self.data, self.tcp_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    J_lin = J[:3, :]
                    score = np.sqrt(np.linalg.det(J_lin @ J_lin.T))
                    
                    qs.append(q)
                    tfs.append(tf)
                    scores.append(score)
            
        print(f"Sampling complete. Found {len(qs)} valid configurations.")
        return {
            'q': np.array(qs),
            'tf': np.array(tfs),
            'score': np.array(scores)
        }

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    URDF_FILE = os.path.join(SCRIPT_DIR, "beta1.0_v2_20251011(1).urdf")
    
    try:
        # Initialize
        arm_calc = ArmManipulability(URDF_FILE, arm_prefix="left")
        
        # 1. Perform sampling (filter TCP pointing up)
        NUM_SAMPLES = 1000
        results = arm_calc.sample_configurations(num_samples=NUM_SAMPLES, filter_pointing_up=True)
        
        # 2. Extract data for visualization
        # Positions are in the 4th column of the 4x4 matrix
        positions = results['tf'][:, :3, 3]  # shape (N, 3)
        scores = results['score']          # shape (N,)
        
        # 3. Visualize
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Arm Base Axis
        base_T = arm_calc.get_base_transform()
        plot_axes(ax, base_T, length=0.2, label=f"{arm_calc.arm_prefix}_base")
        
        # Create scatter plot
        # c is the color array (scores), cmap is the color map
        sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                        c=scores, cmap='viridis', s=20, alpha=0.8)
        
        # Add labels and colorbar
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Yoshikawa Manipulability - {arm_calc.arm_prefix.capitalize()} Arm')
        
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label('Manipulability Score')
        
        print("Displaying plot. Close the window to exit.")
        plt.show()
        
    except AttributeError:
        print("\n[ERROR] Incorrect Pinocchio library detected.")
        print("You likely installed the 'nose-plugin' version of Pinocchio.")
        print("Please fix it with:")
        print("  pip uninstall pinocchio && pip install pin\n")
    except Exception as e:
        print(f"Error: {e}")





