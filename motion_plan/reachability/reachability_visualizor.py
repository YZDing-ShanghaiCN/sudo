import viser
import numpy as np
import trimesh
import time
from typing import List

def visualize_tote_and_tcp(tote_mesh: trimesh.Trimesh, gripper_mesh: trimesh.Trimesh, tcp_points: np.ndarray, scores: np.ndarray = None, tote_base_tf: np.ndarray = None, sample_rotations_tfs: List[np.ndarray] = None):
    """
    Visualize tote mesh, TCP points, and one group of sampled rotations using Viser.
    
    Args:
        tote_mesh: trimesh.Trimesh object of the tote.
        gripper_mesh: trimesh.Trimesh object of the gripper.
        tcp_points: (N, 3) array of TCP positions.
        scores: (N,) array of reachability scores (0 to 1).
        tote_base_tf: (4, 4) transformation matrix for the tote mesh.
        sample_rotations_tfs: Optional list of (4, 4) matrices representing one group of rotations at a point.
    """
    server = viser.ViserServer()
    
    if tote_base_tf is not None:
        wxyz = viser.transforms.SO3.from_matrix(tote_base_tf[:3, :3]).wxyz
        position = tote_base_tf[:3, 3]
    else:
        wxyz = (1.0, 0.0, 0.0, 0.0)
        position = (0.0, 0.0, 0.0)

    if isinstance(tote_mesh, tuple):
        # Assuming tuple is (vertices, faces)
        tote_mesh = trimesh.Trimesh(vertices=tote_mesh[0], faces=tote_mesh[1])

    if isinstance(gripper_mesh, tuple):
        # Assuming tuple is (vertices, faces)
        gripper_mesh = trimesh.Trimesh(vertices=gripper_mesh[0], faces=gripper_mesh[1])

    server.scene.add_mesh_trimesh(
        name="/tote",
        mesh=tote_mesh,
        wxyz=wxyz,
        position=position,
    )

    if scores is not None:
        # Color mapping: Red (score=0) to Green (score=1)
        colors = np.zeros((len(scores), 3))
        colors[:, 0] = 255 * (1.0 - scores) # Red component
        colors[:, 1] = 255 * scores       # Green component
        
        server.scene.add_point_cloud(
            name="/reachability_scores",
            points=tcp_points,
            colors=colors,
            point_size=0.01
        )
    else:
        server.scene.add_point_cloud(
            name="/sampled_tcp_points",
            points=tcp_points,
            colors=(0, 255, 0),
            point_size=0.005
        )

    if sample_rotations_tfs is not None:
        # Show one group of rotations at a dedicated frame (e.g., above the tote or at the first point)
        # We'll put them at the center of the sampled points for visibility
        center_pos = np.mean(tcp_points, axis=0) + np.array([0, 0, 0.2]) # Offset slightly in Z
        for i, tf in enumerate(sample_rotations_tfs):
            server.scene.add_frame(
                name=f"/rotations/frame_{i}",
                wxyz=viser.transforms.SO3.from_matrix(tf[:3, :3]).wxyz,
                position=center_pos,
                axes_length=0.05,
                axes_radius=0.002,
            )
        
        # Visualize gripper at first sample_rotations_tfs at center_pos
        if gripper_mesh is not None and len(sample_rotations_tfs) > 0:
            first_tf = sample_rotations_tfs[0]
            server.scene.add_mesh_trimesh(
                name="/gripper",
                mesh=gripper_mesh,
                wxyz=viser.transforms.SO3.from_matrix(first_tf[:3, :3]).wxyz,
                position=center_pos,
            )
    
    print(f"Viser visualization started with {len(tcp_points)} TCP points.")
    print("You can view the scene at http://localhost:8080")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down visualizor.")
