import numpy as np
import matplotlib
# Try to use an interactive backend
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

INPUT = "reachability_results.npy"

def main():
    # Load the huge matrix [NX, NY, NZ, nx, ny, nz, nrx, nry, nrz, 7]
    matrix = np.load(INPUT)
    print(f"Loaded matrix with shape: {matrix.shape}")
    NX, NY, NZ, nx, ny, nz, nrx, nry, nrz, joints = matrix.shape
    # Check validity by looking at the first joint (if NaN, the whole pose is unreachable)
    # Collapse last 4 dimensions: nrx, nry, nrz, and the joints themselves
    # We want to count how many orientations (nrx, nry, nrz) are reachable for each (X,Y,Z, x,y,z)
    
    # 1. Create a mask of reachable IK solutions (True if reachable)
    # We use matrix[..., 0] because if the first joint is NaN, all 7 joints are NaN.
    reachable_mask = ~np.isnan(matrix[..., 0])

    # 2. Sum over the last 3 rotation dimensions to get reachability count per (NX, NY, NZ, nx, ny, nz)
    reachability_count = np.sum(reachable_mask, axis=(-3, -2, -1))
    print(f"Reduced shape: {reachability_count.shape}")

    reachability_fraction = reachability_count / (nrx * nry * nrz)
    print(reachability_fraction.shape)
    reachability_fraction_sum = np.sum(reachability_fraction, axis=(-3, -2, -1))
    
    idx_flat = reachability_fraction_sum.argmax()
    idx_3d = np.unravel_index(idx_flat, reachability_fraction_sum.shape)
    
    print(f"Max reachability: {reachability_fraction_sum.max():.4f}")
    print(f"Best Crate Position Index (X, Y, Z): {idx_3d}")

    vol = reachability_count[idx_3d]
    
    nx, ny, nz = vol.shape
    x, y, z = np.indices((nx, ny, nz))
    
    # Flatten for scatter plot
    x_coords = x.flatten()
    y_coords = y.flatten()
    z_coords = z.flatten()
    v_values = vol.flatten()
    
    # Split into reachable and unreachable points
    reachable_mask = v_values > 0
    unreachable_mask = v_values == 0
    print(f"unreachable points: {np.sum(unreachable_mask)}, reachable points: {np.sum(reachable_mask)}")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot reachable points with color map
    if np.any(reachable_mask):
        sc = ax.scatter(x_coords[reachable_mask], y_coords[reachable_mask], z_coords[reachable_mask], 
                        c=v_values[reachable_mask], cmap='viridis', s=v_values[reachable_mask]*10, 
                        alpha=0.6, label='Reachable')
        plt.colorbar(sc, label='Number of Reachable Orientations')
    
    # Plot unreachable points as small grey dots
    if np.any(unreachable_mask):
        ax.scatter(x_coords[unreachable_mask], y_coords[unreachable_mask], z_coords[unreachable_mask], 
                   c='grey', s=2, alpha=0.2, label='Unreachable')
    
    ax.set_xlabel('X (grid index)')
    ax.set_ylabel('Y (grid index)')
    ax.set_zlabel('Z (grid index)')
    ax.set_title('Reachability Workspace Visualization (Crate Pose Index: {})'.format(idx_3d))
    ax.legend()
    

    plt.show()

if __name__ == "__main__":
    main()