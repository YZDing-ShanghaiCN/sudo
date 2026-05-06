import numpy as np
import trimesh


def load_stl_points(stl_path: str) -> np.ndarray:
	mesh = trimesh.load(stl_path, force="mesh")
	if mesh.is_empty:
		raise ValueError("Loaded mesh is empty.")
	points = np.asarray(mesh.vertices, dtype=np.float64)
	if points.ndim != 2 or points.shape[1] != 3:
		raise ValueError("Mesh vertices must have shape (N, 3).")
	return points


def add_metric(
	points: np.ndarray,
	R: np.ndarray,
	t: np.ndarray,
	R_gt: np.ndarray,
	t_gt: np.ndarray,
) -> float:
	R = np.asarray(R, dtype=np.float64)
	t = np.asarray(t, dtype=np.float64).reshape(3)
	R_gt = np.asarray(R_gt, dtype=np.float64)
	t_gt = np.asarray(t_gt, dtype=np.float64).reshape(3)

	if R.shape != (3, 3) or R_gt.shape != (3, 3):
		raise ValueError("R and R_gt must be shape (3, 3).")
	if points.ndim != 2 or points.shape[1] != 3:
		raise ValueError("points must be shape (N, 3).")

	pred = (R @ points.T).T + t
	gt = (R_gt @ points.T).T + t_gt
	distances = np.linalg.norm(pred - gt, axis=1)
	return float(np.mean(distances))


def rot_z(theta_rad: float) -> np.ndarray:
	c = np.cos(theta_rad)
	s = np.sin(theta_rad)
	return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


if __name__ == "__main__":
	stl_path = "/home/user/Desktop/main/main2/add/底盘.STL"
	points = load_stl_points(stl_path)

	R = rot_z(np.deg2rad(15.0))
	t = np.array([0.05, -0.02, 0.10], dtype=np.float64)

	R_gt = np.eye(3, dtype=np.float64)
	t_gt = np.array([0.02, 0.00, 0.08], dtype=np.float64)

	add_value = add_metric(points, R, t, R_gt, t_gt)
	print(f"ADD: {add_value:.6f}")
