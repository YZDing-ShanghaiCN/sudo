import time
import numpy as np
import ampl
import trimesh

DIR_DATABASE = f"/home/czhou/Data/arm"


def create_obb_trimesh(obb3: dict):
    import trimesh

    obb_trimesh = trimesh.creation.box(extents=obb3["half_extents"])
    obb_trimesh.apply_scale(2.0)
    tf = np.eye(4)
    tf[:3, :3] = np.array([obb3["u"], obb3["v"], obb3["w"]]).T.copy()
    tf[:3, 3] = np.array(obb3["center"])
    obb_trimesh.apply_transform(tf)
    return obb_trimesh


def len_traj(traj: np.array):
    return np.linalg.norm(traj[1:] - traj[:-1], axis=1).sum()


def pad_traj(traj: np.array, pad_head_tail: int = 20):
    return np.vstack(
        (
            np.array([traj[0]] * pad_head_tail),
            traj,
            np.array([traj[-1]] * pad_head_tail),
        )
    )


def tf_rectify_obb(obb_center, obb_axes, obb_hsize, id_z=None):

    hs = np.array(obb_hsize).flatten()
    ax = np.array(obb_axes)
    ct = np.array(obb_center).flatten()

    if id_z:
        1
    else:
        id_z = np.argmin(hs)
    id_x = (id_z + 1) % 3
    id_y = (id_x + 1) % 3
    tf_0 = np.eye(4, dtype=np.float64)
    tf_0[:3, 3] = (-ct).flatten()

    sz_1 = np.array(hs)[[id_x, id_y, id_z]]
    tf_1 = np.eye(4, dtype=np.float64)
    tf_1[:3, :3] = ax[[id_x, id_y, id_z]]
    tf_2 = np.eye(4, dtype=np.float64)

    if sz_1[1] > sz_1[0]:
        tf_2[:2, :2] = np.array([[0, -1], [1, 0]])
        sz_2 = np.array([sz_1[1], sz_1[0], sz_1[2]])
        tf_2[:3, 3] = sz_2.flatten()
    else:
        tf_1[:3, 3] = sz_1.flatten()

    tf_a2w = tf_2 @ tf_1 @ tf_0

    return tf_a2w


def rainbow_colormap(data):
    """
    Applies a 'rainbow' colormap to a 1D NumPy array or scalar without using matplotlib.
    Input data is expected to be normalized between 0.0 and 1.0.
    Returns an array of RGB values (uint8, shape=(N, 3) for 1D input).
    """
    # Ensure input data is a float array and normalized [0.0, 1.0]
    data = np.atleast_1d(data).astype(float)
    if data.size > 0:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Define color transitions using sine waves for a continuous rainbow effect
    # The phases and frequencies are chosen to cycle through red, green, and blue.
    # Color values range from 0.0 to 1.0 initially
    red = np.sin(2 * np.pi * data + 0 / 3 * np.pi) * 0.5 + 0.5
    green = np.sin(2 * np.pi * data + 2 / 3 * np.pi) * 0.5 + 0.5
    blue = np.sin(2 * np.pi * data + 4 / 3 * np.pi) * 0.5 + 0.5

    # Clamp values to [0.0, 1.0] to avoid issues with floating point arithmetic
    red = np.clip(red, 0.0, 1.0)
    green = np.clip(green, 0.0, 1.0)
    blue = np.clip(blue, 0.0, 1.0)

    # Stack channels to form an Nx3 array and convert to uint8 (0-255 range)
    colored_array = np.stack([red, green, blue], axis=-1)
    return (colored_array * 255).astype(np.uint8)


def colorize(xyz):
    min_z = np.min(xyz[:, 2])
    max_z = np.max(xyz[:, 2])
    z = xyz[:, 2].copy()
    z -= min_z
    z /= max_z - min_z
    return rainbow_colormap(z)


def create_obb_trimesh(obb3: ampl.OBB3):
    obb_trimesh = trimesh.creation.box(extents=obb3.half_extents)
    obb_trimesh.apply_scale(2.0)
    tf = np.eye(4)
    tf[:3, :3] = np.array([obb3.u, obb3.v, obb3.w]).T.copy()
    tf[:3, 3] = np.array(obb3.center)
    obb_trimesh.apply_transform(tf)
    return obb_trimesh


def tic(print_cmd: bool = False, message: str = ""):
    global timer_global
    timer_global = time.perf_counter_ns()
    if print_cmd:
        print(f"# TIC {message}")


def toc(print_cmd: bool = True, message: str = ""):
    global timer_global
    delta = (float(time.perf_counter_ns()) - timer_global) / 1e6
    if print_cmd:
        print(f"# TOC = {delta} MS BY {message}")
    return delta


def join_meshes(meshes):
    combined_vertices = []
    combined_faces = []
    vertex_offset = 0

    for mesh in meshes:
        vertices = mesh[0]
        faces = mesh[1]

        # Append vertices
        combined_vertices.append(vertices)

        # Adjust face indices and append
        adjusted_faces = faces + vertex_offset
        combined_faces.append(adjusted_faces)

        # Update vertex offset
        vertex_offset += len(vertices)

    # Combine all vertices and faces into single arrays
    combined_vertices = np.vstack(combined_vertices)
    combined_faces = np.vstack(combined_faces)

    return combined_vertices, combined_faces


def bbx(points, min_xyz, max_xyz):

    min_x, max_x = min_xyz[0], max_xyz[0]
    min_y, max_y = min_xyz[1], max_xyz[1]
    min_z, max_z = min_xyz[2], max_xyz[2]

    x_mask = (points[:, 0] >= min_x) & (points[:, 0] <= max_x)
    y_mask = (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    z_mask = (points[:, 2] >= min_z) & (points[:, 2] <= max_z)

    bounding_box_mask = x_mask & y_mask & z_mask
    points_in_bbox = points[bounding_box_mask]
    return points_in_bbox
