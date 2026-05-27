import ampl
from .common import *
from contextlib import contextmanager
import time

MAGIC_NUMBER_COLLISION_OBB_OFFSET = 0.025


@contextmanager
def tictoc(comment: str = "", end: str = "\n"):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        duration = t1 - t0
        print(f"# [TIMER] CODE BLOCK = {duration:.4f} SECONDS BY {comment}", end=end)


def create_xyzr_offset(vs: list[list[float]]):
    xyzr = [np.array(v, dtype=np.float32).reshape((-1, 4)) for v in vs]
    offset = [len(a) for a in xyzr]
    xyzr = np.vstack(xyzr)
    offset = np.array(offset, dtype=np.uint32)
    return xyzr, offset


def convex_from_mesh(vf: DTypeConvex) -> ampl.VCvhf:
    if isinstance(vf, Tuple):
        v = vf[0]
        f = vf[1]
        fccs = ampl.trimesh_connected_components(f.astype(np.uint32))
        vs = [
            v[np.unique(f[fcc].flatten())].astype(np.float32)
            for _, fcc in enumerate(fccs)
        ]
    elif isinstance(vf, List):
        vs = [v.astype(np.float32) for v in vf]
    else:
        return None
    cvh = ampl.VCvhf()
    ampl.collision_initialize_object(vs, cvh)
    return cvh


def create_obb3(
    center: DTypeVertices,
    half_extents: DTypeVertices,
    u: DTypeVertices,
    v: DTypeVertices,
    w: DTypeVertices,
) -> ampl.OBB3:
    obb3 = ampl.OBB3()
    obb3.center = center
    obb3.half_extents = half_extents
    obb3.u = u
    obb3.v = v
    obb3.w = w
    return obb3


def mirror_left(tf: np.ndarray):
    tf_m = np.eye(4, dtype=tf.dtype)

    tf_m[:3, 2] = tf[:3, 2].flatten()
    tf_m[1, 2] *= -1

    tf_m[:3, 1] = tf[:3, 1].flatten()
    tf_m[1, 1] *= -1

    tf_m[:3, 0] = np.cross(tf_m[:3, 1], tf_m[:3, 2])
    tf_m[:3, 3] = tf[:3, 3].flatten()
    tf_m[1, 3] *= -1
    return tf_m


def mirror_left_rwt(rwt: np.ndarray):
    return ampl.tf44_to_qt7(mirror_left(ampl.qt7_to_tf44(rwt)))


def spatial_to_mixed_linear_fast(J_spatial, p_world):
    J_mixed = np.copy(J_spatial, order="F")
    px, py, pz = p_world.flatten()
    J_mixed[:3] += -1.0 * (
        np.array([[0, -pz, py], [pz, 0, -px], [-py, px, 0]]) @ J_mixed[3:]
    )
    return J_mixed


def spatial_to_mixed_linear_projection(
    J_spatial, p_world: np.ndarray, g_world: np.ndarray
):
    J_mixed = np.copy(J_spatial, order="F")
    px, py, pz = p_world.flatten()
    J_mixed[:3] += -1.0 * (
        np.array([[0, -pz, py], [pz, 0, -px], [-py, px, 0]]) @ J_mixed[3:]
    )
    return (g_world.reshape((1, -1)) @ J_mixed[:3]).flatten()


def top_k_per_label(floats, labels, k=1):
    # 1. Get indices that would sort the float array
    idx = np.argsort(floats)
    sorted_labels = labels[idx]
    sorted_indices = idx  # These are the original positions
    unique_labels = np.unique(labels)
    results = {}
    for label in unique_labels:
        # 2. For each label, find where it appears in the sorted list
        label_mask = sorted_labels == label
        # 3. Take the first K original positions
        results[label] = sorted_indices[label_mask][:k].tolist()
    return results


def get_top_k_indices_per_label(d, labels, mask, k):
    """
    Finds the original array positions of the top k values in 'd' for each label in 'mask'.
    """
    d = np.asarray(d)
    labels = np.asarray(labels)

    results = {}

    for label in mask:
        # 1. Find the original indices where the current label exists
        label_indices = np.where(labels == label)[0]

        # Handle the case where the label isn't in the array
        if len(label_indices) == 0:
            results[label] = np.array([], dtype=int)
            continue

        # 2. Extract the values corresponding to this label
        label_values = d[label_indices]

        # Handle the case where there are fewer than k elements found
        actual_k = min(k, len(label_values))

        # 3. Find the relative indices of the top k values
        if actual_k == len(label_values):
            # If we want all elements, just sort them descending
            top_k_rel_idx = np.argsort(label_values)[::-1]
        else:
            # Partition to get the top k elements (unsorted) at the end of the array
            part_idx = np.argpartition(label_values, -actual_k)[-actual_k:]
            # Sort only those top k elements in descending order
            top_k_rel_idx = part_idx[np.argsort(label_values[part_idx])[::-1]]

        # 4. Map the relative indices back to the original global indices
        results[label] = label_indices[top_k_rel_idx]

    return results


def rescale_row_norms(A, target_norms):
    """
    Rescales each row in A so its L2 norm matches the corresponding value in target_norms.
    """
    A = np.asarray(A, dtype=float)
    target_norms = np.asarray(target_norms, dtype=float).reshape(-1, 1)

    # 1. Calculate the current L2 norm of each row
    # keepdims=True ensures the shape is (N, 1) for broadcasting
    current_norms = np.linalg.norm(A, axis=1, keepdims=True)

    # 2. Prevent division by zero for rows that are entirely zeros
    safe_norms = np.where(current_norms == 0, 1.0, current_norms)

    # 3. Calculate the scaling factor and broadcast multiply
    rescaled_A = A * (target_norms / safe_norms) * np.sign(target_norms)

    return rescaled_A
