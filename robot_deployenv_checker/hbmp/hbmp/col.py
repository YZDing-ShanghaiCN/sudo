import numpy as np
import ampl
import numpy as np
from .common import *
from .util import convex_from_mesh, create_obb3, create_xyzr_offset


class CollisionObjectBase:
    def __init__(self):
        self.entity = None
        self.entity_move = None
        self._pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=DTypeFloat)
        self.active = False

    def update_parameter(self):
        return

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, rwt: np.ndarray):
        np.copyto(dst=self._pose, src=rwt)


class CollisionObjectPlane(CollisionObjectBase):
    def __init__(
        self,
        center: DTypeVertices = np.array([0, 0, 0], dtype=DTypeFloat),
        normal: DTypeVertices = np.array([0, 0, 1], dtype=DTypeFloat),
    ):
        super().__init__()
        self.entity = ampl.Plane3()

        ampl.collision_initialize_object(
            plane_center=center.astype(DTypeFloat),
            plane_normal=normal.astype(DTypeFloat),
            plane_obj=self.entity,
            extent_consider_large=10.0,
        )
        self.entity_move = ampl.Plane3(self.entity)

    def update_parameter(self, center: DTypeVertices, normal: DTypeVertices):
        ampl.collision_initialize_object(
            plane_center=center.astype(DTypeFloat),
            plane_normal=normal.astype(DTypeFloat),
            plane_obj=self.entity,
            extent_consider_large=10.0,
        )
        self.entity_move = ampl.Plane3(self.entity)
        return

    @property
    def pose(self):
        return super().pose

    @pose.setter
    def pose(self, rwt: np.ndarray):
        np.copyto(dst=self._pose, src=rwt)


class CollisionObjectAAABB(CollisionObjectBase):
    def __init__(
        self,
        xyz_min: DTypeVertices = np.array([-2, -2, 0], dtype=DTypeFloat),
        xyz_max: DTypeVertices = np.array([2, 2, 3], dtype=DTypeFloat),
    ):
        super().__init__()
        self.entity_move = ampl.AAABB3()
        self.entity_move.xyz_max = np.array(xyz_max).astype(DTypeFloat).tolist()
        self.entity_move.xyz_min = np.array(xyz_min).astype(DTypeFloat).tolist()

    def update_parameter(self, xyz_min: DTypeVertices, xyz_max: DTypeVertices):
        self.entity_move.xyz_max = np.array(xyz_max).astype(DTypeFloat).tolist()
        self.entity_move.xyz_min = np.array(xyz_min).astype(DTypeFloat).tolist()
        return

    @property
    def pose(self):
        return super().pose

    @property
    def xyz_min(self):
        return self.entity_move.xyz_min

    @property
    def xyz_max(self):
        return self.entity_move.xyz_max

    @pose.setter
    def pose(self, rwt: np.ndarray):
        np.copyto(dst=self._pose, src=rwt)


class CollisionObjectConvex(CollisionObjectBase):
    def __init__(self, convex: DTypeConvex):
        super().__init__()
        self.entity = convex_from_mesh(convex)
        self.entity_move = ampl.VCvhf(self.entity)

    def update_parameter(self, convex: DTypeConvex):
        self.entity = convex_from_mesh(convex)
        self.entity_move = ampl.VCvhf(self.entity)
        return

    @property
    def pose(self):
        return super().pose

    @pose.setter
    def pose(self, rwt: np.ndarray):
        np.copyto(dst=self._pose, src=rwt)
        ampl.collision_transform_object(self.entity, self.entity_move, self._pose)


class CollisionObjectOBB(CollisionObjectBase):
    def __init__(
        self,
        center: DTypeVertices,
        half_extents: DTypeVertices,
        u: DTypeVertices,
        v: DTypeVertices,
        w: DTypeVertices,
    ):
        super().__init__()
        self.entity = ampl.OBB3()
        self.entity_move = ampl.OBB3()

        self.entity.center = center
        self.entity.half_extents = half_extents
        self.entity.u = u
        self.entity.v = v
        self.entity.w = w
        self.entity_move.center = center
        self.entity_move.half_extents = half_extents
        self.entity_move.u = u
        self.entity_move.v = v
        self.entity_move.w = w

    def update_parameter(
        self,
        center: DTypeVertices,
        half_extents: DTypeVertices,
        u: DTypeVertices,
        v: DTypeVertices,
        w: DTypeVertices,
    ):
        self.entity.center = center
        self.entity.half_extents = half_extents
        self.entity.u = u
        self.entity.v = v
        self.entity.w = w
        self.entity_move = ampl.OBB3(self.entity)
        return

    @property
    def pose(self):
        return super().pose

    @pose.setter
    def pose(self, rwt: np.ndarray):
        np.copyto(dst=self._pose, src=rwt)
        ampl.collision_transform_object(self.entity, self.entity_move, self._pose)


class CollisionObjectOBBSoft(CollisionObjectBase):
    def __init__(
        self,
        center: DTypeVertices,
        half_extents: DTypeVertices,
        u: DTypeVertices,
        v: DTypeVertices,
        w: DTypeVertices,
        chamfer_radius: DTypeFloat,
    ):
        super().__init__()
        self.entity = ampl.OBB3()
        self.entity_move = ampl.OBB3()
        self.entity.center = center
        self.entity.half_extents = [he - chamfer_radius for he in half_extents]
        self.entity.u = u
        self.entity.v = v
        self.entity.w = w
        self.entity_move.center = center
        self.entity_move.half_extents = half_extents
        self.entity_move.u = u
        self.entity_move.v = v
        self.entity_move.w = w
        self.r: float = chamfer_radius

    def update_parameter(
        self,
        center: DTypeVertices,
        half_extents: DTypeVertices,
        u: DTypeVertices,
        v: DTypeVertices,
        w: DTypeVertices,
        chamfer_radius: DTypeFloat,
    ):
        self.entity.center = center
        self.entity.half_extents = half_extents
        self.entity.u = u
        self.entity.v = v
        self.entity.w = w
        self.entity_move = ampl.OBB3(self.entity)
        self.r: float = chamfer_radius
        return

    @property
    def pose(self):
        return super().pose

    @pose.setter
    def pose(self, rwt: np.ndarray):
        np.copyto(dst=self._pose, src=rwt)
        ampl.collision_transform_object(self.entity, self.entity_move, self._pose)


class CollisionObjectVSph(CollisionObjectBase):
    def __init__(self, list_xyzr: DTypeListVertices):
        super().__init__()
        xyzr, offset = create_xyzr_offset(list_xyzr)
        self.entity = ampl.VSphG8f()
        self.entity_move = ampl.VSphG8f()
        ampl.collision_initialize_object(xyzr, offset, self.entity)
        ampl.collision_initialize_object(xyzr, offset, self.entity_move)
        self.centers = np.zeros((self.entity.nb_sph, 3), dtype=np.float32)
        self.grads = np.zeros((self.entity.nb_sph, 3), dtype=np.float32)
        self.dists = np.zeros(self.entity.nb_sph, dtype=np.float32)
        self.labels = np.zeros(self.entity.nb_sph, dtype=np.uint32)

    # @property
    # def pose(self):
    #     return super().pose
    def set_pose(self, rwt: np.ndarray, ids_row_rwt: list[int]):
        for i_rwt in ids_row_rwt:
            ampl.collision_transform_object(
                self.entity, self.entity_move, i_rwt, rwt[i_rwt].astype(DTypeFloat)
            )

    # @pose.setter
    # def pose(self, rwt: np.ndarray, ids_row_rwt: list[int]):
    #     np.copyto(dst=self._pose, src=rwt)
    #     for i in ids_row_rwt:
    #         ampl.collision_transform_object(self.entity, self.entity_move, self._pose,rwt[i])
