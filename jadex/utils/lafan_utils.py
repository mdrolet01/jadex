from typing import Dict

import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from jadex.lmj.genmo_utils import calculate_relative_site_quatities, sd2mat
from jadex.utils import *
from jadex.utils import non_pytree

# TODO: make pull request for loco-mujoco to match genmo's:
from jax import vmap
from jax.scipy.spatial.transform import Rotation as R
from jax.scipy.spatial.transform import Rotation as jnp_R
from loco_mujoco.core.utils.math import (
    calc_rel_positions,
    calc_site_velocities,
    calculate_global_rotation_matrices,
    calculate_relative_rotation_matrices,
    calculate_relative_velocity_in_local_frame,
    quat_scalarfirst2scalarlast,
)
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
from loco_mujoco.trajectory.dataclasses import SingleData, TrajectoryData
from scipy.spatial.transform import Rotation as np_R

DIM_POS = 3
DIM_QUAT = 4

ROT_REPR = "sd"
if ROT_REPR == "sd":
    DIM_ROTVEC = 6
elif ROT_REPR == "rotvec":
    DIM_ROTVEC = 3

DIM_VEL = 6


@struct.dataclass
class RelSitesInputSpace:
    main_site_height: jnp.ndarray
    main_site_xrot: jnp.ndarray
    # main_site_vel_local: jnp.ndarray
    site_rpos: jnp.ndarray
    site_rangles: jnp.ndarray
    site_rvel: jnp.ndarray

    @property
    def n_samples(self):
        return self.main_site_height.shape[0]


@struct.dataclass
class RelSites:
    use_positions: bool = non_pytree()
    use_rotations: bool = non_pytree()
    use_velocities: bool = non_pytree()
    indices: Dict[str, jnp.ndarray] = non_pytree()
    main_site_id: int = non_pytree()
    model_site_body_id: jnp.ndarray = non_pytree()
    model_body_rootid: jnp.ndarray = non_pytree()
    root_qpos_ids: jnp.ndarray = non_pytree()
    rel_site_ids: jnp.ndarray = non_pytree()

    @classmethod
    def create(cls, env, params):
        use_positions = params.get("use_positions", True)
        use_rotations = params.get("use_rotations", True)
        use_velocities = params.get("use_velocities", False)

        model = env.model
        root_joint_name = env.root_free_joint_xml_name
        root_qpos_ids = mj_jntname2qposid(root_joint_name, model)
        site_names = env.sites_for_mimic
        nsites = len(site_names) - 1
        rel_site_ids = jnp.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names]
        )

        # calculate the indices of all properties
        indices = RelSites.calculate_indices(nsites, use_positions, use_rotations, use_velocities)

        main_site_id = 0  # --> zeroth index in rel_site_ids

        return cls(
            use_positions=use_positions,
            use_rotations=use_rotations,
            use_velocities=use_velocities,
            indices=indices,
            main_site_id=main_site_id,
            model_site_body_id=model.site_bodyid,
            model_body_rootid=model.body_rootid,
            root_qpos_ids=root_qpos_ids,
            rel_site_ids=rel_site_ids,
        )

    def main_site_vel_calc(self, data):
        main_site_body_id = self.model_site_body_id[self.main_site_id]
        main_site_root_body_id = self.model_body_rootid[main_site_body_id]
        return calc_site_velocities(
            site_ids=main_site_body_id,
            data=data,
            parent_body_id=main_site_body_id,
            root_body_id=main_site_root_body_id,
            backend=jnp,
        )

    def rel_site_calc(self, data):
        return calculate_relative_site_quatities(
            data,
            rel_site_ids=self.rel_site_ids,
            rel_body_ids=self.model_site_body_id[self.rel_site_ids],
            body_rootid=self.model_body_rootid,
            backend=jnp,
            rot_repr=ROT_REPR,
        )

    def extract_from_trajectory(self, traj_data: TrajectoryData):
        root_quat = traj_data.qpos[:, self.root_qpos_ids[3:]]
        root_xmat = R.from_quat(quat_scalarfirst2scalarlast(root_quat)).as_matrix()
        root_xmat_T = jnp.transpose(root_xmat, axes=(0, 2, 1))

        # input all but the splitpoints attribute
        traj_data = SingleData(
            qpos=traj_data.qpos,
            qvel=traj_data.qvel,
            xpos=traj_data.xpos,
            xquat=traj_data.xquat,
            cvel=traj_data.cvel,
            subtree_com=traj_data.subtree_com,
            site_xpos=traj_data.site_xpos,
            site_xmat=traj_data.site_xmat,
        )

        # calculate the main site quantities
        main_site_height = traj_data.site_xpos[:, 0, 2]
        main_site_xmat = traj_data.site_xmat[:, 0].reshape(-1, 3, 3)

        if ROT_REPR == "rotvec":
            main_site_xrot = R.from_matrix(main_site_xmat).as_rotvec()
        if ROT_REPR == "sd":
            main_site_xrot = main_site_xmat[..., :2].reshape(main_site_xmat.shape[:-2] + (-1,))

        # main_site_vel = vmap(self.main_site_vel_calc)(traj_data)
        # main_site_vel = jnp.squeeze(main_site_vel, axis=1)
        # main_site_vel_lin = main_site_vel[:, :3]
        # main_site_vel_ang = main_site_vel[:, 3:]

        # # calculate local velocity
        # main_site_vel_lin_local = jnp.einsum("nij,nj->ni", root_xmat_T, main_site_vel_lin)
        # main_site_vel_local = jnp.concatenate([main_site_vel_lin_local, main_site_vel_ang], axis=1)

        # calculate the relative site quantities (relative to main site)
        site_rpos, site_rangles, site_rvel = vmap(self.rel_site_calc)(traj_data)

        rel_sites = RelSitesInputSpace(
            main_site_height,
            main_site_xrot,
            # main_site_vel_local,
            site_rpos=site_rpos if self.use_positions else None,
            site_rangles=site_rangles if self.use_rotations else None,
            site_rvel=site_rvel if self.use_velocities else None,
        )

        return rel_sites

    def space2array(self, data):
        # [data.main_site_height.reshape(-1, 1), data.main_site_xrot, data.main_site_vel_local],
        arr = jnp.concatenate(
            [data.main_site_height.reshape(-1, 1), data.main_site_xrot],
            axis=1,
        )

        if self.use_positions:
            rpos = data.site_rpos.reshape(data.site_rpos.shape[0], -1)
            arr = jnp.concatenate([arr, rpos], axis=1)

        if self.use_rotations:
            rangles = data.site_rangles.reshape(data.site_rangles.shape[0], -1)
            arr = jnp.concatenate([arr, rangles], axis=1)

        if self.use_velocities:
            rvels = data.site_rvel.reshape(data.site_rvel.shape[0], -1)
            arr = jnp.concatenate([arr, rvels], axis=1)

        return arr

    def array2space(self, array):
        n_samples = array.shape[0]
        main_site_height = array[:, self.indices["main_site_height"]].squeeze(-1)
        main_site_xrot = array[:, self.indices["main_site_xrot"]]
        # main_site_vel_local = array[:, self.indices["main_site_vel_local"]]
        site_rpos = array[:, self.indices["site_rpos"]].reshape(n_samples, -1, DIM_POS)
        site_rangles = array[:, self.indices["site_rangles"]].reshape(n_samples, -1, DIM_ROTVEC)
        site_rvel = array[:, self.indices["site_rvel"]].reshape(n_samples, -1, DIM_VEL)

        return RelSitesInputSpace(
            main_site_height=main_site_height,
            main_site_xrot=main_site_xrot,
            # main_site_vel_local=main_site_vel_local,
            site_rpos=site_rpos if self.use_positions else None,
            site_rangles=site_rangles if self.use_rotations else None,
            site_rvel=site_rvel if self.use_velocities else None,
        )

    @staticmethod
    def calculate_indices(num_sites: int, use_positions, use_rotations, use_velocities):
        # Initialize indices
        indices = {}
        # Main site properties
        current_index = 0
        indices["main_site_height"] = jnp.array([current_index])
        current_index += 1

        indices["main_site_xrot"] = jnp.arange(current_index, current_index + DIM_ROTVEC)
        current_index += DIM_ROTVEC

        # indices["main_site_vel_local"] = jnp.arange(current_index, current_index + DIM_VEL)
        # current_index += DIM_VEL

        # Site-related properties
        if use_positions:
            indices["site_rpos"] = jnp.arange(current_index, current_index + num_sites * DIM_POS)
            current_index += num_sites * DIM_POS
        else:
            indices["site_rpos"] = jnp.array([], dtype=jnp.int32)

        if use_rotations:
            indices["site_rangles"] = jnp.arange(current_index, current_index + num_sites * DIM_ROTVEC)
            current_index += num_sites * DIM_ROTVEC
        else:
            indices["site_rangles"] = jnp.array([], dtype=jnp.int32)

        if use_velocities:
            indices["site_rvel"] = jnp.arange(current_index, current_index + num_sites * DIM_VEL)
            current_index += num_sites * DIM_VEL
        else:
            indices["site_rvel"] = jnp.array([], dtype=jnp.int32)

        return indices

    def calc_site_global(self, data: RelSitesInputSpace):
        assert (
            self.use_positions and self.use_rotations
        ), "Site positions and rotations must be used for global calculation."

        n_eval_samples = data.n_samples

        main_site_xpos = jnp.concatenate(
            [jnp.zeros((n_eval_samples, 2)), data.main_site_height.reshape(-1, 1)], axis=1
        ).reshape(n_eval_samples, 1, 3)

        if ROT_REPR == "rotvec":
            main_site_xmat = R.from_rotvec(data.main_site_xrot).as_matrix()
            site_rmat = R.from_rotvec(data.site_rangles).as_matrix()
        elif ROT_REPR == "sd":
            main_site_xmat = sd2mat(data.main_site_xrot)
            site_rmat = sd2mat(data.site_rangles)

        site_xpos = jnp.concatenate([main_site_xpos, data.site_rpos + main_site_xpos], axis=1)

        global_site_calc = lambda x, y: calculate_global_rotation_matrices(x, y, jnp)
        site_xmat = vmap(global_site_calc)(main_site_xmat, site_rmat)
        site_xmat = jnp.concatenate([main_site_xmat.reshape(n_eval_samples, 1, 3, 3), site_xmat], axis=1)

        return site_xpos, site_xmat


def calculate_relative_site_quatities(
    data, rel_site_ids, rel_body_ids, body_rootid, backend, rot_repr="rotvec"
):

    if backend == np:
        R = np_R
    else:
        R = jnp_R

    # get site positions and rotations
    site_xpos_traj = data.site_xpos
    site_xmat_traj = data.site_xmat
    site_xpos_traj = site_xpos_traj[rel_site_ids]
    site_xmat_traj = site_xmat_traj[rel_site_ids]

    # get relevant properties and calculate site velocities
    main_site_id = 0  # --> zeroth index in rel_site_ids
    del_indices = np.array([main_site_id])
    site_root_body_id = body_rootid[rel_body_ids]
    site_xvel = calc_site_velocities(rel_site_ids, data, rel_body_ids, site_root_body_id, backend)
    main_site_xvel = site_xvel[main_site_id]
    site_xvel = backend.delete(site_xvel, del_indices, axis=0)

    # calculate the rotation matrix from main site to the other sites
    main_site_xmat_traj = site_xmat_traj[main_site_id].reshape(3, 3)
    site_xmat_traj = backend.delete(site_xmat_traj, del_indices, axis=0).reshape(-1, 3, 3)
    rel_rot_mat = calculate_relative_rotation_matrices(main_site_xmat_traj, site_xmat_traj, backend)

    # calculate relative quantities
    main_site_xpos_traj = site_xpos_traj[main_site_id]
    site_xpos_traj = backend.delete(site_xpos_traj, del_indices, axis=0)
    site_rpos = calc_rel_positions(site_xpos_traj, main_site_xpos_traj, backend)

    site_rots = R.from_matrix(rel_rot_mat)
    if rot_repr == "rotvec":
        site_rangles = site_rots.as_rotvec()
    elif rot_repr == "sd":
        site_rmats = site_rots.as_matrix()
        site_rangles = site_rmats[..., :2].reshape(site_rmats.shape[:-2] + (-1,))
    else:
        raise NotImplementedError

    site_rvel = calculate_relative_velocity_in_local_frame(
        main_site_xvel, site_xvel, main_site_xmat_traj, rel_rot_mat, backend
    )

    return site_rpos, site_rangles, site_rvel


def normalized_vector(vec, backend=jnp):
    vec = backend.asarray(vec)
    norm_vec = broadcast_add_ones(backend.linalg.norm(vec, axis=-1), vec)
    return vec / norm_vec


def sd2mat(in_sd, backend=jnp):
    assert in_sd.shape[-1] == 6
    sd = in_sd.reshape(in_sd.shape[:-1] + (3, 2))
    a1, a2 = sd[..., :, 0], sd[..., :, 1]
    b1 = normalized_vector(a1, backend)
    out_mat = backend.zeros(in_sd.shape[:-1] + (3, 3))
    if backend == jnp:
        out_mat = out_mat.at[..., :, 0].set(b1)
    else:
        out_mat[..., :, 0] = b1

    b1dota2 = broadcast_add_ones(backend.einsum("...i,...i->...", b1, a2), b1)
    b2 = normalized_vector(a2 - b1dota2 * b1, backend)
    if backend == jnp:
        out_mat = out_mat.at[..., :, 1].set(b2)
        out_mat = out_mat.at[..., :, 2].set(backend.cross(b1, b2))
    else:
        out_mat[..., :, 1] = b2
        out_mat[..., :, 2] = backend.cross(b1, b2)
    return out_mat


def broadcast_add_ones(x, target):
    """
    Extend the x tensor with (target.ndim - 1) trailing ones
    """
    return jnp.expand_dims(x, axis=tuple(range(x.ndim, target.ndim)))
