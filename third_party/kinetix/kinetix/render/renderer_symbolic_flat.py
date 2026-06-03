from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jax2d import joint
from jax2d.engine import select_shape
from jax2d.maths import rmat
from jax2d.sim_state import RigidBody
from jaxgl.maths import dist_from_line
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    fragment_shader_quad,
    fragment_shader_edged_quad,
    make_fragment_shader_texture,
    nearest_neighbour,
    make_fragment_shader_quad_textured,
)
from kinetix.render.renderer_symbolic_common import (
    make_circle_features,
    make_joint_features,
    make_polygon_features,
    make_thruster_features,
)
from kinetix.environment.env_state import StaticEnvParams, EnvParams, EnvState
from flax import struct


def make_render_symbolic(params, static_params: StaticEnvParams):
    def render_symbolic(state):

        n_polys = static_params.num_polygons
        nshapes = n_polys + static_params.num_circles

        polygon_features, polygon_mask = make_polygon_features(state, params, static_params)
        mask_to_ignore_walls_ceiling = np.ones(static_params.num_polygons, dtype=bool)
        mask_to_ignore_walls_ceiling[np.array([1, 2, 3])] = False

        polygon_features = polygon_features[mask_to_ignore_walls_ceiling]
        polygon_mask = polygon_mask[mask_to_ignore_walls_ceiling]

        circle_features, circle_mask = make_circle_features(state, params, static_params)
        joint_features, joint_idxs, joint_mask = make_joint_features(state, params, static_params)
        thruster_features, thruster_idxs, thruster_mask = make_thruster_features(state, params, static_params)

        two_J = joint_features.shape[0]
        J = two_J // 2  # for symbolic only have the one
        joint_features = jnp.concatenate(
            [
                joint_features[:J],  # shape (2 * J, K)
                jax.nn.one_hot(joint_idxs[:J, 0], nshapes),  # shape (2 * J, N)
                jax.nn.one_hot(joint_idxs[:J, 1], nshapes),  # shape (2 * J, N)
            ],
            axis=1,
        )
        thruster_features = jnp.concatenate(
            [
                thruster_features,
                jax.nn.one_hot(thruster_idxs, nshapes),
            ],
            axis=1,
        )

        polygon_features = jnp.where(polygon_mask[:, None], polygon_features, 0.0).flatten()
        circle_features = jnp.where(circle_mask[:, None], circle_features, 0.0).flatten()
        joint_features = jnp.where(joint_mask[:J, None], joint_features, 0.0).flatten()
        thruster_features = jnp.where(thruster_mask[:, None], thruster_features, 0.0).flatten()

        def _get_manifold_features(manifold):
            collision_mask_features = jnp.concatenate(
                [
                    manifold.normal,
                    jnp.expand_dims(manifold.penetration, axis=-1),
                    manifold.collision_point,
                    jnp.expand_dims(manifold.acc_impulse_normal, axis=-1),
                    jnp.expand_dims(manifold.acc_impulse_tangent, axis=-1),
                ],
                axis=-1,
            )

            return (collision_mask_features * manifold.active[..., None]).flatten()

        obs = jnp.concatenate(
            [
                polygon_features,
                circle_features,
                joint_features,
                thruster_features,
                jnp.array([state.gravity[1]]) / 10,
                # _get_manifold_features(state.acc_cc_manifolds),
                # _get_manifold_features(state.acc_cr_manifolds),
                # _get_manifold_features(state.acc_rr_manifolds),
            ],
            axis=0,
        )

        obs = jnp.clip(obs, a_min=-10.0, a_max=10.0)
        obs = jnp.nan_to_num(obs)
        return obs

    return render_symbolic
