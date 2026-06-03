from functools import partial

from jax2d.engine import recalculate_mass_and_inertia, recompute_global_joint_positions, select_shape
from kinetix.environment.env_state import EnvState, StaticEnvParams
from kinetix.pcg.pcg_state import PCGState
import jax
import jax.numpy as jnp


def _process_tied_together_shapes(pcg_state: PCGState, sampled_state: EnvState, static_params: StaticEnvParams):

    # Get the matrix of tied together positions. Since we vmap, we only want one entry active for any (i, j, k). Thus, we mask out some of the duplicate ones.
    tied = jnp.triu(pcg_state.tied_together & jnp.logical_not(jnp.eye(pcg_state.tied_together.shape[0], dtype=bool)))
    has_anything_in_column = tied.any(axis=0)
    tied = (
        tied * jnp.logical_not(has_anything_in_column)[:, None]
    )  # if there is something in a column, it means a previous one with a lower index has already been processed

    should_use_delta_positions = tied.any(axis=0)

    # This is the delta we have moved after sampling
    delta_positions = jnp.concatenate(
        [
            sampled_state.polygon.position - pcg_state.env_state.polygon.position,
            sampled_state.circle.position - pcg_state.env_state.circle.position,
        ]
    )

    def _get_effect_of_shape_i_on_all_others(item_index, item_row_of_what_is_tied):
        delta_pos = delta_positions[item_index]
        return jnp.arange(len(item_row_of_what_is_tied)), delta_pos[None] * item_row_of_what_is_tied[:, None]

    indices, positions = jax.vmap(_get_effect_of_shape_i_on_all_others, (0, 0))(jnp.arange(tied.shape[0]), tied)
    indices = indices.flatten()
    positions = positions.reshape(indices.shape[0], -1)

    default_positions = jnp.concatenate(
        [pcg_state.env_state.polygon.position, pcg_state.env_state.circle.position], axis=0
    )
    sampled_positions = jnp.concatenate([sampled_state.polygon.position, sampled_state.circle.position], axis=0)

    updated_positions = default_positions.at[indices].add(positions)
    # Use the deltas or the sampled positions
    positions = jnp.where(should_use_delta_positions[:, None], updated_positions, sampled_positions)

    sampled_state = sampled_state.replace(
        polygon=sampled_state.polygon.replace(position=positions[: static_params.num_polygons]),
        circle=sampled_state.circle.replace(position=positions[static_params.num_polygons :]),
    )
    return sampled_state


@partial(jax.jit, static_argnums=(3,))
def sample_pcg_state(rng, pcg_state: PCGState, params, static_params):
    def _pcg_fn(rng, main_val, max_val, mask):
        pcg_val = jax.random.uniform(rng, shape=main_val.shape) * (
            max_val.astype(float) - main_val.astype(float)
        ) + main_val.astype(float)
        if jnp.issubdtype(main_val.dtype, jnp.integer) or jnp.issubdtype(main_val.dtype, jnp.bool_):
            pcg_val = jnp.round(pcg_val)
        pcg_val = pcg_val.astype(main_val.dtype)
        new_val = jax.lax.select(mask.astype(bool), pcg_val, main_val)
        return new_val

    def _random_split_like_tree(rng, target):
        tree_def = jax.tree_structure(target)
        rngs = jax.random.split(rng, tree_def.num_leaves)
        return jax.tree_unflatten(tree_def, rngs)

    rng, _rng = jax.random.split(rng)
    rng_tree = _random_split_like_tree(_rng, pcg_state.env_state)

    sampled_state = jax.tree_util.tree_map(
        _pcg_fn, rng_tree, pcg_state.env_state, pcg_state.env_state_max, pcg_state.env_state_pcg_mask
    )

    sampled_state = _process_tied_together_shapes(pcg_state, sampled_state, static_params)

    sampled_state = recompute_global_joint_positions(sampled_state, static_params)

    env_state = recalculate_mass_and_inertia(
        sampled_state, static_params, sampled_state.polygon_densities, sampled_state.circle_densities
    )

    return env_state


def env_state_to_pcg_state(env_state: EnvState):
    N = env_state.polygon.active.shape[0] + env_state.circle.active.shape[0]
    pcg_state = PCGState(
        env_state=env_state,
        env_state_max=env_state,
        env_state_pcg_mask=jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=bool), env_state),
        tied_together=jnp.zeros((N, N), dtype=bool),
    )

    return pcg_state
