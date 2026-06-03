from dataclasses import field
import jax.numpy as jnp
from flax import struct

from jax2d.sim_state import SimState, SimParams, StaticSimParams, RigidBody, Joint, Thruster, CollisionManifold
from kinetix.environment.env_state import EnvState


@struct.dataclass
class PCGState:
    # Primary env state
    env_state: EnvState
    # The PCG mask.  If a value is truthy in this, then it is PCG not static
    env_state_pcg_mask: EnvState
    # In the case that a value is PCG, the env_state value is the min and this state represents the max
    env_state_max: EnvState

    tied_together: jnp.ndarray  # NxN matrix of booleans, where N is the number of shapes

    def __setstate__(self, state):
        if "tied_together" not in state:
            num_shapes = state["env_state"].polygon.active.shape[0] + state["env_state"].circle.active.shape[0]
            state["tied_together"] = jnp.zeros((num_shapes, num_shapes), dtype=bool)
        object.__setattr__(self, "__dict__", state)
