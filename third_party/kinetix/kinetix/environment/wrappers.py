import functools
from chex._src.pytypes import PRNGKey
import jax
import jax.numpy as jnp
import chex
from jax.numpy import ndarray
import numpy as np
from flax import struct
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union, Any

from gymnax.environments import spaces, environment

from kinetix.environment.env_state import EnvParams, EnvState
from jaxued.environments import UnderspecifiedEnv


class UnderspecifiedEnvWrapper(UnderspecifiedEnv):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


# From Here: https://github.com/DramaCow/jaxued/blob/main/src/jaxued/wrappers/autoreset.py
class AutoResetWrapper(UnderspecifiedEnvWrapper):
    """
    This is a wrapper around an `UnderspecifiedEnv`, allowing for the environment to be automatically reset upon completion of an episode. This behaviour is similar to the default Gymnax interface. The user can specify a callable `sample_level` that takes in a PRNGKey and returns a level.

    Warning:
        To maintain compliance with UnderspecifiedEnv interface, user can reset to an
        arbitrary level. This includes levels outside the support of sample_level(). Consequently,
        the tagged rng is defaulted to jax.random.PRNGKey(0). If your code relies on this, careful
        attention may be required.
    """

    def __init__(self, env: UnderspecifiedEnv, sample_level: Callable[[chex.PRNGKey], EnvState]):
        self._env = env
        self.sample_level = sample_level

    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params

    def reset_env(self, rng, params):
        rng, rng_sample, rng_reset = jax.random.split(rng, 3)
        state_to_reset_to = self.sample_level(rng_sample)
        return self._env.reset_env_to_pcg_level(rng_reset, state_to_reset_to, params)

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:

        rng_reset, rng_step = jax.random.split(rng, 2)
        obs_st, env_state_st, reward, done, info = self._env.step_env(rng_step, state, action, params)
        obs_re, env_state_re = self.reset_env(rng_reset, params)

        env_state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st)
        obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)

        return obs, env_state, reward, done, info

    def reset_env_to_level(self, rng: chex.PRNGKey, level: EnvState, params: EnvParams) -> Tuple[Any, EnvState]:
        # raise NotImplementedError("This method should not be called directly. Use reset instead.")
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, env_state

    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)


class AutoReplayWrapper(UnderspecifiedEnv):
    """
    This wrapper replay the **same** level over and over again by resetting to the same level after each episode.
    This is useful for training/rolling out multiple times on the same level.
    """

    def __init__(self, env: UnderspecifiedEnv):
        self._env = env

    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        rng_reset, rng_step = jax.random.split(rng)
        obs_re, env_state_re = self._env.reset_to_level(rng_reset, state.level, params)
        obs_st, env_state_st, reward, done, info = self._env.step_env(rng_step, state.env_state, action, params)
        env_state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st)
        obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        return obs, state.replace(env_state=env_state), reward, done, info

    def reset_env_to_level(self, rng: chex.PRNGKey, level: EnvState, params: EnvParams) -> Tuple[Any, EnvState]:
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, AutoReplayState(env_state=env_state, level=level)

    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)


@struct.dataclass
class AutoReplayState:
    env_state: EnvState
    level: EnvState


class AutoReplayWrapper(UnderspecifiedEnvWrapper):
    """
    This wrapper replay the **same** level over and over again by resetting to the same level after each episode.
    This is useful for training/rolling out multiple times on the same level.
    """

    def __init__(self, env: UnderspecifiedEnv):
        self._env = env

    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        rng_reset, rng_step = jax.random.split(rng)
        obs_re, env_state_re = self._env.reset_to_level(rng_reset, state.level, params)
        obs_st, env_state_st, reward, done, info = self._env.step_env(rng_step, state.env_state, action, params)
        env_state = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st)
        obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        return obs, state.replace(env_state=env_state), reward, done, info

    def reset_env_to_level(self, rng: chex.PRNGKey, level: EnvState, params: EnvParams) -> Tuple[Any, EnvState]:
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, AutoReplayState(env_state=env_state, level=level)

    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)


class UnderspecifiedToGymnaxWrapper(environment.Environment):
    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def default_params(self) -> Any:
        return self._env.default_params

    def step_env(
        self, key: jax.Array, state: Any, action: int | float | jax.Array | ndarray | np.bool_ | np.number, params: Any
    ) -> Tuple[jax.Array | ndarray | np.bool_ | np.number | Any | Dict[Any, Any]]:
        return self._env.step_env(key, state, action, params)

    def reset_env(self, key: PRNGKey, params: Any) -> Tuple[PRNGKey | np.ndarray | np.bool_ | np.number | Any]:
        return self._env.reset_env(key, params)

    def action_space(self, params: Any):
        return self._env.action_space(params)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.reset_to_level_fn = jax.vmap(self._env.reset_to_level, in_axes=(0, 0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 3))
    def reset_to_level(self, rng, level, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_to_level_fn(rngs, level, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


@struct.dataclass
class DenseRewardState:
    env_state: EnvState
    last_distance: float = -1.0


class DenseRewardWrapper(GymnaxWrapper):
    def __init__(self, env, dense_reward_scale: float = 1.0) -> None:
        super().__init__(env)
        self.dense_reward_scale = dense_reward_scale

    def step(self, key, state, action: int, params=None):
        obs, env_state, reward, done, info = self._env.step_env(key, state.env_state, action, params)
        delta_dist = (
            -(info["distance"] - state.last_distance) * params.dense_reward_scale
        )  # if distance got less, then reward is positive

        delta_dist = jnp.nan_to_num(delta_dist, nan=0.0, posinf=0.0, neginf=0.0)
        reward = reward + jax.lax.select(
            (state.last_distance == -1) | (self.dense_reward_scale == 0.0), 0.0, delta_dist * self.dense_reward_scale
        )
        return obs, DenseRewardState(env_state, info["distance"]), reward, done, info

    def reset(self, rng, params=None):
        obs, env_state = self._env.reset(rng, params)
        return obs, DenseRewardState(env_state, -1.0)

    def reset_to_level(self, rng, level, params=None):
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, DenseRewardState(env_state, -1.0)


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    def reset_to_level(self, key: chex.PRNGKey, level: EnvState, params=None):
        obs, env_state = self._env.reset_to_level(key, level, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode_solved"] = info["GoalR"]
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info
