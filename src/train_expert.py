import dataclasses
import functools
import json
import pathlib
import pickle
from typing import Sequence

from flax import struct
import flax.nnx as nnx
import imageio
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
import kinetix.util.saving as saving
import optax
from tensorflow_probability.substrates import jax as tfp
import tqdm_loggable.auto as tqdm
import tyro
import wandb


@dataclasses.dataclass
class Config:
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    )
    seed: int = 32
    num_seeds: int = 8
    log_interval: int = 20
    num_updates: int = 1000
    num_steps: int = 256
    num_envs: int = 256
    num_minibatches: int = 8
    num_epochs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.9
    clip_eps: float = 0.2
    v_loss_coef: float = 0.5
    rpo_alpha: float = 0.3
    layer_width: int = 256
    grad_norm_clip: float = 1.0
    lr: float = 3e-4


LOG_DIR = pathlib.Path("logs-expert")
WANDB_PROJECT = "rtc-kinetix-expert"
LARGE_ENV_PARAMS = {
    "num_polygons": 12,
    "num_circles": 4,
    "num_joints": 6,
    "num_thrusters": 2,
    "num_motor_bindings": 4,
    "num_thruster_bindings": 2,
}
FRAME_SKIP = 2
SCREEN_DIM = (512, 512)
ACTION_NOISE_STD = 0.1
LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0
MEAN_MAX_MAGNITUDE = 5
MAX_ACTION = 0.99999


class BatchEnvWrapper(wrappers.GymnaxWrapper):
    """Define our own BatchEnvWrapper (we don't need different levels)"""

    def __init__(self, env, num: int):
        super().__init__(env)
        self.num = num

    def reset(self, rng, params):
        return jax.vmap(self._env.reset, in_axes=(0, None))(jax.random.split(rng, self.num), params)

    def reset_to_level(self, rng, level, params):
        return jax.vmap(self._env.reset_to_level, in_axes=(0, None, None))(
            jax.random.split(rng, self.num), level, params
        )

    def step(self, rng, state, action, params):
        return jax.vmap(self._env.step, in_axes=(0, 0, 0, None))(jax.random.split(rng, self.num), state, action, params)


@struct.dataclass
class DenseRewardState:
    env_state: kenv_state.EnvState
    timestep: int
    action: jax.Array


class DenseRewardWrapper(wrappers.GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step_env(key, state.env_state, action, params)
        dist_penalty = jax.lax.select(reward > 0, 0.0, info["distance"] / 6.0 / params.max_timesteps)
        new_reward = reward - jax.lax.select(done, (params.max_timesteps - state.timestep) * dist_penalty, dist_penalty)

        next_timestep = jax.lax.select(done, 0, state.timestep + 1)
        return obs, DenseRewardState(env_state, next_timestep, action), new_reward, done, info

    def reset(self, rng, params=None):
        obs, env_state = self._env.reset(rng, params)
        return obs, DenseRewardState(env_state, 0, jnp.zeros(self._env.action_space(params).shape[0]))

    def reset_to_level(self, rng, level, params=None):
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, DenseRewardState(env_state, 0, jnp.zeros(self._env.action_space(params).shape[0]))


class ActionHistoryWrapper(wrappers.UnderspecifiedEnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step_env(self, key, state, action, params):
        obs, env_state, reward, done, info = self._env.step_env(key, state, action, params)
        obs = jnp.concatenate([obs, action])
        return obs, env_state, reward, done, info

    def reset_to_level(self, rng, level, params):
        obs, env_state = self._env.reset_to_level(rng, level, params)
        obs = jnp.concatenate([obs, jnp.zeros(self._env.action_space(params).shape[0])])
        return obs, env_state

    def action_space(self, params):
        return self._env.action_space(params)


class NoisyActionWrapper(wrappers.UnderspecifiedEnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step_env(self, key, state, action, params):
        key1, key2 = jax.random.split(key)
        action = action + jax.random.normal(key1, action.shape) * ACTION_NOISE_STD
        return self._env.step_env(key2, state, action, params)

    def reset_to_level(self, rng, level, params):
        return self._env.reset_to_level(rng, level, params)

    def action_space(self, params):
        return self._env.action_space(params)


@struct.dataclass
class StickyActionState:
    env_state: kenv_state.EnvState
    action: jax.Array


class StickyActionWrapper(wrappers.UnderspecifiedEnvWrapper):
    def __init__(self, env, stickiness: float):
        super().__init__(env)
        self.stickiness = stickiness

    def step_env(self, key, state, action, params):
        key1, key2 = jax.random.split(key)
        actual_action = jax.lax.select(jax.random.bernoulli(key1, self.stickiness), state.action, action)
        obs, env_state, reward, done, info = self._env.step_env(key2, state.env_state, actual_action, params)
        return obs, StickyActionState(env_state, actual_action), reward, done, info

    def reset_to_level(self, rng, level, params):
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, StickyActionState(
            env_state,
            jnp.zeros(
                len(self._env.action_space(params).number_of_dims_per_distribution),
                dtype=jnp.int32,
            ),
        )

    def action_space(self, params):
        return self._env.action_space(params)


@struct.dataclass
class ObsHistoryState:
    env_state: kenv_state.EnvState
    stacked_obs: jax.Array
    original_obs: jax.Array


class ObsHistoryWrapper(wrappers.UnderspecifiedEnvWrapper):
    def __init__(self, env, history_length: int):
        super().__init__(env)
        self.history_length = history_length

    def step_env(self, key, state, action, params):
        obs, env_state, reward, done, info = self._env.step_env(key, state.env_state, action, params)
        stacked_obs = jnp.roll(state.stacked_obs, -1, axis=0).at[-1].set(obs)
        return stacked_obs.flatten(), ObsHistoryState(env_state, stacked_obs, obs), reward, done, info

    def reset_to_level(self, rng, level, params):
        obs, env_state = self._env.reset_to_level(rng, level, params)
        stacked_obs = jnp.repeat(obs[None], self.history_length, axis=0)
        return stacked_obs.flatten(), ObsHistoryState(env_state, stacked_obs, obs)

    def action_space(self, params):
        return self._env.action_space(params)

    @staticmethod
    def get_original_obs(env_state) -> jax.Array:
        while not isinstance(env_state, ObsHistoryState):
            env_state = env_state.env_state
        return env_state.original_obs


def make_squashed_normal_diag(mean, std, num_motor_bindings: int):
    bijector = tfp.bijectors.Blockwise(
        [tfp.bijectors.Tanh(), tfp.bijectors.Sigmoid()],
        block_sizes=[num_motor_bindings, mean.shape[-1] - num_motor_bindings],
        maybe_changes_size=False,
        validate_args=True,
    )
    return tfp.distributions.TransformedDistribution(tfp.distributions.MultivariateNormalDiag(mean, std), bijector)


class Agent(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, layer_width: int, *, rngs: nnx.Rngs):
        self.critic = nnx.Sequential(
            nnx.Linear(obs_dim, layer_width, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)), rngs=rngs),
            nnx.tanh,
            nnx.Linear(layer_width, layer_width, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)), rngs=rngs),
            nnx.tanh,
            nnx.Linear(layer_width, 1, kernel_init=nnx.initializers.orthogonal(1), rngs=rngs),
        )
        self.actor = nnx.Sequential(
            nnx.Linear(obs_dim, layer_width, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)), rngs=rngs),
            nnx.tanh,
            nnx.Linear(layer_width, layer_width, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)), rngs=rngs),
            nnx.tanh,
            nnx.Linear(layer_width, action_dim, kernel_init=nnx.initializers.orthogonal(0.01), rngs=rngs),
        )
        self.logstd = nnx.Param(jnp.zeros(action_dim))

    def value(self, obs: jax.Array) -> jax.Array:
        return self.critic(obs)[..., 0]

    def action(self, obs: jax.Array):
        mean = jnp.clip(self.actor(obs), -MEAN_MAX_MAGNITUDE, MEAN_MAX_MAGNITUDE)
        std = jnp.exp(jnp.clip(self.logstd.value, LOG_STD_MIN, LOG_STD_MAX))
        return mean, std


@struct.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    done: jax.Array
    reward: jax.Array
    value: jax.Array
    log_prob: jax.Array
    info: jax.Array
    env_state: kenv_state.EnvState


@struct.dataclass
class StepCarry:
    """Environment-related information that must be carried through the rollout loop."""

    rng: jax.Array
    env_state: kenv_state.EnvState
    obs: jax.Array
    done: jax.Array


@struct.dataclass
class UpdateCarry:
    """Information that must be carried through the outermost update loop."""

    rng: jax.Array
    step_carry: StepCarry
    train_state: nnx.State
    graphdef: nnx.GraphDef[tuple[Agent, nnx.Optimizer]] = struct.field(pytree_node=False)


@struct.dataclass
class TrainCarry:
    rng: jax.Array
    train_state: nnx.State


def make_render_video(render_pixels):
    @jax.vmap
    def render_video(env_state):
        while not isinstance(env_state, kenv_state.EnvState):
            env_state = env_state.env_state
        return render_pixels(env_state).round().astype(jnp.uint8).transpose(1, 0, 2)[::-1]

    return render_video


def load_levels(paths: Sequence[str], static_env_params: kenv_state.StaticEnvParams, env_params: kenv_state.EnvParams):
    levels = []
    for level_path in paths:
        level, level_static_env_params, level_env_params = saving.load_from_json_file(level_path)
        assert level_static_env_params == static_env_params, (
            f"Expected {static_env_params} got {level_static_env_params} for {level_path}"
        )
        assert level_env_params == env_params, f"Expected {env_params} got {level_env_params} for {level_path}"
        levels.append(level)
    return jax.tree.map(lambda *x: jnp.stack(x), *levels)


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**LARGE_ENV_PARAMS, frame_skip=FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    env = BatchEnvWrapper(
        wrappers.LogWrapper(
            DenseRewardWrapper(
                wrappers.AutoReplayWrapper(ActionHistoryWrapper(ObsHistoryWrapper(NoisyActionWrapper(env), 4)))
            )
        ),
        config.num_envs,
    )

    levels = load_levels(config.level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=SCREEN_DIM)

    batch_size = config.num_envs * config.num_steps
    assert batch_size % config.num_minibatches == 0, "Batch size must be divisible by number of minibatches"
    minibatch_size = batch_size // config.num_minibatches
    print(f"Batch size: {batch_size}, minibatch size: {minibatch_size}")

    # create rendering function
    render_pixels = renderer_pixels.make_render_pixels(env_params, static_env_params)
    render_video = make_render_video(render_pixels)

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))

    @functools.partial(jax.jit, out_shardings=sharding)
    @functools.partial(jax.vmap, in_axes=(0, None))  # over seeds
    @jax.vmap  # over levels
    def init(rng: jax.Array, level: kenv_state.EnvState) -> UpdateCarry:
        # initial reset
        rng, key = jax.random.split(rng)
        obs, env_state = env.reset_to_level(key, level, env_params)

        # initialize agent
        action_dim = env.action_space(env_params).shape[0]
        assert len(obs.shape) == 2
        obs_dim = obs.shape[1]
        rng, key = jax.random.split(rng)
        agent = Agent(obs_dim, action_dim, config.layer_width, rngs=nnx.Rngs(key))
        optimizer = nnx.Optimizer(
            agent, optax.chain(optax.clip_by_global_norm(config.grad_norm_clip), optax.adam(config.lr))
        )
        graphdef, initial_train_state = nnx.split((agent, optimizer))

        update_rng, step_rng = jax.random.split(rng)
        return UpdateCarry(
            rng=update_rng,
            step_carry=StepCarry(
                rng=step_rng, env_state=env_state, obs=obs, done=jnp.zeros(config.num_envs, dtype=bool)
            ),
            train_state=initial_train_state,
            graphdef=graphdef,
        )

    # outermost PPO update loop
    def update(update_carry: UpdateCarry, _):
        agent, _ = nnx.merge(update_carry.graphdef, update_carry.train_state)

        # environment rollout loop
        def step(step_carry: StepCarry, _):
            rng, key = jax.random.split(step_carry.rng)
            # action = env.action_space(env_params).sample(key)
            mean, std = agent.action(step_carry.obs)
            action_dist = make_squashed_normal_diag(mean, std, static_env_params.num_motor_bindings)
            action = action_dist.sample(seed=key)
            action = jnp.clip(action, -MAX_ACTION, MAX_ACTION)
            log_prob = action_dist.log_prob(action)
            value = agent.value(step_carry.obs)
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, next_done, info = env.step(key, step_carry.env_state, action, env_params)
            return (
                StepCarry(rng=rng, env_state=next_env_state, obs=next_obs, done=next_done),
                Transition(
                    obs=step_carry.obs,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=step_carry.done,
                    info=info,
                    env_state=step_carry.env_state,
                ),
            )

        # transitions has shape: (NUM_STEPS, NUM_ENVS, ...)
        final_step_carry, transitions = jax.lax.scan(step, update_carry.step_carry, None, length=config.num_steps)

        # gae calculation loop
        def gae_step(carry, transition: Transition):
            gae, next_value, next_done = carry
            delta = transition.reward + config.gamma * next_value * (1 - next_done) - transition.value
            gae = delta + config.gamma * config.gae_lambda * (1 - next_done) * gae
            return (gae, transition.value, transition.done), gae

        last_value = agent.value(final_step_carry.obs)
        last_done = final_step_carry.done
        _, advantages = jax.lax.scan(
            gae_step, (jnp.zeros_like(last_value), last_value, last_done), transitions, reverse=True, unroll=16
        )
        returns = advantages + transitions.value

        # update epochs loop
        def update_epoch(epoch_carry: TrainCarry, _):
            # gradient update loop
            def update_minibatch(minibatch_carry: TrainCarry, minibatch: tuple[Transition, jax.Array, jax.Array]):
                agent, optimizer = nnx.merge(update_carry.graphdef, minibatch_carry.train_state)
                transitions, advantages, returns = minibatch
                rng, key = jax.random.split(minibatch_carry.rng)

                def loss_fn(agent: Agent):
                    mean, std = agent.action(transitions.obs)
                    # RPO LOGIC
                    z = jax.random.uniform(
                        key, transitions.action.shape, minval=-config.rpo_alpha, maxval=config.rpo_alpha
                    )
                    dist = make_squashed_normal_diag(mean + z, std, static_env_params.num_motor_bindings)
                    value = agent.value(transitions.obs)
                    log_prob = dist.log_prob(transitions.action)
                    log_ratio = log_prob - transitions.log_prob
                    ratio = jnp.exp(log_ratio)

                    # actor loss
                    norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    pg_loss1 = -norm_advantages * ratio
                    pg_loss2 = -norm_advantages * jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
                    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

                    # value loss
                    v_loss_unclipped = (value - returns) ** 2
                    v_clipped = transitions.value + (value - transitions.value).clip(-config.clip_eps, config.clip_eps)
                    v_loss_clipped = (v_clipped - returns) ** 2
                    v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()

                    loss = pg_loss + config.v_loss_coef * v_loss
                    info = {
                        "pg_loss": pg_loss,
                        "v_loss": v_loss,
                        "clipfrac": (jnp.abs(ratio - 1) > config.clip_eps).mean(),
                        "approx_kl": ((ratio - 1) - log_ratio).mean(),
                    }
                    return loss, info

                (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent)
                info["loss"] = loss
                info["grad_norm"] = optax.global_norm(grads)
                optimizer.update(grads)
                _, train_state = nnx.split((agent, optimizer))
                return TrainCarry(rng=rng, train_state=train_state), info

            # flatten data in preparation for learning
            data = jax.tree.map(
                lambda x: x.reshape(config.num_steps * config.num_envs, *x.shape[2:]),
                (transitions, advantages, returns),
            )
            # shuffle
            rng, key = jax.random.split(epoch_carry.rng)
            permutation = jax.random.permutation(key, config.num_envs * config.num_steps)
            data = jax.tree.map(lambda x: x[permutation], data)
            # batch
            batches = jax.tree.map(lambda x: x.reshape(config.num_minibatches, minibatch_size, *x.shape[1:]), data)
            # learn!
            final_carry, info = jax.lax.scan(update_minibatch, epoch_carry.replace(rng=rng), batches)
            return final_carry, info

        final_epoch_carry, info = jax.lax.scan(
            update_epoch,
            TrainCarry(rng=update_carry.rng, train_state=update_carry.train_state),
            None,
            length=config.num_epochs,
        )

        for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
            info[key] = (transitions.info[key] * transitions.info["returned_episode"]).sum() / transitions.info[
                "returned_episode"
            ].sum()
        info["reward"] = transitions.reward.mean()

        rollout = jax.tree.map(lambda x: x[:, 0], transitions.env_state)
        return UpdateCarry(
            final_epoch_carry.rng, final_step_carry, final_epoch_carry.train_state, update_carry.graphdef
        ), (info, rollout)

    @functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(jax.vmap, in_axes=(0, None))  # over seeds
    @functools.partial(jax.vmap, in_axes=(0, None))  # over levels
    def update_n(update_carry: UpdateCarry, num: int):
        update_carry, (info, rollout) = jax.lax.scan(update, update_carry, length=num)
        video = render_video(jax.tree.map(lambda x: x[0], rollout))
        return update_carry, (jax.tree.map(jnp.mean, info), video)

    wandb.init(project=WANDB_PROJECT)
    wandb.define_metric("num_env_steps")
    wandb.define_metric("*", step_metric="num_env_steps")
    pbar = tqdm.tqdm(total=config.num_updates * config.num_envs * config.num_steps, dynamic_ncols=True)

    num_levels = len(config.level_paths)
    rngs = jax.random.split(jax.random.key(config.seed), config.num_seeds * num_levels).reshape(
        config.num_seeds, num_levels
    )
    update_carry = init(rngs, levels)
    for update_idx in range(0, config.num_updates, config.log_interval):
        update_carry, (info, video) = update_n(update_carry, config.log_interval)
        if any(jnp.any(jnp.isnan(x)) for x in jax.tree.leaves(info)):
            raise ValueError(f"NaN detected at update {update_idx}")
        pbar.update(config.log_interval * config.num_envs * config.num_steps)
        wandb.log({"num_env_steps": pbar.n}, step=update_idx)
        for seed_idx in range(config.num_seeds):
            for level_idx in range(num_levels):
                level_name = config.level_paths[level_idx].replace("/", "_").replace(".json", "")
                level_info = jax.tree.map(lambda x: x[seed_idx, level_idx].item(), info)
                wandb.log({f"{level_name}/{seed_idx}/{k}": v for k, v in level_info.items()}, step=update_idx)

                log_dir = LOG_DIR / wandb.run.name / f"seed_{seed_idx}" / str(update_idx)
                stats_dir = log_dir / "stats"
                stats_dir.mkdir(parents=True, exist_ok=True)
                with (stats_dir / f"{level_name}.json").open("w") as f:
                    json.dump(level_info, f, indent=2)

                video_dir = log_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_dir / f"{level_name}.mp4", video[seed_idx, level_idx], fps=15)

                policy_dir = log_dir / "policies"
                policy_dir.mkdir(parents=True, exist_ok=True)
                level_train_state = jax.tree.map(lambda x: x[seed_idx, level_idx], update_carry.train_state)
                with (policy_dir / f"{level_name}.pkl").open("wb") as f:
                    agent, _ = nnx.merge(update_carry.graphdef, level_train_state)
                    state_dict = nnx.split(agent)[1].to_pure_dict()
                    pickle.dump(state_dict, f)


if __name__ == "__main__":
    tyro.cli(main)
