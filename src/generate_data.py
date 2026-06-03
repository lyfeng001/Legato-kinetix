import dataclasses
import functools
import json
import pathlib
import pickle
from typing import Sequence

import einops
from flax import struct
import flax.nnx as nnx
import flax.serialization
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro

import train_expert


@dataclasses.dataclass
class Config:
    run_path: str
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
    seed: int = 0
    # Number of environments to run in parallel.
    num_envs: int = 128
    # Batch size for scan in number of steps *per environment*.
    batch_size: int = 256
    # Number of *total* steps to collect (lower bound -- rounded up to nearest multiple of batch size * num_envs).
    num_steps: int = 1_000_000

    solve_rate_threshold: float = 0.65
    action_sample_std: float | None = None


@struct.dataclass
class Data:
    obs: jax.Array
    action: jax.Array
    done: jax.Array
    solved: jax.Array
    return_: jax.Array
    length: jax.Array


@struct.dataclass
class StepCarry:
    rng: jax.Array
    obs: jax.Array
    env_state: kenv_state.EnvState
    policy_idxs: jax.Array


def main(config: Config):
    num_steps_per_env = (
        (config.num_steps // config.num_envs + config.batch_size - 1) // config.batch_size
    ) * config.batch_size
    print(
        f"Generating {num_steps_per_env * config.num_envs:_} steps with {config.num_envs} environments ({num_steps_per_env} steps per env)"
    )

    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(
            wrappers.AutoReplayWrapper(
                train_expert.ActionHistoryWrapper(
                    train_expert.ObsHistoryWrapper(train_expert.NoisyActionWrapper(env), 4)
                )
            )
        ),
        config.num_envs,
    )

    # load policies from best checkpoints by solve rate
    gen = np.random.default_rng(config.seed)
    state_dicts, good_policy_masks = [], []
    for level_path in config.level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        print(level_name)
        level_state_dicts, level_good_policy_mask = [], []
        for seed_dir in pathlib.Path(config.run_path).glob("seed_*"):
            # load stats
            log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), seed_dir.iterdir()))
            level_stats = [json.load((p / "stats" / f"{level_name}.json").open("r")) for p in log_dirs]
            level_stats = jax.tree.map(lambda *x: jnp.stack(x), *level_stats)
            # pick a random policy with solve rate >= threshold
            solved_idxs = np.nonzero(level_stats["returned_episode_solved"] >= config.solve_rate_threshold)[0]
            if len(solved_idxs) == 0:
                chosen_idx = np.argmax(level_stats["returned_episode_solved"])
                level_good_policy_mask.append(False)
            else:
                chosen_idx = gen.choice(solved_idxs)
                level_good_policy_mask.append(True)
            # load policy
            chosen_log_dir = log_dirs[chosen_idx]
            with open(chosen_log_dir / "policies" / f"{level_name}.pkl", "rb") as f:
                level_state_dicts.append(pickle.load(f))
            print(
                f"\t{seed_dir.name}: {level_stats['returned_episode_solved'][chosen_idx]:.3f} {'[MASKED]' if not level_good_policy_mask[-1] else ''}"
            )
        state_dicts.append(jax.tree.map(lambda *x: jnp.array(x), *level_state_dicts))
        good_policy_masks.append(level_good_policy_mask)

    state_dicts = jax.tree.map(lambda *x: jnp.array(x), *state_dicts)
    good_policy_masks = jnp.array(good_policy_masks)
    state_dicts, good_policy_masks = jax.device_put((state_dicts, good_policy_masks))

    def new_policy_idxs(rng: jax.Array, good_policy_mask: jax.Array) -> jax.Array:
        # select a random policy for each environment
        rng, key = jax.random.split(rng)
        randint = jax.random.randint(key, (config.num_envs,), 0, good_policy_mask.sum())
        return jnp.nonzero(good_policy_mask, size=good_policy_mask.shape[0])[0][randint]

    @jax.jit
    @jax.vmap
    def init(rng: jax.Array, level: kenv_state.EnvState, good_policy_mask: jax.Array) -> StepCarry:
        rng, key = jax.random.split(rng)
        obs, env_state = env.reset_to_level(key, level, env_params)
        rng, key = jax.random.split(rng)
        policy_idxs = new_policy_idxs(key, good_policy_mask)
        return StepCarry(rng, obs, env_state, policy_idxs)

    @functools.partial(jax.jit, static_argnums=(3,), donate_argnums=(0,))
    @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
    def step_n(carry: StepCarry, state_dict: dict, good_policy_mask: jax.Array, n: int):
        def step(carry: StepCarry, _):
            # create agent
            action_dim = env.action_space(env_params).shape[0]
            assert len(carry.obs.shape) == 2
            obs_dim = carry.obs.shape[1]

            @jax.vmap  # over environments
            def get_action(key, obs, policy_idx):
                agent = train_expert.Agent(obs_dim, action_dim, 1, rngs=nnx.Rngs(0))
                graphdef, state = nnx.split(agent)
                state.replace_by_pure_dict(jax.tree.map(lambda x: x[policy_idx], state_dict))
                agent = nnx.merge(graphdef, state)
                mean, std = agent.action(obs)
                if config.action_sample_std is not None:
                    std = jnp.full_like(mean, config.action_sample_std)
                action_dist = train_expert.make_squashed_normal_diag(mean, std, static_env_params.num_motor_bindings)
                return action_dist.sample(seed=key)

            # step
            rng, key = jax.random.split(carry.rng)
            action = get_action(jax.random.split(key, config.num_envs), carry.obs, carry.policy_idxs)
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, carry.env_state, action, env_params)

            # select new policies only at episode boundaries
            rng, key = jax.random.split(rng)
            next_policy_idxs = jnp.where(done, new_policy_idxs(key, good_policy_mask), carry.policy_idxs)

            # only retain important info
            info = {
                k: v
                for k, v in info.items()
                if k in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]
            }
            return StepCarry(rng, next_obs, next_env_state, next_policy_idxs), Data(
                train_expert.ObsHistoryWrapper.get_original_obs(carry.env_state),
                action,
                done,
                info["returned_episode_solved"],
                info["returned_episode_returns"],
                info["returned_episode_lengths"],
            )

        return jax.lax.scan(step, carry, None, length=n)

    rng = jax.random.key(config.seed)
    carry = init(jax.random.split(rng, len(config.level_paths)), levels, good_policy_masks)
    pbar = tqdm.tqdm(total=num_steps_per_env * config.num_envs, dynamic_ncols=True)
    data = []
    for _ in range(0, num_steps_per_env, config.batch_size):
        carry, result = step_n(carry, state_dicts, good_policy_masks, config.batch_size)
        data.append(jax.device_get(result))
        pbar.update(config.batch_size * config.num_envs)
    pbar.close()
    with jax.default_device(jax.devices("cpu")[0]):
        data: Data = jax.tree.map(
            lambda *x: einops.rearrange(
                jnp.stack(x),
                "num_batch level batch_size num_env ... -> level (num_batch batch_size) num_env ...",
            ),
            *data,
        )

    for i, level_path in enumerate(config.level_paths):
        level_name = level_path.replace("/", "_").replace(".json", "")
        print_info = {"num_episodes": data.done[i].sum()}
        for key in ["return_", "length", "solved"]:
            print_info[key] = (getattr(data, key)[i] * data.done[i]).sum() / print_info["num_episodes"]

        print(f"{level_name}:")
        for k, v in print_info.items():
            print(f"\t{k}: {v:.3f}")

        data_path = pathlib.Path(config.run_path) / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        level_data = flax.serialization.to_state_dict(jax.tree.map(lambda x: x[i], data))
        np.savez(data_path / f"{level_name}.npz", **level_data)


if __name__ == "__main__":
    tyro.cli(main)
