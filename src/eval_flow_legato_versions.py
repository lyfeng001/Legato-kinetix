
import collections
import dataclasses
import functools
import importlib
import math
import pathlib
import pickle
from typing import Sequence, Literal

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
import pandas as pd
import tyro

import train_expert
import model_legato 


@dataclasses.dataclass(frozen=True)
class LegatoMethodConfig:
    warmup: int = 0


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1 
    num_evals: int = 2048
    num_flow_steps: int = 5
    inference_delay: int = 0
    execute_horizon: int = 1
    method: LegatoMethodConfig = dataclasses.field(default_factory=LegatoMethodConfig)
    model: model_legato.ModelConfig = dataclasses.field(default_factory=lambda: model_legato.ModelConfig())
    version: Literal["legato"] = "legato"


def get_model_module(version: str):
    if version == "legato":
        import model_legato as model_module
        action_method_name = "action_legato"
    else:
        raise ValueError(f"Unknown version: {version}")
    
    return model_module, action_method_name


def eval_legato(
    config: EvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    policy, 
    action_method_name: str,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
):
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), config.num_evals
    )
    render_video = train_expert.make_render_video(renderer_pixels.make_render_pixels(env_params, static_env_params))
    assert config.execute_horizon >= config.inference_delay, f"{config.execute_horizon=} {config.inference_delay=}"

    action_df_method = getattr(policy, action_method_name)

    def execute_chunk(carry, _):
        def step(carry, action):
            rng, obs, env_state = carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            return (rng, next_obs, next_env_state), (done, env_state, info)

        rng, obs, env_state, action_chunk, n = carry
        rng, key = jax.random.split(rng)
        
        next_action_chunk = action_df_method(
            key,
            obs,
            config.num_flow_steps,
            action_chunk,
            config.method.warmup,
        )
        action_chunk_to_execute = jnp.concatenate(
            [
                action_chunk[:, : config.inference_delay],
                next_action_chunk[:, config.inference_delay : config.execute_horizon],
            ],
            axis=1,
        )
        
        next_action_chunk = jnp.concatenate(
            [
                next_action_chunk[:, config.execute_horizon :],
                jnp.zeros((obs.shape[0], config.execute_horizon, policy.action_dim)),
            ],
            axis=1,
        )
        next_n = jnp.concatenate([n[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])
        (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step, (rng, obs, env_state), action_chunk_to_execute.transpose(1, 0, 2)
        )
        return (rng, next_obs, next_env_state, next_action_chunk, next_n), (dones, env_states, infos)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)  # [batch, horizon, action_dim]
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)
    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n),
        None,
        length=scan_length,
    )
    dones, env_states, infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (dones, env_states, infos))
    assert dones.shape[0] >= env_params.max_timesteps, f"{dones.shape=}"

    return_info = {}
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        first_done_idx = jnp.argmax(dones, axis=0)
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    for key in ["match"]:
        if key in infos:
            return_info[key] = jnp.mean(infos[key])
    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    return return_info, video


def main(
    run_path: str,
    version: Literal["legato"] = "legato",
    config: EvalConfig = EvalConfig(),
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
    ),
    seed: int = 0,
    output_dir: str | None = None,  
    epochs: Sequence[int] | None = None, 
):
    if output_dir is None:
        output_dir = f"eval_output_{version}"
    
    model_module, action_method_name = get_model_module(version)
    
    print(f"\n{'='*60}")
    print(f"Evaluating - Version: {version.upper()}")
    print(f"Model Module: {model_module.__name__}")
    print(f"Action Method: {action_method_name}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*60}\n")
    
    config = dataclasses.replace(config, version=version)
    
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
    log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
    
    if epochs is None:
        if config.step == -1:
            epochs_to_eval = [len(log_dirs) - 1]  
        else:
            epochs_to_eval = [config.step]
    else:
        epochs_to_eval = list(epochs)
    
    print(f"\n{'='*60}")
    print(f"Evaluating epochs: {epochs_to_eval}")
    print(f"Available epochs: {[int(d.name) for d in log_dirs]}")
    print(f"{'='*60}\n")

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    def make_eval_fn(model_module_to_use):
        @functools.partial(jax.jit, static_argnums=(0, 1), in_shardings=sharding, out_shardings=sharding)
        @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, None, pspec, pspec, pspec), out_specs=pspec)
        @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
        def _eval(config: EvalConfig, action_method_name: str, rng: jax.Array, level: kenv_state.EnvState, state_dict):
            policy = model_module_to_use.FlowPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=config.model,
                rngs=nnx.Rngs(rng),
            )
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(state_dict)
            policy = nnx.merge(graphdef, state)
            eval_info, _ = eval_legato(config, env, rng, level, policy, action_method_name, env_params, static_env_params)
            return eval_info
        return _eval

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))
    
    _eval = make_eval_fn(model_module)
    
    for epoch_idx in epochs_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating Epoch {epoch_idx} - Version: {version.upper()}")
        print(f"{'='*60}\n")
        
        state_dicts = []
        for level_path in level_paths:
            level_name = level_path.replace("/", "_").replace(".json", "")
            checkpoint_path = log_dirs[epoch_idx] / "policies" / f"{level_name}.pkl"
            
            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
                continue
                
            with checkpoint_path.open("rb") as f:
                state_dicts.append(pickle.load(f))
        
        state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
        
        results = collections.defaultdict(list)
        
        print(f"Evaluating {version.upper()} for epoch {epoch_idx}")
        print(f"Action method: {action_method_name}\n")
        
        for inference_delay in [0, 1, 2, 3, 4]:
            for execute_horizon in range(max(1, inference_delay), 8 - inference_delay + 1):
                warmup = inference_delay
                prefix_attention_horizon = config.model.action_chunk_size - execute_horizon
                print(f"Testing: delay={inference_delay}, horizon={execute_horizon}, "
                      f"warmup={warmup} "
                      f"prefix_attn_horizon={prefix_attention_horizon}")
                c = dataclasses.replace(
                    config,
                    inference_delay=inference_delay,
                    execute_horizon=execute_horizon,
                    method=LegatoMethodConfig(warmup=warmup)
                )
                out = jax.device_get(_eval(c, action_method_name, rngs, levels, state_dicts))
                
                for i in range(len(level_paths)):
                    for k, v in out.items():
                        results[k].append(v[i])
                    results["version"].append(version)
                    results["epoch"].append(epoch_idx)
                    results["delay"].append(inference_delay)
                    results["execute_horizon"].append(execute_horizon)
                    results["warmup"].append(warmup)
                    results["prefix_attention_horizon"].append(prefix_attention_horizon)
                    results["level"].append(level_paths[i])
        
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        output_file = pathlib.Path(output_dir) / f"results_{version}_epoch_{epoch_idx}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        print(f"\n{'='*60}")
        print(f"Epoch {epoch_idx} Summary:")
        print(f"Average Return: {df['returned_episode_returns'].mean():.2f}")
        print(f"Average Solved: {df['returned_episode_solved'].mean():.2%}")
        print(f"{'='*60}\n")
    
    print(f"\n{'='*60}")
    print(f"All evaluations completed!")
    print(f"Version: {version.upper()}")
    print(f"Results saved to {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    tyro.cli(main)

