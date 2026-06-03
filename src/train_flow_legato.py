
import concurrent.futures
import dataclasses
import functools
import pathlib
import pickle
from typing import Sequence

import einops
from flax import struct
import flax.nnx as nnx
import imageio
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import generate_data
import model_legato as _model 
import train_expert

WANDB_PROJECT = "rtc-kinetix-bc-legato"  
LOG_DIR = pathlib.Path("logs-bc-legato") 


@dataclasses.dataclass(frozen=True)
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
    batch_size: int = 512
    num_epochs: int = 32
    seed: int = 0

    learning_rate: float = 3e-4
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000
    
    # Model architecture parameters
    channel_dim: int = 256
    channel_hidden_dim: int = 512
    token_hidden_dim: int = 64
    num_layers: int = 4
    action_chunk_size: int = 8
    
    # legato specific parameters
    warmup_min: int = 0
    warmup_max: int = 4
    inference_num_steps: int = 5 
    warmup_sampling: str = "exp"  # "bell", "uniform", "exp"


@struct.dataclass
class EpochCarry:
    rng: jax.Array
    train_state: nnx.State
    graphdef: nnx.GraphDef[tuple[_model.FlowPolicy, nnx.Optimizer]]


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    mesh = jax.make_mesh((jax.local_device_count(),), ("level",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("level"))

    model_config = _model.ModelConfig(
        channel_dim=config.channel_dim,
        channel_hidden_dim=config.channel_hidden_dim,
        token_hidden_dim=config.token_hidden_dim,
        num_layers=config.num_layers,
        action_chunk_size=config.action_chunk_size,
        warmup_min=config.warmup_min,
        warmup_max=config.warmup_max,
        inference_num_steps=config.inference_num_steps,  
        warmup_sampling=config.warmup_sampling, 
    )
    
    action_chunk_size = model_config.action_chunk_size

    def load_data(level_path: str):
        level_name = level_path.replace("/", "_").replace(".json", "")
        return dict(np.load(pathlib.Path(config.run_path) / "data" / f"{level_name}.npz"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(load_data, config.level_paths))
    with jax.default_device(jax.devices("cpu")[0]):
        data = jax.tree.map(lambda *x: einops.rearrange(jnp.stack(x), "l s e ... -> l (e s) ..."), *data)
        valid_steps = data["obs"].shape[1] - action_chunk_size + 1
        data = jax.tree.map(
            lambda x: x[:, : (valid_steps // config.batch_size) * config.batch_size + action_chunk_size - 1], data
        )
        data = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                x.shape,
                sharding,
                [
                    jax.device_put(y, d)
                    for y, d in zip(jnp.split(x, jax.local_device_count()), jax.local_devices(), strict=True)
                ],
            ),
            data,
        )

    data: generate_data.Data = generate_data.Data(**data)
    print(f"Truncated data to {data.obs.shape[1]:_} steps ({valid_steps // config.batch_size:_} batches)")

    obs_dim = data.obs.shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    @functools.partial(jax.jit, in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def init(rng: jax.Array) -> EpochCarry:
        rng, key = jax.random.split(rng)
        policy = _model.FlowPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=model_config,
            rngs=nnx.Rngs(key),
        )
        total_params = sum(x.size for x in jax.tree.leaves(nnx.state(policy, nnx.Param)))
        print(f"Total params: {total_params:,}")
        optimizer = nnx.Optimizer(
            policy,
            optax.chain(
                optax.clip_by_global_norm(config.grad_norm_clip),
                optax.adamw(
                    optax.warmup_constant_schedule(0, config.learning_rate, config.lr_warmup_steps),
                    weight_decay=config.weight_decay,
                ),
            ),
        )
        graphdef, train_state = nnx.split((policy, optimizer))
        return EpochCarry(rng, train_state, graphdef)

    @functools.partial(jax.jit, donate_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def train_epoch(epoch_carry: EpochCarry, level: kenv_state.EnvState, data: generate_data.Data):
        def train_minibatch(carry: tuple[jax.Array, nnx.State], batch_idxs: jax.Array):
            rng, train_state = carry
            policy, optimizer = nnx.merge(epoch_carry.graphdef, train_state)

            rng, key = jax.random.split(rng)

            def loss_fn(policy: _model.FlowPolicy):
                obs = data.obs[batch_idxs]
                action_chunks = data.action[batch_idxs[:, None] + jnp.arange(action_chunk_size)[None, :]]
                # zero actions after done
                done_chunks = data.done[batch_idxs[:, None] + jnp.arange(action_chunk_size)[None, :]]
                done_idxs = jnp.where(
                    jnp.any(done_chunks, axis=-1),
                    jnp.argmax(done_chunks, axis=-1),
                    action_chunk_size,
                )
                action_chunks = jnp.where(
                    jnp.arange(action_chunk_size)[None, :, None] >= done_idxs[:, None, None],
                    0.0,
                    action_chunks,
                )
                return policy.loss(key, obs, action_chunks)

            loss, grads = nnx.value_and_grad(loss_fn)(policy)
            info = {"loss": loss, "grad_norm": optax.global_norm(grads)}
            optimizer.update(grads)
            _, train_state = nnx.split((policy, optimizer))
            return (rng, train_state), info

        # shuffle
        rng, key = jax.random.split(epoch_carry.rng)
        permutation = jax.random.permutation(key, data.obs.shape[0] - action_chunk_size + 1)
        # batch
        permutation = permutation.reshape(-1, config.batch_size)
        # train
        (rng, train_state), train_info = jax.lax.scan(
            train_minibatch, (epoch_carry.rng, epoch_carry.train_state), permutation
        )
        train_info = jax.tree.map(lambda x: x.mean(), train_info)
        
        eval_info = {}
        video = None
        
        return EpochCarry(rng, train_state, epoch_carry.graphdef), ({**train_info, **eval_info}, video)

    wandb.init(project=WANDB_PROJECT)
    rng = jax.random.key(config.seed)
    epoch_carry = init(jax.random.split(rng, len(config.level_paths)))

    print(f"\n{'='*60}")
    print(f"Legato Training Configuration:")
    print(f"  Warmup range: [{config.warmup_min}, {config.warmup_max}]")
    print(f"  Warmup sampling: {config.warmup_sampling}")
    print(f"  Inference num steps: {config.inference_num_steps}")
    print(f"{'='*60}")
    
    for epoch_idx in tqdm.tqdm(range(config.num_epochs)):
        epoch_carry, (info, video) = train_epoch(epoch_carry, levels, data)

        for i in range(len(config.level_paths)):
            level_name = config.level_paths[i].replace("/", "_").replace(".json", "")
            wandb.log({f"{level_name}/{k}": v[i] for k, v in info.items()}, step=epoch_idx)

            log_dir = LOG_DIR / wandb.run.name / str(epoch_idx)

            if video is not None:
                video_dir = log_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_dir / f"{level_name}.mp4", video[i], fps=15)

            policy_dir = log_dir / "policies"
            policy_dir.mkdir(parents=True, exist_ok=True)
            level_train_state = jax.tree.map(lambda x: x[i], epoch_carry.train_state)
            with (policy_dir / f"{level_name}.pkl").open("wb") as f:
                policy, _ = nnx.merge(epoch_carry.graphdef, level_train_state)
                state_dict = nnx.state(policy).to_pure_dict()
                pickle.dump(state_dict, f)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Models saved to: {LOG_DIR / wandb.run.name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    tyro.cli(main)

