
import dataclasses
import functools
from typing import Literal, TypeAlias, Self

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    channel_dim: int = 256
    channel_hidden_dim: int = 512
    token_hidden_dim: int = 64
    num_layers: int = 4
    action_chunk_size: int = 8
    warmup_min: int = 0
    warmup_max: int = 4
    inference_num_steps: int = 5
    warmup_sampling: str = "bell"  # "bell", "uniform", "exp"


def posemb_sincos(pos: jax.Array, embedding_dim: int, min_period: float, max_period: float) -> jax.Array:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]


def sample_bell_curve(rng: jax.Array, min_val: int, max_val: int, peak: int = 0) -> int:
    range_size = max_val - min_val
    
    def sample_from_beta():
        normalized_peak = (peak - min_val) / jnp.maximum(range_size, 1.0)
        concentration = 3.0  
        alpha_left = concentration * normalized_peak + 1
        beta_left = concentration * (1 - normalized_peak) + 1
 
        alpha_right = concentration * (1 - normalized_peak) + 1
        beta_right = concentration * normalized_peak + 1
        
        alpha = jnp.where(normalized_peak < 0.5, alpha_left, alpha_right)
        beta = jnp.where(normalized_peak < 0.5, beta_left, beta_right)

        beta_sample = jax.random.beta(rng, alpha, beta)

        continuous_value = min_val + beta_sample * (range_size + 1)
        sampled_value = jnp.floor(continuous_value).astype(jnp.int32)
        sampled_value = jnp.clip(sampled_value, min_val, max_val)
        
        return sampled_value
    
    return jnp.where(range_size == 0, min_val, sample_from_beta())


def sample_exp_decay(rng: jax.Array, min_val: int, max_val: int) -> int:

    range_size = max_val - min_val
    
    def sample_from_exp():
        candidates = jnp.arange(min_val, max_val + 1)
        weights = jnp.exp(-(candidates - min_val).astype(jnp.float32))
        probs = weights / jnp.sum(weights)
        sampled_value = jax.random.choice(rng, candidates, p=probs)
        
        return sampled_value

    return jnp.where(range_size == 0, min_val, sample_from_exp())


def build_weight_curve(
    rng: jax.Array,
    action_chunk_size: int,
    warmup_min: int,
    warmup_max: int,
    warmup_sampling: str = "bell"
) -> jax.Array:

    rng, key_warmup = jax.random.split(rng, 2)
    if warmup_sampling == "bell":
        warmup = sample_bell_curve(key_warmup, warmup_min, warmup_max, peak=warmup_min)
    elif warmup_sampling == "exp":
        warmup = sample_exp_decay(key_warmup, warmup_min, warmup_max)
    else: 
        warmup = jax.random.randint(key_warmup, (), warmup_min, warmup_max + 1)
    
    indices = jnp.arange(action_chunk_size, dtype=jnp.float32)
    normalized_pos = indices - warmup.astype(jnp.float32)
    
    def compute_weight(pos):
        return jnp.where(
            pos < 0,
            0.0,
            1.0
        )
    w = jax.vmap(compute_weight)(normalized_pos)
    return w


class MLPMixerBlock(nnx.Module):
    def __init__(
        self, token_dim: int, token_hidden_dim: int, channel_dim: int, channel_hidden_dim: int, *, rngs: nnx.Rngs
    ):
        self.token_mix_in = nnx.Linear(token_dim, token_hidden_dim, use_bias=False, rngs=rngs)
        self.token_mix_out = nnx.Linear(token_hidden_dim, token_dim, use_bias=False, rngs=rngs)
        self.channel_mix_in = nnx.Linear(channel_dim, channel_hidden_dim, use_bias=False, rngs=rngs)
        self.channel_mix_out = nnx.Linear(channel_hidden_dim, channel_dim, use_bias=False, rngs=rngs)
        self.norm_1 = nnx.LayerNorm(channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.norm_2 = nnx.LayerNorm(channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.adaln_1 = nnx.Linear(channel_dim, 3 * channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)
        self.adaln_2 = nnx.Linear(channel_dim, 3 * channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)

    def __call__(self, x: jax.Array, adaln_cond: jax.Array) -> jax.Array:
        scale_1, shift_1, gate_1 = jnp.split(self.adaln_1(adaln_cond)[:, None], 3, axis=-1)
        scale_2, shift_2, gate_2 = jnp.split(self.adaln_2(adaln_cond)[:, None], 3, axis=-1)

        # token mix
        residual = x
        x = self.norm_1(x) * (1 + scale_1) + shift_1
        x = x.transpose(0, 2, 1)
        x = self.token_mix_in(x)
        x = nnx.gelu(x)
        x = self.token_mix_out(x)
        x = x.transpose(0, 2, 1)
        x = residual + gate_1 * x

        # channel mix
        residual = x
        x = self.norm_2(x) * (1 + scale_2) + shift_2
        x = self.channel_mix_in(x)
        x = nnx.gelu(x)
        x = self.channel_mix_out(x)
        x = residual + gate_2 * x
        return x


class FlowPolicy(nnx.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: ModelConfig,
        rngs: nnx.Rngs,
    ):
        self.channel_dim = config.channel_dim
        self.action_dim = action_dim
        self.action_chunk_size = config.action_chunk_size
        self.config = config

        self.in_proj = nnx.Linear(action_dim + 1 + obs_dim, config.channel_dim, rngs=rngs)

        self.mlp_stack = [
            MLPMixerBlock(
                config.action_chunk_size,
                config.token_hidden_dim,
                config.channel_dim,
                config.channel_hidden_dim,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ]
        self.time_mlp = nnx.Sequential(
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
        )
        self.final_norm = nnx.LayerNorm(config.channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.final_adaln = nnx.Linear(
            config.channel_dim, 2 * config.channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        self.out_proj = nnx.Linear(config.channel_dim, action_dim, rngs=rngs)

    def __call__(self, obs: jax.Array, x_t: jax.Array, time: jax.Array) -> jax.Array:

        assert x_t.shape == (obs.shape[0], self.action_chunk_size, self.action_dim + 1), x_t.shape

        time_emb = posemb_sincos(
            jnp.broadcast_to(time, obs.shape[0]), self.channel_dim, min_period=4e-3, max_period=4.0
        )
        time_emb = self.time_mlp(time_emb)
        obs = einops.repeat(obs, "b e -> b c e", c=self.action_chunk_size)
        x = jnp.concatenate([x_t, obs], axis=-1)
        x = self.in_proj(x)
        for mlp in self.mlp_stack:
            x = mlp(x, time_emb)
        assert x.shape == (obs.shape[0], self.action_chunk_size, self.channel_dim), x.shape
        scale, shift = jnp.split(self.final_adaln(time_emb)[:, None], 2, axis=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.out_proj(x)
        return x

    def action(self, rng: jax.Array, obs: jax.Array, num_steps: int) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry
            w_ones = jnp.ones((*x_t.shape[:-1], 1))
            x_t_with_weight = jnp.concatenate([x_t, w_ones], axis=-1)
            v_t = self(obs, x_t_with_weight, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1

    def action_legato(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,
        warmup: int,
    ) -> jax.Array:

        dt = 1 / num_steps
        indices = jnp.arange(self.action_chunk_size, dtype=jnp.float32)
        normalized_pos = indices - float(warmup)

        def compute_weight(pos):
            return jnp.where(
                pos < 0,
                0.0,
                1.0
            )
        w = jax.vmap(compute_weight)(normalized_pos)
        w = w[None, :, None]
        pure_noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))

        def step(carry, _):
            x_t, time = carry
            guided_x_t = (1.0 - w) * prev_action_chunk + w * x_t
            w_expanded = jnp.broadcast_to(w, (guided_x_t.shape[0], self.action_chunk_size, 1))
            guided_x_t_with_weight = jnp.concatenate([guided_x_t, w_expanded], axis=-1)
            v_t = self(obs, guided_x_t_with_weight, time)
            return (guided_x_t + dt * v_t, time + dt), None
        
        (x_1, _), _ = jax.lax.scan(step, (pure_noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1

    def loss(self, rng: jax.Array, obs: jax.Array, action: jax.Array):

        assert action.dtype == jnp.float32
        assert action.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), action.shape
        
        batch_size = obs.shape[0]
        rng, noise_rng, time_rng, weight_rng = jax.random.split(rng, 4)
        time = jax.random.uniform(time_rng, (batch_size,))
        
        noise = jax.random.normal(noise_rng, shape=action.shape)
        def build_weight_for_sample(rng_i):
            return build_weight_curve(
                rng_i,
                self.action_chunk_size,
                self.config.warmup_min,
                self.config.warmup_max,
                self.config.warmup_sampling, 
            )
        
        weight_rngs = jax.random.split(weight_rng, batch_size)
        w = jax.vmap(build_weight_for_sample)(weight_rngs)
        w = w[:, :, None] 
        dt = 1.0 / float(self.config.inference_num_steps)
        kappa = (1.0 - w) / dt 
        time_expanded = time[:, None, None]
        x_t = (1 - time_expanded) * noise + time_expanded * action
        u_t = (action - noise) * (1.0 - kappa * (1 - time_expanded))
        guided_x_t = (1.0 - w) * action + w * x_t
        guided_x_t_with_weight = jnp.concatenate([guided_x_t, w], axis=-1)
        pred = self(obs, guided_x_t_with_weight, time)
        
        return jnp.mean(jnp.square(pred - u_t))

