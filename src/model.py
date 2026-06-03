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
    simulated_delay: int | None = None


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


def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


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
        scale_1, shift_1, gate_1 = jnp.split(self.adaln_1(adaln_cond), 3, axis=-1)
        scale_2, shift_2, gate_2 = jnp.split(self.adaln_2(adaln_cond), 3, axis=-1)

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
        self.simulated_delay = config.simulated_delay

        self.in_proj = nnx.Linear(action_dim + obs_dim, config.channel_dim, rngs=rngs)
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
        assert x_t.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_t.shape
        if time.ndim == 1:
            time = time[:, None]
        time = jnp.broadcast_to(time, (obs.shape[0], self.action_chunk_size))
        time_emb = jax.vmap(
            functools.partial(posemb_sincos, embedding_dim=self.channel_dim, min_period=4e-3, max_period=4.0)
        )(time)
        time_emb = self.time_mlp(time_emb)
        obs = einops.repeat(obs, "b e -> b c e", c=self.action_chunk_size)
        x = jnp.concatenate([x_t, obs], axis=-1)
        x = self.in_proj(x)
        for mlp in self.mlp_stack:
            x = mlp(x, time_emb)
        assert x.shape == (obs.shape[0], self.action_chunk_size, self.channel_dim), x.shape
        scale, shift = jnp.split(self.final_adaln(time_emb), 2, axis=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.out_proj(x)
        return x

    def action(self, rng: jax.Array, obs: jax.Array, num_steps: int) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry
            v_t = self(obs, x_t, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1

    def bid_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        n_samples: int,
        # when below two are None, it becomes backwards loss only (i.e., rejection sampling)
        bid_weak_policy: Self | None = None,
        bid_k: int | None = None,
    ) -> jax.Array:
        obs = einops.repeat(obs, "b ... -> (n b) ...", n=n_samples)
        weights = get_prefix_weights(inference_delay, prefix_attention_horizon, self.action_chunk_size, "exp")

        def backward_loss(action_chunks: jax.Array):
            error = jnp.linalg.norm(action_chunks - prev_action_chunk, axis=-1)  # [n, b, h]
            return jnp.sum(error * weights[None, None, :], axis=-1)  # [n, b]

        strong_actions = einops.rearrange(self.action(rng, obs, num_steps), "(n b) h d -> n b h d", n=n_samples)
        loss = backward_loss(strong_actions)  # [n, b]

        if bid_weak_policy is not None or bid_k is not None:
            assert bid_weak_policy is not None and bid_k is not None, (bid_weak_policy, bid_k)
            weak_actions = einops.rearrange(
                bid_weak_policy.action(rng, obs, num_steps), "(n b) h d -> n b h d", n=n_samples
            )
            weak_loss = backward_loss(weak_actions)  # [n, b]
            weak_idxs = jax.lax.top_k(-weak_loss.T, bid_k)[1].T  # [k, b]
            strong_idxs = jax.lax.top_k(-loss.T, bid_k)[1].T  # [k, b]
            a_plus = jnp.take_along_axis(strong_actions, strong_idxs[:, :, None, None], axis=0)  # [k, b, h, d]
            a_minus = jnp.take_along_axis(weak_actions, weak_idxs[:, :, None, None], axis=0)  # [k, b, h, d]
            # compute forward loss for each action in strong_actions
            forward_loss = jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_plus[None, :], axis=-1),  # [n, k, b, h]
                axis=(1, 3),  # [n, b]
            ) - jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_minus[None, :], axis=-1),  # [n, k, b, h]
                axis=(1, 3),  # [n, b]
            )
            loss += forward_loss / n_samples

        best_idxs = jnp.argmin(loss, axis=0)  # [b]
        return jnp.take_along_axis(strong_actions, best_idxs[None, :, None, None], axis=0).squeeze(0)  # [b, h, d]

    def realtime_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: PrefixAttentionSchedule,
        max_guidance_weight: float,
    ) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry

            @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))  # over batch
            def pinv_corrected_velocity(obs, x_t, y, t):
                def denoiser(x_t):
                    v_t = self(obs[None], x_t[None], t)[0]
                    return x_t + v_t * (1 - t), v_t

                x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
                weights = get_prefix_weights(
                    inference_delay, prefix_attention_horizon, self.action_chunk_size, prefix_attention_schedule
                )
                error = (y - x_1) * weights[:, None]
                pinv_correction = vjp_fun(error)[0]
                # constants from paper
                inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
                c = jnp.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                return v_t + guidance_weight * pinv_correction

            if self.simulated_delay is not None:
                mask = jnp.arange(self.action_chunk_size)[None, :] < inference_delay
                x_t = jnp.where(mask[:, :, None], prev_action_chunk, x_t)
                time_chunk = jnp.where(mask, 1.0, time)
                v_t = self(obs, x_t, time_chunk)
            else:
                v_t = pinv_corrected_velocity(obs, x_t, prev_action_chunk, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1

    def loss(self, rng: jax.Array, obs: jax.Array, action: jax.Array):
        assert action.dtype == jnp.float32
        assert action.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), action.shape
        noise_rng, time_rng, delay_rng = jax.random.split(rng, 3)
        time = jax.random.uniform(time_rng, (obs.shape[0],))
        noise = jax.random.normal(noise_rng, shape=action.shape)
        u_t = action - noise

        if self.simulated_delay is None:
            x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * action
            pred = self(obs, x_t, time)
            return jnp.mean(jnp.square(pred - u_t))

        w = jnp.exp(jnp.arange(0, self.simulated_delay)[::-1])
        w = w / jnp.sum(w)
        delay = jax.random.choice(delay_rng, self.simulated_delay, (obs.shape[0],), p=w)
        mask = jnp.arange(self.action_chunk_size)[None, :] < delay[:, None]
        time = jnp.where(mask, 1.0, time[:, None])
        x_t = (1 - time[:, :, None]) * noise + time[:, :, None] * action
        pred = self(obs, x_t, time)
        loss = jnp.square(pred - u_t)
        loss_mask = jnp.logical_not(mask)[:, :, None]
        return jnp.sum(loss * loss_mask) / (jnp.sum(loss_mask) + 1e-8)
