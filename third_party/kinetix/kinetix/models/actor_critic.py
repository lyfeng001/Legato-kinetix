import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import List, Sequence

import distrax

from kinetix.models.action_spaces import HybridActionDistribution, MultiDiscreteActionDistribution


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], 256),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=256)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=256)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class GeneralActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    fc_layer_depth: int
    fc_layer_width: int
    action_mode: str  # "continuous" or "discrete" or "hybrid"
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = False

    # Given an embedding, return the action/values, since this is shared across all models.
    @nn.compact
    def __call__(self, hidden, obs, embedding, dones, activation):

        if self.add_generator_embedding:
            raise NotImplementedError()

        if self.recurrent:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = embedding
        critic = embedding
        actor_mean_last = embedding
        for _ in range(self.fc_layer_depth):
            actor_mean = nn.Dense(
                self.fc_layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(actor_mean)
            actor_mean = activation(actor_mean)

            critic = nn.Dense(
                self.fc_layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(critic)
            critic = activation(critic)

        actor_mean_last = actor_mean
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        if self.action_mode == "discrete":
            pi = distrax.Categorical(logits=actor_mean)
        elif self.action_mode == "continuous":
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        elif self.action_mode == "multi_discrete":
            pi = MultiDiscreteActionDistribution(actor_mean, self.multi_discrete_number_of_dims_per_distribution)
        else:
            actor_mean_continuous = nn.Dense(
                self.hybrid_action_continuous_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean_last)
            actor_mean_sigma = jnp.exp(
                nn.Dense(self.hybrid_action_continuous_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(
                    actor_mean_last
                )
            )
            pi = HybridActionDistribution(actor_mean, actor_mean_continuous, actor_mean_sigma)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCriticPixelsRNN(nn.Module):

    action_dim: Sequence[int]
    fc_layer_depth: int
    fc_layer_width: int
    action_mode: str
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    activation: str
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = True

    @nn.compact
    def __call__(self, hidden, x, **kwargs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        og_obs, dones = x

        if self.add_generator_embedding:
            obs = og_obs.obs
        else:
            obs = og_obs

        image = obs.image
        global_info = obs.global_info

        x = nn.Conv(features=16, kernel_size=(8, 8), strides=(4, 4))(image)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        embedding = x.reshape(x.shape[0], x.shape[1], -1)

        embedding = jnp.concatenate([embedding, global_info], axis=-1)

        return GeneralActorCriticRNN(
            action_dim=self.action_dim,
            fc_layer_depth=self.fc_layer_depth,
            fc_layer_width=self.fc_layer_width,
            action_mode=self.action_mode,
            hybrid_action_continuous_dim=self.hybrid_action_continuous_dim,
            multi_discrete_number_of_dims_per_distribution=self.multi_discrete_number_of_dims_per_distribution,
            add_generator_embedding=self.add_generator_embedding,
            generator_embedding_number_of_timesteps=self.generator_embedding_number_of_timesteps,
            recurrent=self.recurrent,
        )(hidden, og_obs, embedding, dones, activation)

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)


class ActorCriticSymbolicRNN(nn.Module):
    action_dim: Sequence[int]
    fc_layer_width: int
    action_mode: str
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    fc_layer_depth: int
    activation: str
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = True

    @nn.compact
    def __call__(self, hidden, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        og_obs, dones = x
        if self.add_generator_embedding:
            obs = og_obs.obs
        else:
            obs = og_obs

        embedding = nn.Dense(
            self.fc_layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        return GeneralActorCriticRNN(
            action_dim=self.action_dim,
            fc_layer_depth=self.fc_layer_depth,
            fc_layer_width=self.fc_layer_width,
            action_mode=self.action_mode,
            hybrid_action_continuous_dim=self.hybrid_action_continuous_dim,
            multi_discrete_number_of_dims_per_distribution=self.multi_discrete_number_of_dims_per_distribution,
            add_generator_embedding=self.add_generator_embedding,
            generator_embedding_number_of_timesteps=self.generator_embedding_number_of_timesteps,
            recurrent=self.recurrent,
        )(hidden, og_obs, embedding, dones, activation)

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)
