from kinetix.models.actor_critic import (
    ActorCriticPixelsRNN,
    ActorCriticSymbolicRNN,
)
from kinetix.models.transformer_model import ActorCriticTransformer


def make_network_from_config(env, env_params, config, network_kws={}):

    env_name = config["env_name"]
    if "MultiDiscrete" in env_name:
        action_mode = "multi_discrete"
    elif "Discrete" in env_name:
        action_mode = "discrete"
    elif "Continuous" in env_name:
        action_mode = "continuous"
    elif "Hybrid" in env_name:
        action_mode = "hybrid"
    else:
        raise ValueError(f"Unknown action mode for {env_name}")
    action_dim = (
        env.action_space(env_params).shape[0] if action_mode == "continuous" else env.action_space(env_params).n
    )
    if "hybrid_action_continuous_dim" not in network_kws:
        network_kws["hybrid_action_continuous_dim"] = action_dim

    if "multi_discrete_number_of_dims_per_distribution" not in network_kws:
        num_joint_bindings = config["static_env_params"]["num_motor_bindings"]
        num_thruster_bindings = config["static_env_params"]["num_thruster_bindings"]
        network_kws["multi_discrete_number_of_dims_per_distribution"] = [3 for _ in range(num_joint_bindings)] + [
            2 for _ in range(num_thruster_bindings)
        ]
    network_kws["recurrent"] = config.get("recurrent_model", True)

    if "Pixels" in env_name:
        cls_to_use = ActorCriticPixelsRNN
    elif "Symbolic" in env_name or "Blind" in env_name:
        cls_to_use = ActorCriticSymbolicRNN

    if "Entity" in env_name:
        network = ActorCriticTransformer(
            action_dim=action_dim,
            fc_layer_width=config["fc_layer_width"],
            fc_layer_depth=config["fc_layer_depth"],
            action_mode=action_mode,
            num_heads=config["num_heads"],
            transformer_depth=config["transformer_depth"],
            transformer_size=config["transformer_size"],
            transformer_encoder_size=config["transformer_encoder_size"],
            aggregate_mode=config["aggregate_mode"],
            full_attention_mask=config["full_attention_mask"],
            activation=config["activation"],
            **network_kws,
        )
    else:
        network = cls_to_use(
            action_dim,
            fc_layer_width=config["fc_layer_width"],
            fc_layer_depth=config["fc_layer_depth"],
            activation=config["activation"],
            action_mode=action_mode,
            **network_kws,
        )

    return network
