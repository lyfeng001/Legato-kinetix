import pathlib
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.render.renderer_pixels as renderer_pixels
import kinetix.util.saving as saving
import imageio

# Constants from train_expert.py
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


def load_levels(paths):
    static_env_params = kenv_state.StaticEnvParams(**LARGE_ENV_PARAMS, frame_skip=FRAME_SKIP)
    env_params = kenv_state.EnvParams()

    levels = []
    for level_path in paths:
        level, level_static_env_params, level_env_params = saving.load_from_json_file(level_path)
        # assert level_static_env_params == static_env_params, (
        #     f"Expected {static_env_params} got {level_static_env_params} for {level_path}"
        # )
        # assert level_env_params == env_params, f"Expected {env_params} got {level_env_params} for {level_path}"
        levels.append(level)
    return levels, static_env_params, env_params


def main():
    # Define level paths
    level_paths = [
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
    ]

    # Load levels
    levels, static_env_params, env_params = load_levels(level_paths)

    # Update screen dimensions
    static_env_params = static_env_params.replace(screen_dim=SCREEN_DIM, downscale=2)

    # Create environment and renderer
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    render_pixels = renderer_pixels.make_render_pixels(env_params, static_env_params)

    # Create output directory
    output_dir = pathlib.Path("rendered_levels")
    output_dir.mkdir(exist_ok=True)

    # Render each level
    for i, level in enumerate(levels):
        # Reset environment to level
        _, env_state = env.reset_to_level(jax.random.key(0), level, env_params)

        # Render the state
        image = render_pixels(env_state).round().astype(jnp.uint8).transpose(1, 0, 2)[::-1]

        # Save image
        level_name = level_paths[i].split("/")[-1].replace(".json", "")
        imageio.imwrite(output_dir / f"{level_name}.jpg", image)
        print(f"Saved {level_name}.jpg")


if __name__ == "__main__":
    main()
