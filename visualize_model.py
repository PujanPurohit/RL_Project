import argparse
import os
import sys
import time
from pathlib import Path


def preconfigure_runtime(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["gif", "human"], default="gif")
    parser.add_argument("--gl-backend", choices=["egl", "glfw", "osmesa"])
    args, _ = parser.parse_known_args(argv)

    if args.gl_backend:
        backend = args.gl_backend
    elif os.environ.get("MUJOCO_GL"):
        backend = os.environ["MUJOCO_GL"]
    elif args.mode == "human":
        backend = "glfw"
    else:
        backend = "egl"

    os.environ["MUJOCO_GL"] = backend
    if backend in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)

    return backend


SELECTED_GL_BACKEND = preconfigure_runtime(sys.argv[1:])

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO, SAC

from config import ALGORITHM, REWARD_TYPE
from env_wrapper import make_env


ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a saved PPO/SAC model on the robotics environment."
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to a saved model .zip file. If omitted, saved models are listed.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("outputs/models"),
        help="Directory to scan when listing saved models.",
    )
    parser.add_argument(
        "--algo",
        choices=sorted(ALGO_MAP),
        default=ALGORITHM,
        help="Algorithm class used to load the model.",
    )
    parser.add_argument(
        "--reward-type",
        choices=["dense", "sparse"],
        default=REWARD_TYPE,
        help="Reward wrapper to use while visualizing.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to roll out.",
    )
    parser.add_argument(
        "--mode",
        choices=["gif", "human"],
        default="gif",
        help="Use 'gif' for SSH-friendly visualization or 'human' for local display.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output GIF path. Defaults to outputs/visualizations/<model_name>.gif.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Playback speed for GIF saving or human rendering.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-episode step cap override.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed to Stable Baselines3 load().",
    )
    parser.add_argument(
        "--gl-backend",
        choices=["egl", "glfw", "osmesa"],
        default=SELECTED_GL_BACKEND,
        help="OpenGL backend. 'egl' is recommended for GIF rendering over SSH.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for deterministic resets.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic policy actions.",
    )
    return parser.parse_args()


def list_saved_models(models_dir):
    zip_files = sorted(models_dir.glob("**/*.zip"))
    if not zip_files:
        print(f"No saved models found under {models_dir}")
        return

    print(f"Saved models under {models_dir}:")
    for path in zip_files:
        print(f"  {path}")


def normalize_frame(frame):
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def default_output_path(model_path):
    return Path("outputs/visualizations") / f"{model_path.stem}.gif"


def run_episode(model, env, deterministic, max_steps, fps, capture_frames, seed=None):
    obs, info = env.reset(seed=seed)
    episode_reward = 0.0
    steps = 0
    success = False
    frames = []

    if capture_frames:
        frame = env.render()
        if frame is not None:
            frames.append(normalize_frame(frame))
    else:
        env.render()

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += float(reward)
        steps += 1
        success = success or bool(info.get("is_success", False))

        if capture_frames:
            frame = env.render()
            if frame is not None:
                frames.append(normalize_frame(frame))
        else:
            env.render()
            if fps > 0:
                time.sleep(1.0 / fps)

        if max_steps is not None and steps >= max_steps:
            truncated = True

        if terminated or truncated:
            break

    return {
        "reward": episode_reward,
        "steps": steps,
        "success": success,
        "frames": frames,
    }


def save_gif(frames, output_path, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)


def main():
    args = parse_args()

    if args.model is None:
        list_saved_models(args.models_dir)
        return

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    if args.mode == "human" and args.gl_backend != "glfw":
        print(
            "Warning: human mode usually works best with --gl-backend glfw. "
            f"Current backend is {args.gl_backend}."
        )

    print(f"Using MuJoCo GL backend: {args.gl_backend}")

    render_mode = "rgb_array" if args.mode == "gif" else "human"
    env = make_env(reward_type=args.reward_type, render_mode=render_mode)

    try:
        model_cls = ALGO_MAP[args.algo]
        model = model_cls.load(str(args.model), env=env, device=args.device)

        all_frames = []
        summaries = []
        deterministic = not args.stochastic

        for episode_idx in range(args.episodes):
            episode_seed = None if args.seed is None else args.seed + episode_idx
            result = run_episode(
                model=model,
                env=env,
                deterministic=deterministic,
                max_steps=args.max_steps,
                fps=args.fps,
                capture_frames=args.mode == "gif",
                seed=episode_seed,
            )
            summaries.append(result)

            print(
                f"Episode {episode_idx + 1}: "
                f"reward={result['reward']:.2f}, "
                f"steps={result['steps']}, "
                f"success={result['success']}"
            )

            if args.mode == "gif" and result["frames"]:
                all_frames.extend(result["frames"])
                all_frames.extend([result["frames"][-1]] * max(args.fps // 2, 1))

        if summaries:
            mean_reward = np.mean([item["reward"] for item in summaries])
            success_rate = np.mean([item["success"] for item in summaries])
            print(
                f"Summary: mean_reward={mean_reward:.2f}, "
                f"success_rate={success_rate:.2%}"
            )

        if args.mode == "gif":
            if not all_frames:
                raise RuntimeError(
                    "No frames were captured. Try using a different render mode or "
                    "checking MuJoCo offscreen rendering support on this machine."
                )

            output_path = args.output or default_output_path(args.model)
            save_gif(all_frames, output_path, fps=args.fps)
            print(f"Saved GIF to {output_path}")

    except Exception as exc:
        if "OpenGL platform library" in str(exc):
            raise RuntimeError(
                "MuJoCo could not create an OpenGL context. On the lab PC, try "
                "`--mode gif --gl-backend egl`. If that still fails, retry with "
                "`--gl-backend osmesa`."
            ) from exc
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
