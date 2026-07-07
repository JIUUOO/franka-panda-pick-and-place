from __future__ import annotations

import argparse
import os
import pickle
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a LeRobot ACT policy in a custom Isaac Lab env.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--policy_path", type=Path, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--num_success_steps", type=int, default=1)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--step_hz", type=int, default=30)
parser.add_argument("--policy_device", default="cuda")
parser.add_argument("--max_translation", type=float, default=0.1)
parser.add_argument("--max_rotation", type=float, default=0.1)
parser.add_argument("--continuous_gripper", action="store_true")
parser.add_argument("--no_auto_server", action="store_true")
parser.add_argument("--lerobot_env", default="env_lerobot")
parser.add_argument("--lerobot_python", type=Path, default=None)
parser.add_argument("--server_start_timeout", type=float, default=120.0)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_tasks  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATH = PROJECT_ROOT / "source" / "franka_policy_isaac_lab"
sys.path.insert(0, str(PACKAGE_PATH))

import franka_policy_isaac_lab.tasks  # noqa: F401, E402


def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    bytes_received = 0
    while bytes_received < num_bytes:
        chunk = sock.recv(num_bytes - bytes_received)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data.")
        chunks.append(chunk)
        bytes_received += len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket):
    header = _recv_exact(sock, 8)
    (payload_size,) = struct.unpack("!Q", header)
    payload = _recv_exact(sock, payload_size)
    return pickle.loads(payload)


def send_message(sock: socket.socket, message) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


class PolicyClient:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port))

    def close(self) -> None:
        self.sock.close()

    def ping(self) -> None:
        send_message(self.sock, {"command": "ping"})
        response = recv_message(self.sock)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Policy ping failed."))

    def reset(self) -> None:
        send_message(self.sock, {"command": "reset"})
        response = recv_message(self.sock)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Policy reset failed."))

    def act(self, images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        send_message(self.sock, {"command": "act", "images": images, "state": state})
        response = recv_message(self.sock)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Policy action failed."))
        return np.asarray(response["action"], dtype=np.float32)

    def shutdown(self) -> None:
        send_message(self.sock, {"command": "shutdown"})
        recv_message(self.sock)


def _find_lerobot_python() -> str | None:
    if args_cli.lerobot_python is not None:
        return str(args_cli.lerobot_python)
    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda_exe is not None:
        return conda_exe
    return None


def _server_command() -> list[str]:
    script = PROJECT_ROOT / "scripts" / "serve_lerobot_policy.py"
    launcher = _find_lerobot_python()
    if launcher is None:
        raise RuntimeError(
            "Could not find conda or --lerobot_python. Start scripts/serve_lerobot_policy.py manually "
            "or pass --lerobot_python /path/to/env_lerobot/bin/python."
        )

    if Path(launcher).name == "conda":
        return [
            launcher,
            "run",
            "--no-capture-output",
            "-n",
            args_cli.lerobot_env,
            "python",
            "-u",
            str(script),
            "--policy_path",
            str(args_cli.policy_path),
            "--host",
            args_cli.host,
            "--port",
            str(args_cli.port),
            "--device",
            args_cli.policy_device,
        ]

    return [
        launcher,
        str(script),
        "--policy_path",
        str(args_cli.policy_path),
        "--host",
        args_cli.host,
        "--port",
        str(args_cli.port),
        "--device",
        args_cli.policy_device,
    ]


def _connect_policy_client() -> tuple[PolicyClient, subprocess.Popen | None]:
    if args_cli.no_auto_server:
        client = PolicyClient(args_cli.host, args_cli.port)
        client.ping()
        return client, None

    command = _server_command()
    print("[POLICY] starting server:")
    print(" ".join(command))
    process = subprocess.Popen(command)
    deadline = time.time() + args_cli.server_start_timeout
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited early with code {process.returncode}.")
        try:
            client = PolicyClient(args_cli.host, args_cli.port)
            client.ping()
            return client, process
        except OSError as exc:
            last_error = exc
            time.sleep(1.0)
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"Timed out connecting to policy server: {last_error}")


def _create_env_cfg() -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    return env_cfg, success_term


def _create_env(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    return gym.make(args_cli.task, cfg=env_cfg).unwrapped


def _to_numpy_image(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 4:
            image = image[0]
        image = image.numpy()
    return np.asarray(image)


def _to_numpy_state(state) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu()
        if state.ndim == 2:
            state = state[0]
        state = state.numpy()
    return np.asarray(state, dtype=np.float32)


def _extract_obs(obs: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    try:
        rgb_camera_obs = obs["rgb_camera"]
        state = obs["policy"]["joint_pos"]
    except Exception as exc:
        raise KeyError(f"Unexpected observation structure. Top-level keys: {list(obs.keys())}") from exc

    images = {
        f"observation.images.{camera_name}": _to_numpy_image(image)
        for camera_name, image in rgb_camera_obs.items()
    }

    return images, _to_numpy_state(state)


def _clip_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != 7:
        raise ValueError(f"Expected action shape (7,), got {action.shape}")
    action[:3] = np.clip(action[:3], -args_cli.max_translation, args_cli.max_translation)
    action[3:6] = np.clip(action[3:6], -args_cli.max_rotation, args_cli.max_rotation)
    action[6] = float(np.clip(action[6], -1.0, 1.0))
    if not args_cli.continuous_gripper:
        action[6] = 1.0 if action[6] >= 0.0 else -1.0
    return action


def _success_reached(env: gym.Env, success_term: object | None) -> bool:
    if success_term is None:
        return False
    return bool(success_term.func(env, **success_term.params)[0])


def _sleep(rate_start_time: float) -> float:
    if args_cli.step_hz <= 0:
        return time.time()
    next_time = rate_start_time + 1.0 / args_cli.step_hz
    remaining_time = next_time - time.time()
    while remaining_time > 0.0:
        time.sleep(min(0.01, remaining_time))
        remaining_time = next_time - time.time()
    return next_time if next_time > time.time() - 1.0 else time.time()


def main() -> None:
    client, server_process = _connect_policy_client()
    env_cfg, success_term = _create_env_cfg()
    env = _create_env(env_cfg)

    successes = 0
    try:
        for episode_idx in range(args_cli.num_episodes):
            if not simulation_app.is_running():
                break
            if server_process is not None and server_process.poll() is not None:
                raise RuntimeError(f"Policy server exited with code {server_process.returncode}.")

            seed = None if args_cli.seed is None else args_cli.seed + episode_idx
            obs, _ = env.reset(seed=seed)
            try:
                client.reset()
            except Exception as exc:
                if server_process is not None and server_process.poll() is not None:
                    raise RuntimeError(f"Policy server exited with code {server_process.returncode}.") from exc
                raise
            success_streak = 0
            episode_success = False
            rate_time = time.time()
            print(f"[EVAL] episode={episode_idx + 1}/{args_cli.num_episodes} reset seed={seed}")

            for step_idx in range(args_cli.max_steps):
                images, state = _extract_obs(obs)
                action = _clip_action(client.act(images, state))
                action_tensor = torch.tensor(action, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
                obs, _, _, _, _ = env.step(action_tensor)

                if _success_reached(env, success_term):
                    success_streak += 1
                    if success_streak >= args_cli.num_success_steps:
                        successes += 1
                        episode_success = True
                        print(f"[EVAL] episode={episode_idx + 1} success step={step_idx + 1}")
                        break
                else:
                    success_streak = 0

                if env.sim.is_stopped():
                    break
                rate_time = _sleep(rate_time)

            if not episode_success:
                print(f"[EVAL] episode={episode_idx + 1} failure max_steps={args_cli.max_steps}")

        print(f"[EVAL] success_rate={successes}/{args_cli.num_episodes} = {successes / max(args_cli.num_episodes, 1):.3f}")
    finally:
        env.close()
        client.close()
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()


if __name__ == "__main__":
    main()
    simulation_app.close()
