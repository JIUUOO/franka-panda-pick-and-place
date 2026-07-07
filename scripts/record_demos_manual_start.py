from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record Isaac Lab demos after a manual teleop start.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="gamepad", help="Teleop device name.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of successful demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous success steps required to export a demo.",
)
parser.add_argument("--start_key", type=str, default="SPACE", help="Keyboard key that starts recording.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio before launching the app.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    vars(args_cli)["xr"] = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
import omni.ui as ui


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATH = PROJECT_ROOT / "source" / "franka_policy_isaac_lab"
sys.path.insert(0, str(PACKAGE_PATH))

import franka_policy_isaac_lab.tasks  # noqa: F401, E402
from franka_policy_isaac_lab.recorders_cfg import ActionStateRGBRecorderManagerCfg  # noqa: E402


logger = logging.getLogger(__name__)


class RateLimiter:
    """Small rate limiter that keeps rendering responsive while sleeping."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def _setup_output() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def _create_env_cfg(output_dir: str, output_file_name: str) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object]:
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as exc:
        logger.error(f"Failed to parse environment configuration: {exc}")
        raise

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning("No terminations.success found. Successful demos cannot be exported automatically.")

    if args_cli.xr:
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRGBRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def _create_env(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    try:
        return gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as exc:
        logger.error(f"Failed to create environment: {exc}")
        raise


def _setup_teleop_device(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, callbacks: dict[str, Callable]) -> object:
    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        return create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)

    logger.warning(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5, sim_device=args_cli.device)
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5, sim_device=args_cli.device)
        )
    else:
        raise ValueError(f"Unsupported teleop device for manual-start recording: {args_cli.teleop_device}")

    for key, callback in callbacks.items():
        teleop_interface.add_callback(key, callback)
    return teleop_interface


def _setup_ui(env: gym.Env, label_text: str):
    if args_cli.xr:
        return None
    window = EmptyWindow(env, "Instruction")
    with window.ui_window_elements["main_vstack"]:
        return ui.Label(label_text)


def _set_label(label, text: str):
    if label is not None:
        label.text = text


def _start_prompt() -> str:
    if args_cli.teleop_device.lower() == "gamepad":
        return "Press A to start a new demo. Press LB+RB to discard/reset."
    return f"Press {args_cli.start_key} to start a new demo. Press R to discard/reset."


def _target_demo_number(current_count: int) -> int:
    return current_count + 1


def _reset_for_waiting(
    env: gym.Env,
    teleop_interface: object,
    label,
    current_count: int,
    reason: str = "ready",
    reset_env: bool = True,
) -> bool:
    if reset_env:
        env.sim.reset()
        env.recorder_manager.reset()
        env.reset()
    env.recorder_manager.reset()
    teleop_interface.reset()
    message = (
        f"Recorded {current_count} successful demonstrations. "
        f"Next demo: #{_target_demo_number(current_count)}.\n"
        f"{_start_prompt()}"
    )
    _set_label(label, message)
    print(f"[WAIT] {reason}. recorded={current_count}, next_demo=#{_target_demo_number(current_count)}")
    print(f"[WAIT] {_start_prompt()}")
    return reset_env


def _start_recording(env: gym.Env, teleop_interface: object, label, current_count: int, reset_env: bool):
    demo_number = _target_demo_number(current_count)
    if reset_env:
        print(f"[START] Demo #{demo_number}: resetting environment and clearing recorder buffer.")
        env.sim.reset()
        env.recorder_manager.reset()
        env.reset()
    else:
        print(f"[START] Demo #{demo_number}: using current reset state and clearing recorder buffer.")
    env.recorder_manager.reset()
    teleop_interface.reset()
    message = (
        f"Recording demo #{demo_number}. "
        "Move after the reset settles. Press discard/reset to skip this demo."
    )
    _set_label(label, message)
    print(f"[START] Demo #{demo_number}: recording is active from the first post-reset step.")


def _process_success(
    env: gym.Env, success_term: object | None, success_step_count: int, demo_number: int
) -> tuple[int, bool]:
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count == 1:
            print(f"[SUCCESS] Demo #{demo_number}: success condition detected.")
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print(
                f"[EXPORT] Demo #{demo_number}: exported after "
                f"{success_step_count}/{args_cli.num_success_steps} success steps."
            )
            return success_step_count, True
    else:
        if success_step_count > 0:
            print(f"[SUCCESS] Demo #{demo_number}: success streak reset.")
        success_step_count = 0

    return success_step_count, False


def _run_loop(env: gym.Env, env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, success_term: object | None):
    rate_limiter = None if args_cli.xr else RateLimiter(args_cli.step_hz)
    current_recorded_demo_count = 0
    success_step_count = 0
    recording_active = False
    start_requested = False
    reset_requested = False
    waiting_env_is_reset = False

    def request_start():
        nonlocal start_requested
        start_requested = True
        print(f"[INPUT] Start requested for demo #{_target_demo_number(current_recorded_demo_count)}.")

    def request_reset():
        nonlocal reset_requested
        reset_requested = True
        if recording_active:
            print(f"[INPUT] Discard/reset requested for demo #{_target_demo_number(current_recorded_demo_count)}.")
        else:
            print("[INPUT] Reset requested while waiting.")

    callbacks = {
        "START": request_start,
        "SPACE": request_start,
        args_cli.start_key: request_start,
        "R": request_reset,
        "RESET": request_reset,
    }
    teleop_interface = _setup_teleop_device(env_cfg, callbacks)
    teleop_interface.add_callback(args_cli.start_key, request_start)
    teleop_interface.add_callback("R", request_reset)

    label = _setup_ui(env, _start_prompt())
    print("[READY] Manual-start recording initialized.")
    print(f"[READY] Dataset file: {args_cli.dataset_file}")
    if args_cli.num_demos > 0:
        print(f"[READY] Target successful demos: {args_cli.num_demos}")
    else:
        print("[READY] Target successful demos: infinite")
    waiting_env_is_reset = _reset_for_waiting(
        env, teleop_interface, label, current_recorded_demo_count, reason="startup"
    )

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running():
            action = teleop_interface.advance()

            if start_requested:
                _start_recording(
                    env,
                    teleop_interface,
                    label,
                    current_recorded_demo_count,
                    reset_env=not waiting_env_is_reset,
                )
                recording_active = True
                waiting_env_is_reset = False
                success_step_count = 0
                start_requested = False
                reset_requested = False
                action = teleop_interface.advance()

            if reset_requested:
                discarded_demo_number = _target_demo_number(current_recorded_demo_count)
                if recording_active:
                    print(f"[DISCARD] Demo #{discarded_demo_number}: buffer cleared, demo not exported.")
                waiting_env_is_reset = _reset_for_waiting(
                    env, teleop_interface, label, current_recorded_demo_count, reason="reset/discard"
                )
                recording_active = False
                success_step_count = 0
                reset_requested = False
                start_requested = False
                continue

            if recording_active:
                demo_number = _target_demo_number(current_recorded_demo_count)
                actions = action.repeat(env.num_envs, 1)
                env.step(actions)
                success_step_count, success_reached = _process_success(
                    env, success_term, success_step_count, demo_number
                )
                if success_reached:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    print(f"[DONE] Demo #{demo_number}: total_successful={current_recorded_demo_count}.")
                    if args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
                        message = f"All {current_recorded_demo_count} demonstrations recorded. Exiting the app."
                        _set_label(label, message)
                        print(f"[FINISH] {message}")
                        break
                    waiting_env_is_reset = _reset_for_waiting(
                        env, teleop_interface, label, current_recorded_demo_count, reason="success"
                    )
                    recording_active = False
                    success_step_count = 0
            else:
                env.sim.render()

            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


def main() -> None:
    output_dir, output_file_name = _setup_output()
    env_cfg, success_term = _create_env_cfg(output_dir, output_file_name)
    env = _create_env(env_cfg)

    try:
        demo_count = _run_loop(env, env_cfg, success_term)
    finally:
        env.close()

    print(f"Recording session completed with {demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
