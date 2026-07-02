from __future__ import annotations

import sys
from pathlib import Path


CUSTOM_TASK_IMPORT = "import franka_panda_pick_and_place.tasks  # noqa: F401"
PLACEHOLDER = "# PLACEHOLDER: Extension template (do not remove this comment)"
ISAACLAB_TASKS_IMPORT = "import isaaclab_tasks  # noqa: F401"


def _patch_target_source(source: str, target_script: Path) -> str:
    if target_script.name == "teleop_se3_agent.py":
        old = "                    env.step(actions)"
        new = (
            "                    _, _, terminated, truncated, _ = env.step(actions)\n"
            "                    if torch.any(terminated | truncated):\n"
            "                        teleop_interface.reset()"
        )
        if old not in source:
            raise SystemExit(f"Could not patch automatic reset handling in: {target_script}")
        source = source.replace(old, new, 1)

    if target_script.name == "record_demos.py":
        old = "                success_step_count = handle_reset(env, success_step_count, instruction_display, label_text)"
        new = (
            "                success_step_count = handle_reset(env, success_step_count, instruction_display, label_text)\n"
            "                teleop_interface.reset()"
        )
        if old not in source:
            raise SystemExit(f"Could not patch recording reset handling in: {target_script}")
        source = source.replace(old, new, 1)

    return source


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_isaaclab_with_tasks.py <isaaclab-script.py> [args...]")

    repo_root = Path(__file__).resolve().parents[1]
    package_path = repo_root / "source" / "franka_panda_pick_and_place"
    sys.path.insert(0, str(package_path))

    target_script = Path(sys.argv[1])
    if not target_script.is_absolute():
        target_script = Path.cwd() / target_script
    if not target_script.exists():
        raise SystemExit(f"Isaac Lab script not found: {target_script}")

    sys.argv = [str(target_script), *sys.argv[2:]]
    source = target_script.read_text()
    if PLACEHOLDER in source:
        source = source.replace(PLACEHOLDER, CUSTOM_TASK_IMPORT)
    elif ISAACLAB_TASKS_IMPORT in source:
        source = source.replace(ISAACLAB_TASKS_IMPORT, f"{ISAACLAB_TASKS_IMPORT}\n{CUSTOM_TASK_IMPORT}", 1)
    else:
        raise SystemExit(f"Could not find an Isaac Lab task import hook in: {target_script}")
    source = _patch_target_source(source, target_script)

    globals_dict = {
        "__file__": str(target_script),
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(source, str(target_script), "exec"), globals_dict)


if __name__ == "__main__":
    main()
