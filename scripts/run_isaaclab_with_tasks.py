from __future__ import annotations

import sys
from pathlib import Path


CUSTOM_TASK_IMPORT = "import franka_panda_pick_and_place.tasks  # noqa: F401"
PLACEHOLDER = "# PLACEHOLDER: Extension template (do not remove this comment)"
ISAACLAB_TASKS_IMPORT = "import isaaclab_tasks  # noqa: F401"


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

    globals_dict = {
        "__file__": str(target_script),
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(source, str(target_script), "exec"), globals_dict)


if __name__ == "__main__":
    main()
