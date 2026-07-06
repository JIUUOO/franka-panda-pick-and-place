from __future__ import annotations

import argparse
from pathlib import Path

import h5py


CAMERA_PATHS = (
    "obs/rgb_camera/camera_front",
    "obs/rgb_camera/camera_top",
    "obs/rgb_camera/camera_wrist",
)


def _shape(dataset) -> str:
    return f"shape={tuple(dataset.shape)}, dtype={dataset.dtype}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Task2 Isaac Lab HDF5 demos.")
    parser.add_argument("dataset_file", type=Path)
    args = parser.parse_args()

    with h5py.File(args.dataset_file, "r") as file:
        demo_names = sorted(file["data"].keys())
        print(f"Dataset: {args.dataset_file}")
        print(f"Num demos: {len(demo_names)}")

        for demo_name in demo_names:
            demo = file[f"data/{demo_name}"]
            print(f"\n[{demo_name}]")
            for key in ("actions", "processed_actions", "obs/joint_pos", "obs/gripper_pos", "obs/actions"):
                if key in demo:
                    print(f"  {key}: {_shape(demo[key])}")
            for camera_path in CAMERA_PATHS:
                if camera_path not in demo:
                    print(f"  {camera_path}: MISSING")
                    continue
                print(f"  {camera_path}: {_shape(demo[camera_path])}")


if __name__ == "__main__":
    main()
