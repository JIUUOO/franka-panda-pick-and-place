from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_CAMERA_PATH = "obs/rgb_camera/oblique_cam"


def _iter_datasets(group: h5py.Group, prefix: str = ""):
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Dataset):
            yield path, value
        elif isinstance(value, h5py.Group):
            yield from _iter_datasets(value, path)


def _format_shape(dataset: h5py.Dataset) -> str:
    return f"shape={tuple(dataset.shape)}, dtype={dataset.dtype}"


def _get_demo_names(file: h5py.File) -> list[str]:
    if "data" not in file:
        raise KeyError("Missing top-level 'data' group.")
    return sorted(file["data"].keys())


def _read_dataset(demo_group: h5py.Group, path: str) -> h5py.Dataset:
    if path not in demo_group:
        available = "\n".join(path for path, _ in _iter_datasets(demo_group))
        raise KeyError(f"Missing dataset '{path}'. Available datasets:\n{available}")
    value = demo_group[path]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"Path '{path}' is not a dataset.")
    return value


def _validate_demo(demo_name: str, demo_group: h5py.Group, camera_path: str) -> None:
    print(f"\n[{demo_name}]")
    print(f"  num_samples: {demo_group.attrs.get('num_samples', 'missing')}")
    print(f"  success: {demo_group.attrs.get('success', 'missing')}")

    actions = _read_dataset(demo_group, "actions")
    processed_actions = _read_dataset(demo_group, "processed_actions")
    camera = _read_dataset(demo_group, camera_path)

    print(f"  actions: {_format_shape(actions)}")
    print(f"  processed_actions: {_format_shape(processed_actions)}")
    print(f"  {camera_path}: {_format_shape(camera)}")

    action_steps = actions.shape[0]
    processed_action_steps = processed_actions.shape[0]
    camera_steps = camera.shape[0]
    if not (action_steps == processed_action_steps == camera_steps):
        raise ValueError(
            "Timestep count mismatch: "
            f"actions={action_steps}, processed_actions={processed_action_steps}, camera={camera_steps}"
        )
    if len(camera.shape) != 4 or camera.shape[-1] < 3:
        raise ValueError(f"Expected camera shape (T, H, W, C>=3), got {tuple(camera.shape)}")


def _save_preview_image(demo_group: h5py.Group, camera_path: str, output_path: Path, frame_index: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --preview. Install it or run without --preview.") from exc

    camera = _read_dataset(demo_group, camera_path)
    num_frames = camera.shape[0]
    if num_frames == 0:
        raise ValueError("Camera dataset has no frames.")
    if frame_index < 0:
        frame_index = num_frames + frame_index
    frame_index = max(0, min(frame_index, num_frames - 1))

    image = np.asarray(camera[frame_index])
    image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    print(f"\nSaved preview frame {frame_index} to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Isaac Lab HDF5 demos before LeRobot conversion.")
    parser.add_argument("dataset_file", type=Path, help="Path to an Isaac Lab HDF5 dataset.")
    parser.add_argument("--camera_path", default=DEFAULT_CAMERA_PATH, help="Camera dataset path inside each demo.")
    parser.add_argument("--list", action="store_true", help="Print all dataset paths.")
    parser.add_argument("--preview", type=Path, help="Optional PNG path for exporting one camera frame.")
    parser.add_argument("--preview_frame", type=int, default=0, help="Frame index to export with --preview.")
    args = parser.parse_args()

    with h5py.File(args.dataset_file, "r") as file:
        demo_names = _get_demo_names(file)
        if not demo_names:
            raise RuntimeError("No demos found under 'data/'.")

        print(f"Dataset: {args.dataset_file}")
        print(f"Num demos: {len(demo_names)}")
        print(f"Camera path: {args.camera_path}")

        if args.list:
            print("\nDatasets:")
            for demo_name in demo_names:
                for dataset_path, dataset in _iter_datasets(file[f"data/{demo_name}"]):
                    print(f"  data/{demo_name}/{dataset_path}: {_format_shape(dataset)}")

        for demo_name in demo_names:
            _validate_demo(demo_name, file[f"data/{demo_name}"], args.camera_path)

        if args.preview is not None:
            _save_preview_image(file[f"data/{demo_names[0]}"], args.camera_path, args.preview, args.preview_frame)


if __name__ == "__main__":
    main()
