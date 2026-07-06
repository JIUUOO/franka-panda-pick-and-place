from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


DEFAULT_CAMERA_PATH = "obs/rgb_camera/oblique_cam"
DEFAULT_STATE_PATH = "obs/joint_pos"
DEFAULT_ACTION_PATH = "actions"
DEFAULT_IMAGE_KEY = "observation.images.oblique_cam"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"
DEFAULT_TASK = "Pick the cube and place it inside the target square."


def _parse_camera_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid camera spec '{spec}'. Expected '<hdf5_path>:<lerobot_image_key>'."
        )
    camera_path, image_key = spec.split(":", 1)
    if not camera_path or not image_key:
        raise argparse.ArgumentTypeError(
            f"Invalid camera spec '{spec}'. Expected '<hdf5_path>:<lerobot_image_key>'."
        )
    return camera_path, image_key


def _camera_specs_from_args(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.camera:
        return args.camera
    return [(args.camera_path, args.image_key)]


def _import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "LeRobot is not installed in this Python environment.\n"
            "Install it first, then rerun this converter. Example:\n"
            "  pip install lerobot"
        ) from exc
    return LeRobotDataset


def _get_demo_names(file: h5py.File) -> list[str]:
    if "data" not in file:
        raise KeyError("Missing top-level 'data' group.")
    return sorted(file["data"].keys())


def _read_dataset(demo_group: h5py.Group, path: str) -> h5py.Dataset:
    if path not in demo_group:
        raise KeyError(f"Missing dataset '{path}' in demo '{demo_group.name}'.")
    value = demo_group[path]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"Path '{path}' in demo '{demo_group.name}' is not a dataset.")
    return value


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected image shape (H, W, C>=3), got {image.shape}.")
    image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _feature_names(prefix: str, size: int) -> list[str]:
    return [f"{prefix}_{index}" for index in range(size)]


def _get_feature_shapes(
    dataset_file: h5py.File,
    camera_specs: list[tuple[str, str]],
    state_path: str,
    action_path: str,
) -> tuple[dict[str, tuple[int, int, int]], int, int]:
    demo_names = _get_demo_names(dataset_file)
    if not demo_names:
        raise RuntimeError("No demos found under 'data/'.")

    first_demo = dataset_file[f"data/{demo_names[0]}"]
    state = _read_dataset(first_demo, state_path)
    action = _read_dataset(first_demo, action_path)

    image_shapes = {}
    for camera_path, image_key in camera_specs:
        camera = _read_dataset(first_demo, camera_path)
        if len(camera.shape) != 4 or camera.shape[-1] < 3:
            raise ValueError(f"Expected camera shape (T, H, W, C>=3), got {tuple(camera.shape)}.")
        image_shapes[image_key] = (camera.shape[1], camera.shape[2], 3)
    if len(state.shape) != 2:
        raise ValueError(f"Expected state shape (T, D), got {tuple(state.shape)}.")
    if len(action.shape) != 2:
        raise ValueError(f"Expected action shape (T, D), got {tuple(action.shape)}.")

    return image_shapes, state.shape[1], action.shape[1]


def _build_features(
    image_shapes: dict[str, tuple[int, int, int]],
    state_key: str,
    action_key: str,
    state_dim: int,
    action_dim: int,
    use_videos: bool,
) -> dict:
    features = {
        image_key: {
            "dtype": "video" if use_videos else "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }
        for image_key, image_shape in image_shapes.items()
    }
    features.update(
        {
        state_key: {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": _feature_names("joint", state_dim),
        },
        action_key: {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": _feature_names("action", action_dim),
        },
        }
    )
    return features


def _validate_lengths(
    demo_name: str,
    cameras: dict[str, h5py.Dataset],
    state: h5py.Dataset,
    action: h5py.Dataset,
) -> int:
    lengths = {
        **{camera_name: camera.shape[0] for camera_name, camera in cameras.items()},
        "state": state.shape[0],
        "action": action.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Timestep mismatch in {demo_name}: {lengths}")
    return action.shape[0]


def convert(args: argparse.Namespace) -> None:
    LeRobotDataset = _import_lerobot_dataset()
    camera_specs = _camera_specs_from_args(args)

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output root already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    with h5py.File(args.input_hdf5, "r") as input_file:
        image_shapes, state_dim, action_dim = _get_feature_shapes(
            input_file,
            camera_specs,
            args.state_path,
            args.action_path,
        )
        features = _build_features(
            image_shapes,
            args.state_key,
            args.action_key,
            state_dim,
            action_dim,
            use_videos=not args.no_videos,
        )

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=output_root,
            fps=args.fps,
            robot_type=args.robot_type,
            features=features,
            use_videos=not args.no_videos,
            image_writer_threads=args.image_writer_threads,
            batch_encoding_size=1,
        )

        demo_names = _get_demo_names(input_file)
        converted_episodes = 0
        converted_frames = 0

        try:
            for demo_name in demo_names:
                demo_group = input_file[f"data/{demo_name}"]
                if args.success_only and not bool(demo_group.attrs.get("success", False)):
                    print(f"Skipping {demo_name}: success={demo_group.attrs.get('success', 'missing')}")
                    continue

                cameras = {
                    image_key: _read_dataset(demo_group, camera_path)
                    for camera_path, image_key in camera_specs
                }
                state = _read_dataset(demo_group, args.state_path)
                action = _read_dataset(demo_group, args.action_path)
                num_frames = _validate_lengths(demo_name, cameras, state, action)

                print(f"Converting {demo_name}: {num_frames} frames")
                for frame_index in range(num_frames):
                    frame = {
                        **{
                            image_key: _ensure_rgb(camera[frame_index])
                            for image_key, camera in cameras.items()
                        },
                        args.state_key: np.asarray(state[frame_index], dtype=np.float32),
                        args.action_key: np.asarray(action[frame_index], dtype=np.float32),
                        "task": args.task,
                    }
                    dataset.add_frame(frame)

                dataset.save_episode()
                converted_episodes += 1
                converted_frames += num_frames
        finally:
            dataset.finalize()

    print("\nConversion complete")
    print(f"  input: {args.input_hdf5}")
    print(f"  output_root: {output_root}")
    print(f"  repo_id: {args.repo_id}")
    print(f"  episodes: {converted_episodes}")
    print(f"  frames: {converted_frames}")
    for camera_path, image_key in camera_specs:
        print(f"  image: {camera_path} -> {image_key}")
    print(f"  state: {args.state_path} -> {args.state_key}")
    print(f"  action: {args.action_path} -> {args.action_key}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab HDF5 demos to a local LeRobotDataset.",
        allow_abbrev=False,
    )
    parser.add_argument("input_hdf5", type=Path, help="Input Isaac Lab HDF5 dataset.")
    parser.add_argument("--output_root", type=Path, required=True, help="Local output directory for LeRobotDataset.")
    parser.add_argument("--repo_id", required=True, help="LeRobot dataset repo id, e.g. local/franka_pick_place.")
    parser.add_argument("--fps", type=int, default=50, help="Dataset FPS. Isaac Lab env step is 0.02s by default.")
    parser.add_argument("--robot_type", default="franka_panda", help="Robot type metadata for LeRobot.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task language string stored per frame.")
    parser.add_argument(
        "--camera",
        action="append",
        type=_parse_camera_spec,
        help=(
            "Input/output camera pair as '<hdf5_path>:<lerobot_image_key>'. "
            "Can be passed multiple times. Overrides --camera_path/--image_key."
        ),
    )
    parser.add_argument("--camera_path", default=DEFAULT_CAMERA_PATH, help="Input HDF5 camera dataset path.")
    parser.add_argument("--state_path", default=DEFAULT_STATE_PATH, help="Input HDF5 state dataset path.")
    parser.add_argument("--action_path", default=DEFAULT_ACTION_PATH, help="Input HDF5 action dataset path.")
    parser.add_argument("--image_key", default=DEFAULT_IMAGE_KEY, help="Output LeRobot image feature key.")
    parser.add_argument("--state_key", default=DEFAULT_STATE_KEY, help="Output LeRobot state feature key.")
    parser.add_argument("--action_key", default=DEFAULT_ACTION_KEY, help="Output LeRobot action feature key.")
    parser.add_argument("--image_writer_threads", type=int, default=4, help="Async image writer threads.")
    parser.add_argument("--no_videos", action="store_true", help="Store image files instead of encoded videos.")
    parser.add_argument("--include_failed", action="store_true", help="Convert demos even if success attr is false.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output_root before conversion.")
    args = parser.parse_args()

    args.success_only = not args.include_failed
    convert(args)


if __name__ == "__main__":
    main()
