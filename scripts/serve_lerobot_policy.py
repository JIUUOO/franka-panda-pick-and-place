from __future__ import annotations

import argparse
import pickle
import socket
import struct
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)


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


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image_tensor = torch.from_numpy(np.asarray(image))
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image shape HxWxC or CxHxW, got {tuple(image_tensor.shape)}")
    if image_tensor.shape[-1] in (3, 4):
        image_tensor = image_tensor[..., :3].permute(2, 0, 1)
    elif image_tensor.shape[0] != 3:
        raise ValueError(f"Expected 3-channel image, got shape {tuple(image_tensor.shape)}")
    image_tensor = image_tensor.float()
    if image_tensor.max() > 1.0:
        image_tensor = image_tensor / 255.0
    return image_tensor.contiguous()


def _state_to_tensor(state: np.ndarray) -> torch.Tensor:
    state_tensor = torch.from_numpy(np.asarray(state)).float()
    if state_tensor.ndim == 2 and state_tensor.shape[0] == 1:
        state_tensor = state_tensor[0]
    if state_tensor.ndim != 1:
        raise ValueError(f"Expected state shape D, got {tuple(state_tensor.shape)}")
    return state_tensor.contiguous()


class PolicyServer:
    def __init__(self, policy_path: Path, device: str):
        self.policy_path = policy_path
        self.policy = ACTPolicy.from_pretrained(policy_path, local_files_only=True)
        self.policy.to(device)
        self.policy.eval()
        self.policy.reset()
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            policy_path,
            config_filename="policy_preprocessor.json",
            local_files_only=True,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            policy_path,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        print(f"[POLICY] loaded={policy_path}")
        print(f"[POLICY] device={next(self.policy.parameters()).device}")

    @torch.inference_mode()
    def act(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        batch = {
            "observation.images.oblique_cam": _image_to_tensor(image),
            "observation.state": _state_to_tensor(state),
        }
        batch = self.preprocessor(batch)
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def reset(self) -> None:
        self.policy.reset()


def run_server(policy_server: PolicyServer, host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen(1)
        print(f"[SERVER] listening on {host}:{port}")
        while True:
            client_sock, client_addr = server_sock.accept()
            print(f"[SERVER] client connected: {client_addr}")
            with client_sock:
                while True:
                    try:
                        request = recv_message(client_sock)
                    except ConnectionError:
                        print("[SERVER] client disconnected")
                        break

                    command = request.get("command", "act")
                    if command == "ping":
                        send_message(client_sock, {"ok": True})
                        continue
                    if command == "shutdown":
                        send_message(client_sock, {"ok": True})
                        print("[SERVER] shutdown requested")
                        return
                    if command == "reset":
                        policy_server.reset()
                        send_message(client_sock, {"ok": True})
                        continue
                    if command != "act":
                        send_message(client_sock, {"ok": False, "error": f"Unknown command: {command}"})
                        continue

                    try:
                        action = policy_server.act(request["image"], request["state"])
                        send_message(client_sock, {"ok": True, "action": action.tolist()})
                    except Exception as exc:
                        send_message(client_sock, {"ok": False, "error": repr(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a LeRobot ACT policy over a local TCP socket.")
    parser.add_argument("--policy_path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy_server = PolicyServer(args.policy_path, args.device)
    run_server(policy_server, args.host, args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[SERVER] interrupted")
