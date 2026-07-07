from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_light_intensity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    prim_path: str = "/World/light",
    intensity_range: tuple[float, float] = (2400.0, 3600.0),
) -> None:
    """Randomize a USD light intensity once per reset."""
    del env_ids
    prim = env.sim.stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    intensity_attr = prim.GetAttribute("inputs:intensity")
    if not intensity_attr.IsValid():
        return

    intensity = torch.empty((), device="cpu").uniform_(*intensity_range).item()
    intensity_attr.Set(float(intensity))
