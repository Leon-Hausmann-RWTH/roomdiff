import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


@dataclass(frozen=True)
class Config:
    frames_per_video: int = 28
    """Keyframes per video for the joint reconstruction."""
    pixels_per_frame: int = 141_120
    """Pixel budget per keyframe (504 x 280 for 16:9)."""
    conf_threshold: float = 0.1
    """Minimum sigmoid confidence of a reconstructed pixel."""
    edge_rtol: float = 0.04
    """Pixels whose 3x3 depth range exceeds this share of their depth are dropped as flying pixels."""
    voxel: float = 0.02
    """Point cloud voxel size in metres."""
    tau_abs: float = 0.04
    """Fixed part of the depth agreement band in metres."""
    tau_rel: float = 0.03
    """Depth-proportional part of the agreement band."""
    window: int = 2
    """Half-width in pixels of the window that must agree before a frame calls a point free or occluded."""
    min_decisive: int = 3
    """Frames that must see or see through a point before it gets a verdict."""
    free_ratio: float = 0.6
    """Share of decisive frames that must look through a point to call it free."""
    cluster_voxel: float = 0.04
    """Voxel size for grouping free points into candidates, metres."""
    min_cluster_points: int = 30
    """Smallest candidate kept, in 2 cm points."""
    min_thickness: float = 0.03
    """Candidates thinner than this (metres, along their flattest axis) are misregistration sheets."""
    device: str = "auto"
    """cuda, mps, cpu or auto (cuda, then mps, then cpu)."""
    cache_dir: Path = Path("cache")
    """Where reconstructions are cached so later stages rerun without the model."""


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
