import gc
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# One printed line per stage; transformers' per-model progress bars and advisories drown them.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch  # noqa: E402
from transformers.utils import logging as transformers_logging  # noqa: E402

transformers_logging.disable_progress_bar()


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
    min_cluster_points: int = 20
    """Smallest candidate kept, in 2 cm points."""
    max_depth: float = 6.0
    """Candidates seen from farther than this (metres) are outside the room: windows, mirrors."""
    viewpoint_spacing: float = 0.25
    """Cameras closer than this (metres) count as one viewpoint."""
    min_multiview: float = 0.3
    """Share of a candidate's points that cameras in at least two places must have looked through."""
    max_frame_conflict: float = 0.3
    """Keyframes whose pixels the rest of their own video looks through more often than this are dropped."""
    min_observed: float = 0.2
    """Share of an object's points the other video must decide on; below it the object is unverified."""
    confirm_ratio: float = 0.5
    """Share of the decided points that must be free to confirm a change."""
    partial_ratio: float = 0.2
    """Share of free points above which a mostly-still-there object counts as partly changed."""
    match_similarity: float = 0.5
    """Minimum DINOv2 cosine similarity for a disappeared and an appeared object to be one moved object."""
    search_similarity: float = 0.5
    """DINOv2 patch similarity needed to look for an unmatched object's other end in the other video."""
    min_fit: float = 0.3
    """Share of points that must align for the rigid fit of a move to be trusted."""
    min_move: float = 0.08
    """A matched pair that moved less than this (metres) and turned less than 15 degrees did not move."""
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


def free_device(device: torch.device) -> None:
    # Models run one after another; return each one's memory before loading the next.
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
