from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .evidence import FREE
from .geometry import components


@dataclass
class Box:
    center: np.ndarray  # (3,) metres
    size: np.ndarray  # (3,) extent along the box's own x, y and world z, metres
    yaw: float  # radians, rotation of the box x axis about world z

    def corners(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        signs = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)])
        return self.center + (signs * self.size / 2) @ R.T  # (8, 3)


@dataclass
class Change:
    id: int
    kind: str  # removed, added or moved
    status: str  # confirmed, partial, unverified or dismissed
    reason: str
    before: np.ndarray | None = None  # point indices in the before cloud
    after: np.ndarray | None = None  # point indices in the after cloud
    label: str = "object"
    sentence: str = ""
    facts: dict = field(default_factory=dict)


def fit_box(xyz: np.ndarray) -> Box:
    """Gravity-aligned box: yaw from the principal axis of the footprint, robust to a few stray points."""
    lo, hi = np.percentile(xyz, 2, 0), np.percentile(xyz, 98, 0)
    xy = xyz[:, :2] - xyz[:, :2].mean(0)
    evals, evecs = np.linalg.eigh(np.cov(xy.T) if len(xy) > 2 else np.eye(2))
    yaw = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
    c, s = np.cos(yaw), np.sin(yaw)
    local = xy @ np.array([[c, -s], [s, c]])
    llo, lhi = np.percentile(local, 2, 0), np.percentile(local, 98, 0)
    mid = (llo + lhi) / 2
    center_xy = xyz[:, :2].mean(0) + np.array([[c, -s], [s, c]]) @ mid
    size = np.array([*(lhi - llo), hi[2] - lo[2]]).clip(0.02)
    return Box(np.array([*center_xy, (lo[2] + hi[2]) / 2]), size, yaw)


def candidates(xyz: np.ndarray, state: np.ndarray, cfg: Config) -> list[np.ndarray]:
    """Groups of connected free points: places where the other video looked through a surface."""
    free = np.nonzero(state == FREE)[0]
    if len(free) == 0:
        return []
    labels = components(xyz[free], cfg.cluster_voxel)
    out = []
    for lab in np.unique(labels):
        idx = free[labels == lab]
        if len(idx) < cfg.min_cluster_points:
            continue
        pts = xyz[idx]
        # Misregistration leaves thin sheets on floors and walls; real objects have some thickness.
        thickness = np.sqrt(np.linalg.eigvalsh(np.cov(pts.T))[0]) * 4
        if thickness < cfg.min_thickness or np.percentile(pts[:, 2], 90) < -0.05:
            continue
        out.append(idx)
    return out
