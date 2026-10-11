from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .evidence import FREE, Evidence
from .geometry import Cloud, components
from .moves import Motion
from .objects import Obj


@dataclass
class Box:
    center: np.ndarray  # (3,) metres
    size: np.ndarray  # (3,) extent along the box's own x, y and world z, metres
    yaw: float  # radians, rotation of the box x axis about world z


@dataclass
class Change:
    id: int
    kind: str  # removed, added or moved
    status: str  # confirmed or unverified
    reason: str
    before: Obj | None = None  # the object as the before video saw it
    after: Obj | None = None
    lead: Obj | None = None  # the end whose evidence the reason and the shares describe
    motion: Motion | None = None
    label: str = "object"
    sentence: str = ""
    facts: dict = field(default_factory=dict)


def fit_box(xyz: np.ndarray) -> Box:
    """Gravity-aligned box, yaw along the footprint's principal axis, extent from 2nd to 98th percentiles."""
    lo, hi = np.percentile(xyz, 2, 0), np.percentile(xyz, 98, 0)
    mean_xy = xyz[:, :2].mean(0)
    xy = xyz[:, :2] - mean_xy
    _, evecs = np.linalg.eigh(np.cov(xy.T) if len(xy) > 2 else np.eye(2))
    yaw = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    local = xy @ rot
    llo, lhi = np.percentile(local, 2, 0), np.percentile(local, 98, 0)
    center_xy = mean_xy + rot @ ((llo + lhi) / 2)
    size = np.array([*(lhi - llo), hi[2] - lo[2]]).clip(0.02)
    return Box(np.array([*center_xy, (lo[2] + hi[2]) / 2]), size, yaw)


def candidates(cloud: Cloud, state: np.ndarray, ev: Evidence, cfg: Config) -> list[np.ndarray]:
    """Groups of connected free points: places where the other video looked through a surface."""
    free = np.nonzero(state == FREE)[0]
    if len(free) == 0:
        return []
    labels = components(cloud.xyz[free], cfg.cluster_voxel)
    out = []
    for lab in np.unique(labels):
        idx = free[labels == lab]
        if len(idx) < cfg.min_cluster_points:
            continue
        if np.mean(ev.free_views[idx] >= 2) < cfg.min_multiview:
            continue
        # Below the floor, or only seen from far away.
        if np.percentile(cloud.xyz[idx, 2], 90) < -0.05 or np.median(cloud.depth[idx]) > cfg.max_depth:
            continue
        out.append(idx)
    return out


def assemble(objs: list[Obj], pairs, cfg: Config) -> list[Change]:
    """Turn verdicts and matches into the reported changes."""
    changes, paired = [], set()
    for a, b, m in pairs:
        paired |= {id(a), id(b)}
        if m.distance < cfg.min_move and (m.turn is None or abs(m.turn) < cfg.min_turn):
            continue  # the same object in the same place
        lead = next(
            (o for o in (a, b) if o.status == "confirmed"), max((a, b), key=lambda o: o.shares["free"])
        )
        status, reason = "confirmed" if lead.status == "confirmed" else "unverified", lead.reason
        # Two partly free ends that align rigidly confirm the move.
        if a.status == b.status == "partial" and m.fit >= cfg.min_fit:
            status = "confirmed"
            reason = f"both ends are partly free and {100 * m.fit:.0f} % of its points align after the move"
        changes.append(Change(0, "moved", status, reason, a, b, lead, m))
    for o in objs:
        if o.status == "dismissed" or id(o) in paired:
            continue
        status = "confirmed" if o.status == "confirmed" else "unverified"
        ends = (o, None) if o.side == 0 else (None, o)
        changes.append(Change(0, "removed" if o.side == 0 else "added", status, o.reason, *ends, o))
    order = {"moved": 0, "removed": 1, "added": 2}
    changes.sort(key=lambda c: (c.status != "confirmed", order[c.kind], -c.lead.shares["points"]))
    for k, c in enumerate(changes):
        c.id = k + 1
    return changes
