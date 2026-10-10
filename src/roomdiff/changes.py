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
    motion: Motion | None = None
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
        # Keyframes close together see almost the same ray, so one burst of bad depth or pose looks like
        # many votes; on the tuning pairs most false candidates came from a single camera position.
        if np.mean(ev.free_views[idx] >= 2) < cfg.min_multiview:
            continue
        if np.percentile(cloud.xyz[idx, 2], 90) < -0.05 or np.median(cloud.depth[idx]) > cfg.max_depth:
            continue
        out.append(idx)
    return out


def assemble(objs: list[Obj], pairs, cfg: Config) -> tuple[list[Change], list[Obj]]:
    """Turn verdicts and matches into reported changes; returns them and the objects dismissed on the way."""
    changes, dismissed, paired = [], [o for o in objs if o.status == "dismissed"], set()
    for a, b, m in pairs:
        paired |= {id(a), id(b)}
        if m.distance < cfg.min_move and (m.turn is None or abs(m.turn) < 15):
            # The same object in the same place: what looked like change was registration noise.
            a.reason = b.reason = f"matched itself in the other video, {m.distance * 100:.0f} cm away"
            dismissed += [a, b]
            continue
        # Where an object moved less than its own size, both ends overlap and each is only partly free;
        # two partial ends plus a rigid fit are as good as one confirmed end.
        confirmed = "confirmed" in (a.status, b.status) or (
            a.status == b.status == "partial" and m.fit >= cfg.min_fit
        )
        reason = a.reason if a.status == "confirmed" else b.reason
        changes.append(Change(0, "moved", "confirmed" if confirmed else "unverified", reason, a, b, m))
    for o in objs:
        if o.status == "dismissed" or id(o) in paired:
            continue
        status = "confirmed" if o.status == "confirmed" else "unverified"
        changes.append(
            Change(
                0,
                "removed" if o.side == 0 else "added",
                status,
                o.reason,
                *((o, None) if o.side == 0 else (None, o)),
            )
        )
    order = {"moved": 0, "removed": 1, "added": 2}
    size = lambda c: sum(len(o.idx) for o in (c.before, c.after) if o is not None)  # noqa: E731
    changes.sort(key=lambda c: (c.status != "confirmed", order[c.kind], -size(c)))
    for k, c in enumerate(changes):
        c.id = k + 1
    return changes, dismissed
