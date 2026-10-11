from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from .config import Config
from .geometry import Views, project

SEEN, FREE, OCCLUDED, UNREACHED = 0, 1, 2, 3


@dataclass
class Evidence:
    seen: np.ndarray  # (M,) frames whose depth agrees with the point
    free: np.ndarray  # (M,) frames that looked through the point: evidence of absence
    occluded: np.ndarray  # (M,) frames where a nearer surface hid the point
    free_views: np.ndarray  # (M,) distinct camera positions among the frames that looked through it


def viewpoints(views: Views, spacing: float) -> np.ndarray:
    """Group keyframes whose cameras stand within `spacing` metres; returns a bin id per frame."""
    centres = views.c2w[:, :3, 3]
    bins, anchors = np.zeros(len(centres), np.int64), []
    for i, c in enumerate(centres):
        d = [np.linalg.norm(c - a) for a in anchors]
        if d and min(d) < spacing:
            bins[i] = int(np.argmin(d))
        else:
            bins[i] = len(anchors)
            anchors.append(c)
    return bins % 63  # each bin is one bit of an int64 mask


def count_evidence(xyz: np.ndarray, src_depth: np.ndarray, views: Views, cfg: Config) -> Evidence:
    """Count per point the frames that see it, look through it or hide it behind a nearer surface.
    Ray test as in SceneDiff's foreground_distance, Fehr et al. and Khronos."""
    m = len(xyz)
    seen, free, occ = (np.zeros(m, np.int16) for _ in range(3))
    free_bits = np.zeros(m, np.int64)
    n, h, w = views.depth.shape
    bins = viewpoints(views, cfg.viewpoint_spacing)
    r = cfg.window
    size = 2 * r + 1
    for i in range(n):
        d = views.depth[i]
        invalid = d <= 0
        # Free or occluded only if the whole window agrees; a window touching invalid depth stays undecided.
        nearest = minimum_filter(np.where(invalid, np.inf, d), size)
        farthest = maximum_filter(np.where(invalid, -np.inf, d), size)
        holes = maximum_filter(invalid, size)
        u, v, z = project(xyz, views.K[i], views.c2w[i])
        ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
        idx = np.nonzero((z > 0.05) & (ui >= r) & (ui < w - r) & (vi >= r) & (vi < h - r))[0]
        ui, vi, zk = ui[idx], vi[idx], z[idx]
        ok = ~invalid[vi, ui]
        idx, ui, vi, zk = idx[ok], ui[ok], vi[ok], zk[ok]
        tau = cfg.tau_abs + cfg.tau_rel * np.maximum(zk, src_depth[idx])  # depth tolerance, metres
        decided = ~holes[vi, ui]
        is_free = decided & (nearest[vi, ui] > zk + tau)
        free[idx[is_free]] += 1
        free_bits[idx[is_free]] |= np.int64(1) << bins[i]
        occ[idx[decided & (farthest[vi, ui] < zk - tau)]] += 1
        seen[idx[np.abs(d[vi, ui] - zk) <= tau]] += 1
    return Evidence(seen, free, occ, np.bitwise_count(free_bits).astype(np.int16))


def point_states(ev: Evidence, cfg: Config) -> np.ndarray:
    """One state per point from its votes; the free-share rule follows Khronos."""
    decisive = ev.seen + ev.free
    enough = decisive >= cfg.min_decisive
    state = np.full(len(decisive), UNREACHED, np.int8)
    # Hidden only without a verdict: blocked at least once and at least as often as decided.
    state[~enough & (ev.occluded >= np.maximum(1, decisive))] = OCCLUDED
    state[enough] = SEEN
    state[enough & (ev.free >= cfg.free_ratio * decisive)] = FREE
    return state


def self_consistent(xyz: np.ndarray, src_depth: np.ndarray, own: Views, cfg: Config) -> np.ndarray:
    """False for points their own video looks through: flying pixels or things that moved while filming."""
    ev = count_evidence(xyz, src_depth, own, cfg)
    return ev.free < np.maximum(2, ev.seen)
