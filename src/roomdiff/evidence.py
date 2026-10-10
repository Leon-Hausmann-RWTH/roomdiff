from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from .config import Config
from .geometry import Views, backproject, project

SEEN, FREE, OCCLUDED, UNREACHED = 0, 1, 2, 3
STATE_NAMES = ("seen", "free", "occluded", "unreached")


@dataclass
class Evidence:
    seen: np.ndarray  # (M,) frames whose depth agrees with the point
    free: np.ndarray  # (M,) frames that looked through the point: evidence of absence
    occluded: np.ndarray  # (M,) frames where a nearer surface hid the point
    in_view: np.ndarray  # (M,) frames where the point falls inside the image on valid depth
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
    return bins % 63


def count_evidence(
    xyz: np.ndarray, src_depth: np.ndarray, views: Views, cfg: Config, skip: int | None = None
) -> Evidence:
    """Test every point against every frame of a video along that frame's line of sight.

    The per-pixel test is the one in SceneDiff's foreground_distance (Wu et al., ECCV 2026) and the
    ray reasoning of Fehr et al. (ICRA 2017) and Khronos (Schmid et al., RSS 2024); here the votes
    are counted over all frames and camera positions instead of averaged per pixel.
    """
    m = len(xyz)
    seen, free, occ, view = (np.zeros(m, np.int16) for _ in range(4))
    free_bits = np.zeros(m, np.int64)
    n, h, w = views.depth.shape
    bins = viewpoints(views, cfg.viewpoint_spacing)
    r = cfg.window
    size = 2 * r + 1
    for i in range(n):
        if i == skip:
            continue
        d = views.depth[i]
        invalid = d <= 0
        # Free or occluded needs the whole window to agree, which absorbs a few pixels of registration
        # error at outlines; a window touching invalid depth stays undecided.
        nearest = minimum_filter(np.where(invalid, np.inf, d), size)
        farthest = maximum_filter(np.where(invalid, -np.inf, d), size)
        holes = maximum_filter(invalid, size)
        u, v, z = project(xyz, views.K[i], views.c2w[i])
        ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
        idx = np.nonzero((z > 0.05) & (ui >= r) & (ui < w - r) & (vi >= r) & (vi < h - r))[0]
        ui, vi, zk = ui[idx], vi[idx], z[idx]
        ok = ~invalid[vi, ui]
        idx, ui, vi, zk = idx[ok], ui[ok], vi[ok], zk[ok]
        # A far-away source view is noisier than a close check, so the band follows the larger depth.
        tau = cfg.tau_abs + cfg.tau_rel * np.maximum(zk, src_depth[idx])
        decided = ~holes[vi, ui]
        is_free = decided & (nearest[vi, ui] > zk + tau)
        view[idx] += 1
        free[idx[is_free]] += 1
        free_bits[idx[is_free]] |= np.int64(1) << bins[i]
        occ[idx[decided & (farthest[vi, ui] < zk - tau)]] += 1
        seen[idx[np.abs(d[vi, ui] - zk) <= tau]] += 1
    return Evidence(seen, free, occ, view, np.bitwise_count(free_bits).astype(np.int16))


def point_states(ev: Evidence, cfg: Config) -> np.ndarray:
    decisive = ev.seen + ev.free
    # A verdict needs several decisive frames; the 60 % share follows Khronos' absence rule.
    enough = decisive >= cfg.min_decisive
    state = np.full(len(decisive), UNREACHED, np.int8)
    state[~enough & (ev.occluded >= np.maximum(1, decisive))] = OCCLUDED
    state[enough] = SEEN
    state[enough & (ev.free >= cfg.free_ratio * decisive)] = FREE
    return state


def self_consistent(xyz: np.ndarray, src_depth: np.ndarray, own: Views, cfg: Config) -> np.ndarray:
    # A point its own video looks through is a flying pixel, hallucinated geometry or something
    # that moved during the recording; none of these should count as a surface.
    ev = count_evidence(xyz, src_depth, own, cfg)
    return ev.free < np.maximum(2, ev.seen)


def frame_conflicts(views: Views, cfg: Config, stride: int = 6) -> np.ndarray:
    """Per keyframe, the share of its pixels that the rest of its own video looks through.

    A keyframe whose pose or depth is off contradicts its neighbours; left in, a run of such frames
    alone can create or erase evidence against the other video.
    """
    scores = np.zeros(len(views.depth))
    for i in range(len(views.depth)):
        d = views.depth[i][::stride, ::stride]
        ok = d > 0
        if ok.sum() < 50:
            scores[i] = 1.0
            continue
        sub = views.K[i].copy()
        sub[:2] /= stride
        xyz = backproject(d, sub, views.c2w[i])[ok]
        ev = count_evidence(xyz, d[ok], views, cfg, skip=i)
        scores[i] = float(np.mean(ev.free >= np.maximum(2, ev.seen)))
    return scores
