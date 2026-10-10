from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .config import Config
from .objects import Obj


@dataclass
class Motion:
    yaw: float  # radians about world z
    t: np.ndarray  # (3,) metres; p_after = Rz(yaw) @ p_before + t
    fit: float  # share of before points within 4 cm of an after point after alignment
    shift: np.ndarray  # (3,) how the object's centre travelled, metres
    turn: float | None  # degrees, None when the yaw is ambiguous (symmetric object) or not measured

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.shift))


def rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def solve_yaw(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    """Least-squares yaw and translation mapping p onto q (rotation about z only: objects stay upright)."""
    pc, qc = p.mean(0), q.mean(0)
    a, b = p[:, :2] - pc[:2], q[:, :2] - qc[:2]
    yaw = float(np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum()))
    return yaw, qc - rz(yaw) @ pc


def register(src: np.ndarray, dst: np.ndarray, inlier: float = 0.04, steps: int = 12) -> Motion:
    """Trimmed point-to-point ICP over yaw and translation, started from a sweep of yaw angles.

    The two ends of a move are often filmed from different sides, so the start angle matters;
    the sweep makes the result independent of it.
    """
    tree = cKDTree(dst)
    results = []
    for yaw0 in np.linspace(0, 2 * np.pi, steps, endpoint=False):
        yaw, t = yaw0, dst.mean(0) - rz(yaw0) @ src.mean(0)
        for _ in range(25):
            d, j = tree.query(src @ rz(yaw).T + t)
            keep = d < max(np.percentile(d, 70), inlier)
            yaw, t = solve_yaw(src[keep], dst[j[keep]])
        d, _ = tree.query(src @ rz(yaw).T + t)
        results.append((float(np.mean(d < inlier)), yaw % (2 * np.pi), t))
    fit = max(r[0] for r in results)
    good = [r for r in results if r[0] >= 0.9 * fit]
    # A box fits equally well turned by 180 degrees; report the smallest turn that fits. If good fits
    # spread beyond that symmetry (cylinder, square top) the turn cannot be measured.
    _, yaw, t = min(good, key=lambda r: abs(np.angle(np.exp(1j * r[1]))))
    half_turns = np.array([r[1] for r in good]) % np.pi
    spread = np.abs(np.angle(np.exp(2j * (half_turns - yaw % np.pi)))).max() / 2
    centre = src.mean(0)
    turn = float(np.degrees(np.angle(np.exp(1j * yaw))))
    return Motion(yaw, t, fit, rz(yaw) @ centre + t - centre, None if spread > np.radians(20) else turn)


def footprint_motion(src: np.ndarray, dst: np.ndarray) -> Motion:
    # Fallback when the two ends share too little surface to register: compare box centres, no turn.
    c0 = (np.percentile(src, 5, 0) + np.percentile(src, 95, 0)) / 2
    c1 = (np.percentile(dst, 5, 0) + np.percentile(dst, 95, 0)) / 2
    return Motion(0.0, c1 - c0, 0.0, c1 - c0, None)


def same_kind(a: Obj, b: Obj, xyz: tuple[np.ndarray, np.ndarray], cfg: Config) -> bool:
    """Could these be one object: similar DINOv2 features and heights within a factor of two."""
    if a.feature is None or b.feature is None or float(a.feature @ b.feature) < cfg.match_similarity:
        return False
    ha, hb = (np.ptp(xyz[o.side][o.idx][:, 2]) + 0.02 for o in (a, b))
    return max(ha / hb, hb / ha) < 2.0


def fit_move(a: Obj, b: Obj, xyz: tuple[np.ndarray, np.ndarray], cfg: Config) -> Motion:
    src, dst = xyz[0][a.idx], xyz[1][b.idx]
    m = register(src, dst)
    return m if m.fit >= cfg.min_fit else footprint_motion(src, dst)


def match(before: list[Obj], after: list[Obj], xyz: tuple[np.ndarray, np.ndarray], cfg: Config) -> list:
    """Pair disappeared with appeared objects: same appearance, compatible size, then a rigid fit.

    A move is a removal and an addition of the same object (as in SceneDiff, REACT and 3DGS-CD).
    Returns (before object, after object, Motion) triples.
    """
    if not before or not after:
        return []
    cost = np.full((len(before), len(after)), 9.0)
    for i, a in enumerate(before):
        for j, b in enumerate(after):
            if same_kind(a, b, xyz, cfg):
                cost[i, j] = 1 - float(a.feature @ b.feature)
    pairs = zip(*linear_sum_assignment(cost))
    return [
        (before[i], after[j], fit_move(before[i], after[j], xyz, cfg)) for i, j in pairs if cost[i, j] < 9
    ]
