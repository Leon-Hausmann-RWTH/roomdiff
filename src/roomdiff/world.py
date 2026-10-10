from dataclasses import dataclass, replace

import numpy as np

from .geometry import Views, backproject


@dataclass
class Frame:
    T: np.ndarray  # (4, 4) world from reconstruction frame; z up, floor at z = 0
    floor_found: bool  # a level plane with the cameras a plausible 0.9 to 2.2 m above it
    camera_height: float  # median camera height above the floor, metres
    up_from_cameras: bool  # False when the cameras barely turned and gravity came from surface normals


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def points_and_normals(views: Views, stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    pts, nrm = [], []
    for i in range(len(views.depth)):
        p = backproject(views.depth[i], views.K[i], views.c2w[i])
        d = views.depth[i]
        dx, dy = p[:-1, 1:] - p[:-1, :-1], p[1:, :-1] - p[:-1, :-1]
        n = np.cross(dx, dy)
        ok = (d[:-1, :-1] > 0) & (d[:-1, 1:] > 0) & (d[1:, :-1] > 0)
        sparse = np.zeros_like(ok)
        sparse[::stride, ::stride] = True
        ok &= sparse
        n = n[ok] / (np.linalg.norm(n[ok], axis=1, keepdims=True) + 1e-12)
        pts.append(p[:-1, :-1][ok])
        nrm.append(n)
    return np.concatenate(pts), np.concatenate(nrm)


def up_from_normals(c2w: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """Up when the cameras barely turned: pitch is a rotation about the mean camera x axis, so search
    that plane for the direction most surface normals share (floors and tables face up)."""
    x = c2w[:, :3, 0].mean(0)
    x /= np.linalg.norm(x)
    u0 = -c2w[:, :3, 1].mean(0)
    u0 -= (u0 @ x) * x
    u0 /= np.linalg.norm(u0)
    e2 = np.cross(x, u0)
    n = nrm[np.abs(nrm @ x) < 0.2]
    n = n * np.sign(n @ u0)[:, None]
    phi = np.arctan2(n @ e2, n @ u0)
    hist, edges = np.histogram(phi[np.abs(phi) < np.radians(60)], bins=60)
    best = (edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2
    return np.cos(best) * u0 + np.sin(best) * e2


def world_frame(views_a: Views, views_b: Views) -> Frame:
    """Gravity from the camera axes, refined by the floor plane; yaw from the dominant wall direction."""
    c2w = np.concatenate([views_a.c2w, views_b.c2w])
    # Handheld phones keep roll near zero, so every camera x axis is horizontal: up is the direction
    # orthogonal to all of them. Unlike the mean camera y axis, this does not tilt with pitch.
    pts, nrm = (np.concatenate(x) for x in zip(points_and_normals(views_a), points_and_normals(views_b)))
    _, sv, vt = np.linalg.svd(c2w[:, :3, 0], full_matrices=False)
    turned = bool(sv[1] >= 0.15 * sv[0])
    up = vt[2] if turned else up_from_normals(c2w, nrm)
    up = up if up @ c2w[:, :3, 1].mean(0) < 0 else -up
    cams = c2w[:, :3, 3]
    floor_found = False
    floor_z = 0.0
    for _ in range(2):
        R = rotation_between(up, np.array([0.0, 0, 1]))
        z, nz, cz = pts @ R[2], nrm @ R[2], cams @ R[2]
        level = np.abs(nz) > 0.9
        # Handheld phones are 1 to 2 m above the floor; level surfaces far lower are reflections or the
        # view through a window.
        below = level & (z < np.median(cz) - 0.3) & (z > np.median(cz) - 2.5)
        if below.sum() < 500:
            break
        bins = np.arange(z[below].min(), z[below].max() + 0.02, 0.02)
        hist, edges = np.histogram(z[below], bins)
        significant = np.nonzero((hist >= 0.15 * hist.max()) & (hist >= 0.005 * below.sum()))[0]
        # The floor is the lowest large horizontal plane; tables and beds sit above it.
        h0 = edges[significant[0]] + 0.01
        near = below & (np.abs(z - h0) < 0.05)
        centred = pts[near] - pts[near].mean(0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][2]
        up = normal if normal @ up > 0 else -normal
        floor_z = float(pts[near].mean(0) @ (up / np.linalg.norm(up)))
        floor_found = True
    R = rotation_between(up, np.array([0.0, 0, 1]))
    if not floor_found:
        floor_z = float(np.percentile(pts @ R[2], 1))
    walls = np.abs(nrm @ R[2]) < 0.3
    nw = nrm[walls] @ R.T
    theta = np.angle(np.exp(4j * np.arctan2(nw[:, 1], nw[:, 0])).sum()) / 4
    Rz = np.array([[np.cos(-theta), -np.sin(-theta), 0], [np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    R = Rz @ R
    T = np.eye(4)
    T[:3, :3] = R
    xy = np.median(pts @ R.T, 0)
    T[:3, 3] = [-xy[0], -xy[1], -floor_z]
    cam_h = float(np.median(cams @ R[2] - floor_z))
    # When a bed or desk fills the view its top can pass for the floor; a phone is held 1 to 2 m up.
    floor_found = floor_found and 0.9 < cam_h < 2.2
    return Frame(T=T.astype(np.float32), floor_found=floor_found, camera_height=cam_h, up_from_cameras=turned)


def transform(views: Views, T: np.ndarray) -> Views:
    return replace(views, c2w=(T @ views.c2w).astype(np.float32))
