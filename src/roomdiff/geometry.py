from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Views:
    images: np.ndarray  # (N, H, W, 3) uint8
    depth: np.ndarray  # (N, H, W) float32 metres along the optical axis, 0 where invalid
    K: np.ndarray  # (N, 3, 3) pixel intrinsics
    c2w: np.ndarray  # (N, 4, 4) world from camera, OpenCV axes (x right, y down, z forward)
    frame_ids: np.ndarray  # (N,) frame index in the source video


@dataclass
class Cloud:
    xyz: np.ndarray  # (M, 3) float32 world metres
    rgb: np.ndarray  # (M, 3) uint8
    depth: np.ndarray  # (M,) float32 mean depth at which the own video saw the point, metres


def subset(cloud: Cloud, keep: np.ndarray) -> Cloud:
    return Cloud(*(getattr(cloud, f)[keep] for f in cloud.__dataclass_fields__))


def rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def backproject(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """(H, W, 3) world points of a depth map."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    cam = np.stack([x, y, depth], -1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def project(xyz: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(c2w)
    cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    safe = np.where(z > 1e-6, z, 1e-6)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return u, v, z


def depth_edges(depth: np.ndarray, rtol: float) -> np.ndarray:
    """Mask of flying pixels: the 3x3 depth range exceeds rtol times the depth."""
    return maximum_filter(depth, 3, mode="nearest") - minimum_filter(depth, 3, mode="nearest") > rtol * depth


def pack_keys(cells: np.ndarray) -> np.ndarray:
    """One int64 key per integer cell, 21 bits per axis."""
    c = cells.astype(np.int64) + (1 << 20)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def cloud_from_views(views: Views, voxel: float) -> Cloud:
    """All valid pixels of all frames, averaged per voxel."""
    xyz, rgb, dep = [], [], []
    for i in range(len(views.depth)):
        ok = views.depth[i] > 0
        xyz.append(backproject(views.depth[i], views.K[i], views.c2w[i])[ok])
        rgb.append(views.images[i][ok])
        dep.append(views.depth[i][ok])
    xyz, rgb, dep = (np.concatenate(a) for a in (xyz, rgb, dep))
    keys = pack_keys(np.floor(xyz / voxel))
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    m = len(uniq)

    def mean(a):
        return np.stack([np.bincount(inv, a[:, k], m) for k in range(a.shape[1])], 1) / counts[:, None]

    return Cloud(
        xyz=mean(xyz).astype(np.float32),
        rgb=mean(rgb.astype(np.float64)).round().astype(np.uint8),
        depth=(np.bincount(inv, dep, m) / counts).astype(np.float32),
    )


def components(xyz: np.ndarray, cell: float) -> np.ndarray:
    """Label 26-connected groups of occupied cells; returns (M,) component id per point."""
    cells = np.floor(xyz / cell).astype(np.int64)
    uniq, first, inv = np.unique(pack_keys(cells), return_index=True, return_inverse=True)
    ucells = cells[first]
    rows, cols = [], []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    # The graph is undirected, so the 13 offsets after (0, 0, 0) cover all 26 neighbours.
    for off in offsets[14:]:
        nb = pack_keys(ucells + np.array(off))
        j = np.searchsorted(uniq, nb).clip(0, len(uniq) - 1)
        hit = uniq[j] == nb
        rows.append(np.nonzero(hit)[0])
        cols.append(j[hit])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(uniq), len(uniq)))
    return connected_components(graph, directed=False)[1][inv]
