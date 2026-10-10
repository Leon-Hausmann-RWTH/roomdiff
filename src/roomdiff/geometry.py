from dataclasses import dataclass

import numpy as np
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


def pixel_grid(h: int, w: int) -> np.ndarray:
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([u, v], -1)  # (H, W, 2)


def backproject(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    uv = pixel_grid(h, w)
    x = (uv[..., 0] - K[0, 2]) / K[0, 0] * depth
    y = (uv[..., 1] - K[1, 2]) / K[1, 1] * depth
    cam = np.stack([x, y, depth], -1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]  # (H, W, 3)


def project(xyz: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(c2w)
    cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    safe = np.where(z > 1e-6, z, 1e-6)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return u, v, z


def depth_edges(depth: np.ndarray, rtol: float) -> np.ndarray:
    # Flying pixels sit on depth discontinuities; drop them before they become fake surfaces.
    pad = np.pad(depth, 1, mode="edge")
    h, w = depth.shape
    stack = np.stack([pad[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)])
    return (stack.max(0) - stack.min(0)) > rtol * depth


def pack_keys(cells: np.ndarray) -> np.ndarray:
    c = cells.astype(np.int64) + (1 << 20)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def cloud_from_views(views: Views, voxel: float) -> Cloud:
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
    keys = pack_keys(cells)
    uniq, inv = np.unique(keys, return_inverse=True)
    ucells = cells[np.unique(inv, return_index=True)[1]]
    rows, cols = [], []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for off in offsets[14:]:
        nb = pack_keys(ucells + np.array(off))
        j = np.searchsorted(uniq, nb).clip(0, len(uniq) - 1)
        hit = uniq[j] == nb
        rows.append(np.nonzero(hit)[0])
        cols.append(j[hit])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(uniq), len(uniq)))
    return connected_components(graph, directed=False)[1][inv]
