import numpy as np
import pytest

from roomdiff.changes import candidates
from roomdiff.config import Config
from roomdiff.evidence import (
    FREE,
    OCCLUDED,
    SEEN,
    UNREACHED,
    Evidence,
    count_evidence,
    point_states,
    self_consistent,
)
from roomdiff.geometry import Cloud, Views, cloud_from_views, components

# Axis-aligned boxes (min corner, max corner), metres, z up.
STRUCTURE = {
    "floor": ((-3, -3, -0.05), (3, 3, 0)),
    "back_wall": ((-3, 3, 0), (3, 3.05, 2.5)),
    "east_wall": ((3, -3, 0), (3.05, 3, 2.5)),
}
STATIC = {
    "S": ((-0.6, 2.4, 0), (-0.2, 2.8, 0.5)),  # seen by both videos
    "O": ((1.0, 1.3, 0), (1.3, 2.0, 1.2)),  # tall panel that hides H from the after video
    "H": ((1.5, 1.5, 0), (1.8, 1.8, 0.3)),  # never visible after: must never come out free
    "U": ((-2.5, 0.0, 0), (-2.2, 0.3, 0.3)),  # outside every after view
}
REMOVED = {"R": ((0.0, 1.0, 0), (0.4, 1.4, 0.4))}
ADDED = {"N": ((0.5, 2.2, 0), (0.9, 2.6, 0.5))}

H, W, F = 120, 160, 100.0


def look_at(pos, target):
    fwd = np.subtract(target, pos, dtype=float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, np.cross(fwd, right), fwd], 1)
    c2w[:3, 3] = pos
    return c2w


def render(boxes, c2w):
    """Depth along the optical axis and the name of the first box hit, per pixel."""
    v, u = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u, float)], -1)
    d = d_cam @ c2w[:3, :3].T
    o = c2w[:3, 3]
    best = np.full((H, W), np.inf)
    name = np.full((H, W), "", object)
    with np.errstate(divide="ignore", invalid="ignore"):
        for key, (lo, hi) in boxes.items():
            t1, t2 = (np.array(lo) - o) / d, (np.array(hi) - o) / d
            near = np.nanmax(np.minimum(t1, t2), -1)
            far = np.nanmin(np.maximum(t1, t2), -1)
            hit = (near <= far) & (near > 0) & (near < best)
            best[hit] = near[hit]
            name[hit] = key
    return np.where(np.isfinite(best), best, 0).astype(np.float32), name


def make_views(boxes, poses):
    rendered = [render(boxes, p) for p in poses]
    n = len(poses)
    K = np.tile(np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], np.float32), (n, 1, 1))
    views = Views(
        images=np.full((n, H, W, 3), 128, np.uint8),
        depth=np.stack([r[0] for r in rendered]),
        K=K,
        c2w=np.stack(poses).astype(np.float32),
        frame_ids=np.arange(n),
    )
    return views, np.stack([r[1] for r in rendered])


def owner(xyz, boxes, pad=0.015):
    """Name of the box each point lies in, "" for none."""
    out = np.full(len(xyz), "", object)
    for key, (lo, hi) in boxes.items():
        inside = np.all((xyz >= np.array(lo) - pad) & (xyz <= np.array(hi) + pad), 1)
        out[inside & (out == "")] = key
    return out


@pytest.fixture(scope="module")
def scene():
    before = {**STRUCTURE, **STATIC, **REMOVED}
    after = {**STRUCTURE, **STATIC, **ADDED}
    poses_a = [look_at((x, -2.5, 1.4), (0.3, 1.5, 0.3)) for x in np.linspace(-1.5, 1.5, 6)]
    poses_b = [look_at((-2.0, y, 0.8), (2.0, y + 0.2, 0.3)) for y in np.linspace(0.8, 2.2, 6)]
    views_a, names_a = make_views(before, poses_a)
    views_b, names_b = make_views(after, poses_b)
    return before, after, views_a, names_a, views_b, names_b


def states_by_owner(cloud, boxes, other, cfg):
    st = point_states(count_evidence(cloud.xyz, cloud.depth, other, cfg), cfg)
    return st, owner(cloud.xyz, boxes)


def test_removed_box_is_free_and_hidden_boxes_never_are(scene):
    before, _, views_a, _, views_b, _ = scene
    cfg = Config()
    cloud = cloud_from_views(views_a, cfg.voxel)
    st, own = states_by_owner(cloud, before, views_b, cfg)
    assert np.mean(st[own == "R"] == FREE) > 0.6
    for key in ("S", "O", "H", "U"):
        assert not np.any(st[own == key] == FREE), key
    assert np.mean(st[own == "H"] == OCCLUDED) > 0.5
    assert np.all(st[own == "U"] == UNREACHED)
    assert np.mean(st[own == "S"] == SEEN) > 0.5
    structure = np.isin(own, list(STRUCTURE))
    assert np.mean(st[structure] == FREE) < 0.01


def test_added_box_is_free_against_the_before_video(scene):
    _, after, views_a, _, views_b, _ = scene
    cfg = Config()
    cloud = cloud_from_views(views_b, cfg.voxel)
    st, own = states_by_owner(cloud, after, views_a, cfg)
    assert np.mean(st[own == "N"] == FREE) > 0.6
    for key in ("S", "O"):
        assert not np.any(st[own == key] == FREE), key


def test_static_points_pass_their_own_video(scene):
    _, _, views_a, _, _, _ = scene
    cfg = Config()
    cloud = cloud_from_views(views_a, cfg.voxel)
    assert np.mean(self_consistent(cloud.xyz, cloud.depth, views_a, cfg)) > 0.98


def test_floating_point_is_dropped_by_its_own_video(scene):
    _, _, views_a, _, _, _ = scene
    cfg = Config()
    ghost = np.array([[0.3, 0.0, 1.0]], np.float32)  # in mid-air, every A camera looks through it
    assert not self_consistent(ghost, np.array([3.0], np.float32), views_a, cfg)[0]


def test_components_split_separate_blobs():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 0.2, (200, 3))
    b = rng.uniform(0, 0.2, (200, 3)) + [1.0, 0, 0]
    labels = components(np.concatenate([a, b]), 0.04)
    assert len(np.unique(labels[:200])) == 1
    assert len(np.unique(labels[200:])) == 1
    assert labels[0] != labels[200]


@pytest.mark.parametrize(
    "seen, free, occluded, expected",
    [
        (5, 0, 0, SEEN),
        (1, 4, 0, FREE),
        (3, 2, 0, SEEN),  # below the 60 % share
        (2, 0, 1, UNREACHED),  # glimpsed twice and hidden once: no verdict, and not "hidden"
        (0, 0, 3, OCCLUDED),
        (0, 0, 0, UNREACHED),
    ],
)
def test_point_states(seen, free, occluded, expected):
    ev = Evidence(*(np.array([x], np.int16) for x in (seen, free, occluded, 1)))
    assert point_states(ev, Config())[0] == expected


def test_candidates_need_more_than_one_camera_position():
    g = np.stack(np.meshgrid(np.arange(0, 0.5, 0.02), np.arange(0, 0.5, 0.02)), -1).reshape(-1, 2)
    one = np.c_[g, np.full(len(g), 0.5)]  # looked through from a single spot: one burst of bad depth
    many = np.c_[g[:, 0] + 2, np.full(len(g), 1.0), g[:, 1] + 0.3]  # front face of a removed board
    xyz = np.concatenate([one, many]).astype(np.float32)
    n = len(xyz)
    cloud = Cloud(xyz, np.zeros((n, 3), np.uint8), np.full(n, 2.0, np.float32))
    views = np.r_[np.ones(len(g)), np.full(len(g), 3)].astype(np.int16)
    z = np.zeros(n, np.int16)
    found = candidates(cloud, np.full(n, FREE, np.int8), Evidence(z, z + 4, z, views), Config())
    assert len(found) == 1 and found[0].min() >= len(g)
