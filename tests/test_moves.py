import numpy as np

from roomdiff.moves import register, rz


def box_surface(size, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-0.5, 0.5, (n, 3)) * size
    axis = rng.integers(0, 3, n)
    p[np.arange(n), axis] = np.sign(p[np.arange(n), axis]) * size[axis] / 2
    return p + [0, 0, size[2] / 2]


def test_recovers_turn_and_shift():
    src = box_surface(np.array([0.6, 0.4, 0.8]))
    dst = src @ rz(np.radians(40)).T + [0.8, -0.3, 0]
    m = register(src, dst)
    assert m.fit > 0.9
    assert abs(m.turn - 40) < 3
    assert abs(m.distance - np.hypot(0.8, 0.3)) < 0.03


def test_partial_views_of_the_two_ends():
    # The before video saw only the front half, the after video the whole object, moved by half its width.
    full = box_surface(np.array([0.6, 0.4, 0.8]))
    src = full[full[:, 1] < 0]
    dst = full @ rz(np.radians(-25)).T + [0.3, 0, 0]
    m = register(src, dst)
    # Half a box slides onto the whole box at several angles; a wrong turn must never be reported.
    assert m.turn is None or abs(m.turn + 25) < 4
    assert 0.15 < m.distance < 0.6


def test_symmetric_object_has_no_turn():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 2 * np.pi, 3000)
    src = np.stack([0.2 * np.cos(a), 0.2 * np.sin(a), rng.uniform(0, 0.5, 3000)], 1)
    m = register(src, src + [0.5, 0, 0])
    assert m.turn is None
    assert abs(m.distance - 0.5) < 0.03
