import numpy as np
from scipy.spatial import cKDTree

from .config import Config, free_device
from .evidence import FREE, SEEN
from .geometry import Cloud, Views, backproject
from .moves import Motion, fit_move, same_kind
from .objects import Obj, body, dino, embed, hires, patch_mask, sam_masks, tokens, visible

PATCHES = 64  # DINOv2 patches along the long image side for the dense search


def dense_features(images: tuple[np.ndarray, np.ndarray], device) -> list[np.ndarray]:
    """Per video, (N, rows, cols, 768) unit DINOv2 patch features of whole keyframes."""
    model = dino(device)
    out = []
    for frames in images:
        h, w = frames.shape[1:3]
        rows, cols = (round(PATCHES * h / w), PATCHES) if w >= h else (PATCHES, round(PATCHES * w / h))
        out.append(
            np.stack([tokens(model, im, (cols * 14, rows * 14)).half().cpu().numpy() for im in frames])
        )
    del model
    free_device(device)
    return out


def patch_points(views: Views, rows: int, cols: int) -> np.ndarray:
    """World point under each patch centre, NaN where the depth is invalid: (N, rows, cols, 3)."""
    h, w = views.depth.shape[1:]
    v = ((np.arange(rows) + 0.5) * h / rows).astype(int)
    u = ((np.arange(cols) + 0.5) * w / cols).astype(int)
    out = np.full((len(views.depth), rows, cols, 3), np.nan, np.float32)
    for i in range(len(views.depth)):
        world = backproject(views.depth[i], views.K[i], views.c2w[i])[np.ix_(v, u)]
        ok = views.depth[i][np.ix_(v, u)] > 0
        out[i][ok] = world[ok]
    return out


def descriptor(o: Obj, feats: np.ndarray) -> np.ndarray | None:
    """Unit mean feature of the patches over 40 % inside the object's masks, or None if there are none."""
    rows, cols = feats.shape[1:3]
    parts = [
        feats[i][sel].astype(np.float32).mean(0)
        for i, m in o.masks.items()
        if (sel := patch_mask(m, cols, rows, 0.4)).any()
    ]
    if not parts:
        return None
    f = np.mean(parts, 0)
    return f / np.linalg.norm(f)


def find_other_ends(
    lonely: list[Obj], clouds: tuple[Cloud, Cloud], states, views, images, taken, cfg: Config, device
) -> list[tuple[Obj, Obj, Motion]]:
    """Other end of each unpaired confirmed object: its best appearance match in the other video, kept
    only where the first video saw that place empty."""
    if not lonely:
        return []
    feats = dense_features(images, device)
    pts = [patch_points(views[s], *feats[s].shape[1:3]) for s in (0, 1)]
    prompts, owners = {}, {}
    for o in lonely:
        t, d = 1 - o.side, descriptor(o, feats[o.side])
        if d is None:
            continue
        # Skip the object's own place and points that belong to other changes.
        owned = np.full(pts[t].shape[:3], np.inf)
        if taken[t].any():
            owned = cKDTree(clouds[t].xyz[taken[t]]).query(
                np.nan_to_num(pts[t], nan=1e6), distance_upper_bound=0.05
            )[0]
        far = np.linalg.norm((pts[t] - clouds[o.side].xyz[o.idx].mean(0))[..., :2], axis=-1) > 0.4
        sim = np.where(far & ~np.isnan(pts[t][..., 0]) & (owned > 0.05), feats[t].astype(np.float32) @ d, -1)
        i, r, c = np.unravel_index(np.argmax(sim), sim.shape)
        # At least two frames must show a similar patch at that place.
        near = np.linalg.norm(pts[t] - pts[t][i, r, c], axis=-1) < 0.15
        if (
            sim[i, r, c] < cfg.search_similarity
            or len(np.unique(np.nonzero((sim >= 0.9 * sim[i, r, c]) & near)[0])) < 2
        ):
            continue
        hh, ww = images[t].shape[1:3]
        px = np.array([(c + 0.5) * ww / sim.shape[2], (r + 0.5) * hh / sim.shape[1]])
        # Prompt with a box the size the object would have at that depth (keyframes are at 2x focal).
        z = (np.linalg.inv(views[t].c2w[i])[:3] @ np.append(pts[t][i, r, c], 1))[2]
        half = 0.35 * np.ptp(clouds[o.side].xyz[o.idx], 0).max() * 2 * views[t].K[i, 0, 0] / max(z, 0.3)
        prompts.setdefault((t, int(i)), []).append(np.array([px - half, px + half, px]))
        owners.setdefault((t, int(i)), []).append((o, px))
    candidates = []
    for (t, i), results in sam_masks(prompts, images, device).items():
        for (o, px), (_, masks, _, _) in zip(owners[(t, i)], results):
            # The smallest of SAM's three masks that holds the matched point.
            hits = [k for k in masks if k[int(px[1]), int(px[0])]]
            if not hits:
                continue
            mask = min(hits, key=lambda k: k.sum())
            u, v, ok = visible(clouds[t].xyz, views[t], i)
            q = np.round(hires(u, v)).astype(int).clip([0, 0], [mask.shape[1] - 1, mask.shape[0] - 1])
            inside = np.nonzero(ok & mask[q[:, 1], q[:, 0]] & ~taken[t])[0]
            if len(inside) < cfg.min_cluster_points:
                continue
            # The first video must have looked through the place (the rule of verdict()).
            st = states[t][body(inside, clouds[t])]
            decided = np.isin(st, (SEEN, FREE)).mean()
            if decided < cfg.min_observed or (st == SEEN).mean() >= (1 - cfg.confirm_ratio) * decided:
                continue
            candidates.append((o, Obj(t, inside, inside, {i: mask})))
    embed([c for _, c in candidates], images, device)
    found, xyz = [], (clouds[0].xyz, clouds[1].xyz)
    for o, cand in candidates:
        a, b = (o, cand) if o.side == 0 else (cand, o)
        if taken[cand.side][cand.idx].any() or not same_kind(a, b, xyz, cfg):
            continue
        motion = fit_move(a, b, xyz, cfg)
        if motion.distance < cfg.min_move:
            continue
        end, other = ("new", "before") if o.side == 0 else ("old", "after")
        o.reason += f"; its {end} place, which the {other} video saw empty, was found by appearance"
        cand.status, cand.shares, cand.reason = o.status, o.shares, o.reason
        taken[cand.side][cand.idx] = True
        found.append((a, b, motion))
    return found
