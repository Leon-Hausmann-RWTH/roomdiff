import numpy as np
import torch
from PIL import Image

from .config import Config
from .evidence import SEEN
from .geometry import Cloud, Views, backproject
from .moves import Motion, footprint_motion, register
from .objects import (
    DINO_REPO,
    DINO_REVISION,
    Obj,
    describe,
    free_device,
    hires,
    lift,
    sam_masks,
    visible,
)

PATCHES = 64  # DINOv2 patches along the long image side for the dense search


def dense_features(images: np.ndarray, device) -> np.ndarray:
    """(N, rows, cols, 768) unit DINOv2 patch features of whole keyframes."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(DINO_REPO, revision=DINO_REVISION).to(device).eval()
    h, w = images.shape[1:3]
    rows, cols = (round(PATCHES * h / w), PATCHES) if w >= h else (PATCHES, round(PATCHES * w / h))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for im in images:
        x = np.array(Image.fromarray(im).resize((cols * 14, rows * 14)))
        x = torch.from_numpy(x).to(device).permute(2, 0, 1)[None].float().div(255)
        with torch.no_grad():
            tok = model(pixel_values=(x - mean) / std).last_hidden_state[0, 1:]
        out.append(torch.nn.functional.normalize(tok, dim=-1).view(rows, cols, -1).half().cpu().numpy())
    del model
    free_device(device)
    return np.stack(out)


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


def descriptor(o: Obj, feats: np.ndarray) -> np.ndarray:
    rows, cols = feats.shape[1:3]
    parts = []
    for i, m in o.masks.items():
        small = np.asarray(Image.fromarray(m).resize((cols, rows), Image.Resampling.BOX)) > 0.4
        if small.any():
            parts.append(feats[i][small].astype(np.float32).mean(0))
    f = np.mean(parts, 0)
    return f / np.linalg.norm(f)


def find_other_ends(
    lonely: list[Obj], clouds: tuple[Cloud, Cloud], states, views, images, taken, cfg: Config, device
) -> list[tuple[Obj, Obj, Motion]]:
    """For a confirmed removal (or addition) with no partner, look for the same object in the other
    video at a place the first video never saw. A move whose new place one video skipped otherwise
    reads as a removal: the brief's "incomplete observations" case.
    """
    if not lonely:
        return []
    feats = [dense_features(images[s], device) for s in (0, 1)]
    pts = [patch_points(views[s], *feats[s].shape[1:3]) for s in (0, 1)]
    found, prompts, owners = [], {}, {}
    for o in lonely:
        t = 1 - o.side
        sim = feats[t].astype(np.float32) @ descriptor(o, feats[o.side])  # (N, rows, cols)
        centre = clouds[o.side].xyz[o.idx].mean(0)
        far = np.linalg.norm((pts[t] - centre)[..., :2], axis=-1) > 0.4
        sim = np.where(far & ~np.isnan(pts[t][..., 0]), sim, -1)
        i, r, c = np.unravel_index(np.argmax(sim), sim.shape)
        best = pts[t][i, r, c]
        # The same place must look alike from several frames, not one lucky patch.
        near = np.linalg.norm(pts[t] - best, axis=-1) < 0.15
        frames = len(np.unique(np.nonzero((sim >= 0.9 * sim[i, r, c]) & near)[0]))
        if sim[i, r, c] < cfg.search_similarity or frames < 2:
            continue
        rows, cols = sim.shape[1:]
        hh, ww = images[t].shape[1:3]
        px = np.array([(c + 0.5) * ww / cols, (r + 0.5) * hh / rows])
        # Prompt with a box the size the object would have at that depth (keyframes are at 2x focal).
        size = np.ptp(clouds[o.side].xyz[o.idx], 0).max()
        z = (np.linalg.inv(views[t].c2w[i])[:3] @ np.append(best, 1))[2]
        half = 0.35 * size * 2 * views[t].K[i, 0, 0] / max(z, 0.3)
        corners = np.array([px - half, px + half, px])
        prompts.setdefault((t, int(i)), []).append(corners)
        owners.setdefault((t, int(i)), []).append((o, px))
    for (t, i), results in sam_masks(prompts, images, device).items():
        for (o, px), (_, masks, _, _) in zip(owners[(t, i)], results):
            # The smallest of SAM's three masks that holds the matched point: the object, not its shelf.
            hits = [k for k in masks if k[int(px[1]), int(px[0])]]
            if not hits:
                continue
            m = min(hits, key=lambda k: k.sum())
            cloud = clouds[t]
            u, v, ok = visible(cloud.xyz, views[t], i)
            q = np.round(hires(u, v)).astype(int).clip([0, 0], [m.shape[1] - 1, m.shape[0] - 1])
            inside = np.nonzero(ok & m[q[:, 1], q[:, 0]] & ~taken[t])[0]
            if len(inside) < cfg.min_cluster_points:
                continue
            cand = Obj(t, inside, inside, {i: m})
            lift(cand, cloud, views[t])
            # The first video must not have seen the object standing there; then it would not have moved.
            z = cloud.xyz[cand.idx, 2]
            body = cand.idx[z > z.min() + min(0.10, max(0.03, 0.3 * float(np.ptp(z))))]
            if len(body) == 0 or np.mean(states[t][body] == SEEN) >= 0.5:
                continue
            describe([cand], images, device)
            ha, hb = (np.ptp(clouds[x.side].xyz[x.idx][:, 2]) + 0.02 for x in (o, cand))
            if cand.feature @ o.feature < cfg.match_similarity or max(ha / hb, hb / ha) >= 2.0:
                continue
            a, b = (o, cand) if o.side == 0 else (cand, o)
            src, dst = clouds[0].xyz[a.idx], clouds[1].xyz[b.idx]
            m_ = register(src, dst)
            m_ = m_ if m_.fit >= cfg.min_fit else footprint_motion(src, dst)
            if m_.distance >= cfg.min_move:
                cand.status = o.status
                cand.reason = (
                    "its place was never filmed by the " + ("before" if t == 1 else "after") + " video"
                )
                found.append((a, b, m_))
    return found
