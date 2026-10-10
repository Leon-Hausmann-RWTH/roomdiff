import gc
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_erosion

from .config import Config
from .evidence import FREE, OCCLUDED, SEEN, UNREACHED, Evidence, point_states
from .geometry import Cloud, Views, project

SAM_REPO, SAM_REVISION = "facebook/sam2.1-hiera-base-plus", "b7320756a13354e7530a63935656d35b2f91a290"
DINO_REPO, DINO_REVISION = "facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"


@dataclass
class Obj:
    side: int  # 0 before, 1 after
    idx: np.ndarray  # point indices in that side's cloud
    seed: np.ndarray  # the free points that started it
    masks: dict[int, np.ndarray] = field(default_factory=dict)  # keyframe -> (2H, 2W) bool mask
    feature: np.ndarray | None = None  # unit DINOv2 descriptor
    shares: dict = field(default_factory=dict)  # evidence shares over the object's points
    status: str = ""
    reason: str = ""


def visible(xyz: np.ndarray, views: Views, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates in keyframe i and whether each point is the surface that frame actually saw."""
    u, v, z = project(xyz, views.K[i], views.c2w[i])
    h, w = views.depth.shape[1:]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (z > 0.05) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    d = np.zeros_like(z)
    d[inside] = views.depth[i][vi[inside], ui[inside]]
    return u, v, inside & (d > 0) & (np.abs(d - z) < 0.05 + 0.03 * z)


def best_frames(xyz: np.ndarray, views: Views, k: int = 3, min_points: int = 8) -> list[int]:
    counts = np.array([visible(xyz, views, i)[2].sum() for i in range(len(views.depth))])
    order = [int(i) for i in np.argsort(-counts) if counts[i] >= min_points]
    # Prefer well-separated views: a neighbouring keyframe adds little new about the object.
    picked = []
    for i in order:
        if all(abs(i - j) >= max(2, len(counts) // 10) for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    return picked


def hires(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Keyframe images are twice the reconstruction size; pixel centres map as 2x + 0.5.
    return np.stack([2 * u + 0.5, 2 * v + 0.5], -1)


def free_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def sam_masks(jobs: dict[tuple[int, int], list[np.ndarray]], images, device) -> dict:
    """SAM 2.1 for each (side, keyframe): one prompt per pixel set (box around it, point at its centre).

    Returns, per job, a list of (pixels, masks (3, H, W), scores (3,), box) in the order given.
    """
    from transformers import Sam2Model, Sam2Processor

    model = Sam2Model.from_pretrained(SAM_REPO, revision=SAM_REVISION).to(device).eval()
    processor = Sam2Processor.from_pretrained(SAM_REPO, revision=SAM_REVISION)
    out = {}
    for (side, i), pixel_sets in jobs.items():
        boxes, points = [], []
        for px in pixel_sets:
            lo, hi = px.min(0), px.max(0)
            pad = 0.15 * (hi - lo) + 8
            boxes.append([float(x) for x in (*(lo - pad), *(hi + pad))])
            points.append([[float(x) for x in px[np.argmin(np.linalg.norm(px - np.median(px, 0), axis=1))]]])
        inputs = processor(
            images=images[side][i],
            input_boxes=[boxes],
            input_points=[points],
            input_labels=[[[1]] * len(pixel_sets)],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            res = model(**inputs, multimask_output=True)
        masks = processor.post_process_masks(res.pred_masks.cpu(), inputs["original_sizes"])[0].numpy()
        scores = res.iou_scores[0].float().cpu().numpy()  # (prompts, 3)
        out[(side, i)] = list(zip(pixel_sets, masks.astype(bool), scores, boxes))
    del model
    free_device(device)
    return out


def segment(objs: list[Obj], clouds: tuple[Cloud, Cloud], views: tuple[Views, Views], images, device) -> None:
    """Masks in each object's best views, prompted with the box and centre of its free points."""
    owners, jobs = {}, {}
    for o in objs:
        seed_xyz = clouds[o.side].xyz[o.seed]
        for i in best_frames(seed_xyz, views[o.side]):
            u, v, ok = visible(seed_xyz, views[o.side], i)
            jobs.setdefault((o.side, i), []).append(hires(u[ok], v[ok]))
            owners.setdefault((o.side, i), []).append(o)
    for key, results in sam_masks(jobs, images, device).items():
        for o, (px, m, s, box) in zip(owners[key], results):
            pick = choose_mask(m, s, px, box)
            if pick is not None:
                o.masks[key[1]] = pick


def choose_mask(masks: np.ndarray, scores: np.ndarray, px: np.ndarray, box) -> np.ndarray | None:
    # SAM often answers a box prompt with the supporting surface (pillow -> sofa). Accept only masks
    # that cover most of the free points and are not much larger than the prompt box.
    h, w = masks.shape[1:]
    q = np.round(px).astype(int).clip([0, 0], [w - 1, h - 1])
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    best, best_score = None, -1.0
    for m, s in zip(masks, scores):
        coverage = m[q[:, 1], q[:, 0]].mean()
        if coverage >= 0.6 and m.sum() <= 1.5 * box_area and s > best_score:
            best, best_score = m, s
    return best


def lift(o: Obj, cloud: Cloud, views: Views) -> None:
    """Grow the object to the points that fall inside its masks in most views where they are visible."""
    if not o.masks:
        o.idx = o.seed
        return
    seed = cloud.xyz[o.seed]
    lo, hi = seed.min(0) - 0.3, seed.max(0) + 0.3
    near = np.nonzero(np.all((cloud.xyz >= lo) & (cloud.xyz <= hi), 1))[0]
    inside, seen = np.zeros(len(near), int), np.zeros(len(near), int)
    for i, m in o.masks.items():
        m = binary_erosion(m, iterations=2)
        u, v, ok = visible(cloud.xyz[near], views, i)
        p = np.round(hires(u, v)).astype(int).clip([0, 0], [m.shape[1] - 1, m.shape[0] - 1])
        seen += ok
        inside += ok & m[p[:, 1], p[:, 0]]
    grown = near[(inside >= 1) & (inside >= 0.5 * seen)]
    idx = np.union1d(o.seed, grown)
    # A mask that leaked onto the support makes the object far larger than its free core; keep the core then.
    span = lambda a: np.ptp(cloud.xyz[a], 0).max()  # noqa: E731
    o.idx = idx if span(idx) <= max(3 * span(o.seed), span(o.seed) + 0.3) else o.seed


def same_object(p: Obj, o: Obj) -> bool:
    if p.side != o.side:
        return False
    if len(np.intersect1d(p.idx, o.idx)) >= 0.2 * min(len(p.idx), len(o.idx)):
        return True
    # Occlusion splits one object into several free clusters; SAM then returns the same whole-object
    # mask for each of them in a view they share.
    for i in set(p.masks) & set(o.masks):
        a, b = p.masks[i], o.masks[i]
        if (a & b).sum() >= 0.5 * min(a.sum(), b.sum()):
            return True
    return False


def merge(objs: list[Obj]) -> list[Obj]:
    out: list[Obj] = []
    for o in sorted(objs, key=lambda o: -len(o.idx)):
        for p in out:
            if same_object(p, o):
                p.idx, p.seed = np.union1d(p.idx, o.idx), np.union1d(p.seed, o.seed)
                for i, m in o.masks.items():
                    p.masks[i] = p.masks[i] | m if i in p.masks else m
                break
        else:
            out.append(o)
    return out


def describe(objs: list[Obj], images, device) -> None:
    """DINOv2 descriptor per object: patch tokens averaged inside the mask, averaged over views.

    DINO features re-identify the same object across views far better than CLIP-style features
    (SOCO benchmark, 2026); averaging inside the mask follows SceneDiff (Wu et al., ECCV 2026).
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(DINO_REPO, revision=DINO_REVISION).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for o in objs:
        feats = []
        for i, m in o.masks.items():
            ys, xs = np.nonzero(m)
            pad = 0.1 * max(np.ptp(xs), np.ptp(ys)) + 4
            x0, x1 = int(max(xs.min() - pad, 0)), int(min(xs.max() + pad, m.shape[1] - 1)) + 1
            y0, y1 = int(max(ys.min() - pad, 0)), int(min(ys.max() + pad, m.shape[0] - 1)) + 1
            crop = np.array(Image.fromarray(images[o.side][i][y0:y1, x0:x1]).resize((224, 224)))
            small = np.asarray(Image.fromarray(m[y0:y1, x0:x1]).resize((16, 16), Image.Resampling.BOX)) > 0.3
            x = torch.from_numpy(crop).to(device).permute(2, 0, 1)[None].float().div(255)
            with torch.no_grad():
                tokens = model(pixel_values=(x - mean) / std).last_hidden_state[0, 1:]  # (256, 768)
            sel = torch.from_numpy(small.ravel()).to(device)
            f = tokens[sel].mean(0) if sel.any() else tokens.mean(0)
            feats.append(torch.nn.functional.normalize(f, dim=0).cpu().numpy())
        if feats:
            f = np.mean(feats, 0)
            o.feature = f / np.linalg.norm(f)
    del model
    free_device(device)


def verdict(o: Obj, cloud: Cloud, ev: Evidence, cfg: Config) -> None:
    """Confirmed only where the other video positively saw free space; otherwise say why not."""
    z = cloud.xyz[o.idx, 2]
    # The lowest part of an object touches its support, which is still there, and tests as seen up to
    # roughly the depth band times the sine of the viewing angle: up to about 10 cm.
    contact = min(0.10, max(0.03, 0.3 * float(np.ptp(z))))
    body = o.idx[z > z.min() + contact] if np.ptp(z) > 0.06 else o.idx
    state = point_states(ev, cfg)[body]
    names = (("free", FREE), ("seen", SEEN), ("occluded", OCCLUDED), ("unreached", UNREACHED))
    o.shares = {k: round(float(np.mean(state == s)), 3) for k, s in names} | {"points": int(len(body))}
    decided = o.shares["free"] + o.shares["seen"]
    ratio = o.shares["free"] / max(decided, 1e-9)
    other = "after" if o.side == 0 else "before"
    pct = lambda x: f"{100 * x:.0f} %"  # noqa: E731
    if decided < cfg.min_observed:
        why = "hidden behind other things" if o.shares["occluded"] > o.shares["unreached"] else "out of view"
        o.status, o.reason = (
            "unverified",
            f"the {other} video could check only {pct(decided)} of it; the rest was {why}",
        )
    elif ratio >= cfg.confirm_ratio:
        o.status, o.reason = (
            "confirmed",
            f"the {other} video looked through {pct(ratio)} of the points it could check",
        )
    elif ratio >= cfg.partial_ratio:
        o.status, o.reason = (
            "partial",
            f"the {other} video looked through {pct(ratio)} of it; the rest is still there",
        )
    else:
        o.status, o.reason = "dismissed", f"the {other} video still saw {pct(1 - ratio)} of it"
