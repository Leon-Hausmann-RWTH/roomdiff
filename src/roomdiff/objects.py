from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

from .config import Config, free_device
from .evidence import FREE, OCCLUDED, SEEN, UNREACHED
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


def best_frames(xyz: np.ndarray, views: Views, k: int = 3) -> list[int]:
    """Up to k keyframes that see the most points, spaced apart in time."""
    counts = np.array([visible(xyz, views, i)[2].sum() for i in range(len(views.depth))])
    order = [int(i) for i in np.argsort(-counts) if counts[i] >= 8]
    picked = []
    for i in order:
        if all(abs(i - j) >= max(2, len(counts) // 10) for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    return picked


def hires(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reconstruction pixels to keyframe image pixels (twice the size, centres aligned): (N, 2)."""
    return np.stack([2 * u + 0.5, 2 * v + 0.5], -1)


def sam_masks(jobs: dict[tuple[int, int], list[np.ndarray]], images, device) -> dict:
    """SAM 2.1 per (side, keyframe), prompted with a box around each pixel set and a point at its centre.
    Returns per job a list of (pixels, masks (3, H, W), scores (3,), box)."""
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
            o.masks[key[1]] = pick if pick is not None else splat(px, m.shape[1:])


def splat(px: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """A mask from projected point pixels, dilated over the gaps between 2 cm points."""
    m = np.zeros(hw, bool)
    q = np.round(px).astype(int).clip([0, 0], [hw[1] - 1, hw[0] - 1])
    m[q[:, 1], q[:, 0]] = True
    return binary_dilation(m, iterations=4)


def choose_mask(masks: np.ndarray, scores: np.ndarray, px: np.ndarray, box) -> np.ndarray | None:
    """Best-scoring mask that covers most points and is not much larger than the box, else None."""
    h, w = masks.shape[1:]
    q = np.round(px).astype(int).clip([0, 0], [w - 1, h - 1])
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    best, best_score = None, -1.0
    for m, s in zip(masks, scores):
        if m[q[:, 1], q[:, 0]].mean() >= 0.6 and m.sum() <= 1.5 * box_area and s > best_score:
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
    full, core = np.ptp(cloud.xyz[idx], 0).max(), np.ptp(seed, 0).max()
    # Much larger than the seed means the mask leaked onto the support; keep the seed then.
    o.idx = idx if full <= max(3 * core, core + 0.3) else o.seed


def same_object(p: Obj, o: Obj) -> bool:
    """Same side and either many shared points or one mask mostly inside the other in a shared view."""
    if p.side != o.side:
        return False
    if len(np.intersect1d(p.idx, o.idx)) >= 0.2 * min(len(p.idx), len(o.idx)):
        return True
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


def dino(device):
    from transformers import AutoModel

    return AutoModel.from_pretrained(DINO_REPO, revision=DINO_REVISION).to(device).eval()


def tokens(model, rgb: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Unit DINOv2 patch tokens of an RGB image resized to size (multiples of 14): (rows, cols, 768)."""
    x = torch.from_numpy(np.array(Image.fromarray(rgb).resize(size))).to(model.device).permute(2, 0, 1)[None]
    mean = torch.tensor([0.485, 0.456, 0.406], device=model.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=model.device).view(1, 3, 1, 1)
    with torch.no_grad():
        tok = model(pixel_values=(x.float() / 255 - mean) / std).last_hidden_state[0, 1:]
    return torch.nn.functional.normalize(tok, dim=-1).view(size[1] // 14, size[0] // 14, -1)


def patch_mask(mask: np.ndarray, cols: int, rows: int, share: float) -> np.ndarray:
    """(rows, cols) bool: patches whose area inside the mask exceeds share."""
    # Via uint8: Pillow resizes bool images with nearest neighbour whatever the filter.
    area = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((cols, rows), Image.Resampling.BOX))
    return area > share * 255


def embed(objs: list[Obj], images, device) -> None:
    """DINOv2 descriptor per object: patch tokens averaged in the mask as in SceneDiff, then over views."""
    model = dino(device)
    for o in objs:
        feats = []
        for i, m in o.masks.items():
            ys, xs = np.nonzero(m)
            pad = 0.1 * max(np.ptp(xs), np.ptp(ys)) + 4
            x0, x1 = int(max(xs.min() - pad, 0)), int(min(xs.max() + pad, m.shape[1] - 1)) + 1
            y0, y1 = int(max(ys.min() - pad, 0)), int(min(ys.max() + pad, m.shape[0] - 1)) + 1
            tok = tokens(model, images[o.side][i][y0:y1, x0:x1], (224, 224))
            sel = torch.from_numpy(patch_mask(m[y0:y1, x0:x1], 16, 16, 0.3)).to(device)
            f = tok[sel].mean(0) if sel.any() else tok.reshape(-1, tok.shape[-1]).mean(0)
            feats.append(torch.nn.functional.normalize(f, dim=0).cpu().numpy())
        if feats:
            f = np.mean(feats, 0)
            o.feature = f / np.linalg.norm(f)
    del model
    free_device(device)


def body(idx: np.ndarray, cloud: Cloud) -> np.ndarray:
    """The object's points above its contact zone, where it touches the support and still tests as seen."""
    z = cloud.xyz[idx, 2]
    lo, hi = np.percentile(z, [3, 97])
    keep = idx[z > lo + min(0.10, max(0.03, 0.3 * (hi - lo)))]
    return keep if hi - lo > 0.06 and len(keep) >= 0.3 * len(idx) else idx


def verdict(o: Obj, cloud: Cloud, state: np.ndarray, cfg: Config) -> None:
    """Status and reason from how much of the object the other video looked through."""
    st = state[body(o.idx, cloud)]
    names = (("free", FREE), ("seen", SEEN), ("occluded", OCCLUDED), ("unreached", UNREACHED))
    o.shares = {k: round(float(np.mean(st == s)), 3) for k, s in names} | {"points": int(len(st))}
    decided = o.shares["free"] + o.shares["seen"]
    ratio = o.shares["free"] / max(decided, 1e-9)
    other, still = ("after", "still") if o.side == 0 else ("before", "already")
    looked, saw = f"{100 * ratio:.0f} %", f"{100 * (1 - ratio):.0f} %"
    z = cloud.xyz[o.idx, 2]
    height = np.percentile(z, 97) - np.percentile(z, 3)
    if decided < cfg.min_observed:
        hidden = o.shares["occluded"] > o.shares["unreached"]
        why = "hidden behind other things" if hidden else "seen too rarely"
        o.status = "unverified"
        o.reason = f"the {other} video could check only {100 * decided:.0f} % of it; the rest was {why}"
    elif ratio >= cfg.confirm_ratio:
        o.status = "confirmed"
        o.reason = f"the {other} video looked through {looked} of the points it could check"
    elif ratio >= cfg.partial_ratio:
        o.status = "partial"
        # body() keeps the contact zone of objects this flat.
        o.reason = (
            f"the {other} video looked through {looked} of it; at {100 * height:.0f} cm tall, the rest "
            "cannot be told apart from the surface under it"
            if height <= 0.06
            else f"the {other} video looked through {looked} of it and {still} saw {saw}"
        )
    else:
        o.status, o.reason = "dismissed", f"the {other} video {still} saw {saw} of it"
