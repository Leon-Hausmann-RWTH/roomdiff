"""Evaluation on the SceneDiff benchmark (huggingface.co/datasets/yuqun/SceneDiff, MIT), unpacked in data/.

Scores the reported changes against the ground truth lifted into 3D, next to a naive baseline. Optionally
exports masks for the official evaluator, or runs half-video negative controls instead of the pairs.

Run from the repository root: uv run python scripts/evaluate.py --set tuning
"""

import io
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from clips import reencode
from PIL import Image
from pycocotools import mask as rle
from scipy.ndimage import binary_erosion
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from roomdiff.changes import Change
from roomdiff.config import Config, pick_device
from roomdiff.evidence import FREE, SEEN, UNREACHED
from roomdiff.geometry import backproject, components
from roomdiff.objects import Obj, choose_mask, hires, sam_masks, splat, visible
from roomdiff.pipeline import Result, analyse, write_outputs
from roomdiff.video import probe

DATA = Path("data/scenediff_benchmark/data")
TUNING = [
    "living_room_1_living_room_2",
    "bedroom_2_bedroom_4",
    "lounge_1_lounge_2",
    "living_room_35_living_room_36",
]
REPORT = [
    "living_room_11_living_room_12", "living_room_13_living_room_14", "living_room_19_living_room_20",
    "living_room_43_living_room_44", "living_room_3_living_room_4", "livingroom_1_livingroom_2",
    "living_room_37_living_room_38", "bedroom_14_bedroom_15", "bedroom_20_bedroom_21", "bedroom_6_bedroom_7",
    "bedroom_10_bedroom_11", "bedroom_32_bedroom_33", "bedroom_36_bedroom_37", "office_1_office_2",
    "office_3_office_4", "lounge_11_lounge_12", "study_1_study_2", "recreation_room_1_recreation_room_2",
]  # fmt: skip
FLAT = 0.08  # metres; lower ground-truth objects count as flat
COUNTS = ("pred", "tp", "pred_confirmed", "tp_confirmed", "type_ok", "tp_flat", "fp_unseen",
          "moves", "moves_placed")  # fmt: skip
SAFE = {("builtins", n) for n in ("dict", "list", "tuple", "set", "str", "int", "float", "bool", "bytes")} | {
    (m, n) for m in ("numpy.core.multiarray", "numpy._core.multiarray") for n in ("_reconstruct", "scalar")
} | {("numpy", "ndarray"), ("numpy", "dtype")}  # fmt: skip


@dataclass
class Args:
    set: str = "tuning"
    """tuning or report."""
    pairs: tuple[str, ...] = ()
    """Explicit pair names; overrides --set."""
    out: Path = Path("runs/eval")
    write: bool = False
    """Also write the normal outputs (viewer, report) per pair."""
    official: bool = False
    """Export object_masks.pkl per pair for SceneDiff's evaluate_multiview.py."""
    control: bool = False
    """Instead of the pairs, run each pair's first video against itself split in two halves."""
    config: Config = field(default_factory=Config)


class SafeUnpickler(pickle.Unpickler):
    """Unpickler for downloaded files: loads only containers and numpy arrays."""

    def find_class(self, module, name):
        if (module, name) not in SAFE:
            raise pickle.UnpicklingError(f"refusing {module}.{name}")
        return super().find_class(module, name)


def load_gt(pair: str) -> dict:
    return SafeUnpickler(io.BytesIO((DATA / pair / "segments.pkl").read_bytes())).load()


def videos(pair: str) -> tuple[Path, Path]:
    found = sorted(p for p in (DATA / pair).iterdir() if p.name.startswith("original_video"))
    return found[0], found[1]


def object_type(obj: dict) -> str:
    return "moved" if obj["in_video1"] and obj["in_video2"] else "removed" if obj["in_video1"] else "added"


def scored_times(path: Path) -> tuple[float, ...]:
    """Whole seconds into the video: the times of the frames SceneDiff scores."""
    duration = probe(path)[2]
    return tuple(float(k) for k in range(int(duration) + 1) if k <= duration - 0.02)


def scored_frames(res: Result, side: int) -> dict[int, int]:
    """SceneDiff frame k (k seconds into the video) -> keyframe index."""
    t = res.recon.keys[side].times
    return {
        k: int(np.abs(t - k).argmin()) for k in range(int(round(t.max())) + 1) if np.abs(t - k).min() < 0.05
    }


def to_recon(mask: np.ndarray, res: Result, side: int) -> np.ndarray:
    """A source-resolution mask cropped and resized to the reconstruction's depth maps."""
    x0, y0, x1, y1 = res.recon.keys[side].crop
    h, w = res.views[side].depth.shape[1:]
    return np.asarray(Image.fromarray(mask[y0:y1, x0:x1]).resize((w, h), Image.Resampling.NEAREST))


def lift(gt: dict, res: Result, side: int) -> dict[int, np.ndarray]:
    """World points of each GT object, from the scored keyframes of one video."""
    views, keys, pts = res.views[side], res.recon.keys[side], defaultdict(list)
    for k, i in scored_frames(res, side).items():
        world = backproject(views.depth[i], views.K[i], views.c2w[i])
        for oid, masks in gt[f"video{side + 1}_objects"].items():
            # GT masks are keyed by frame index at 30 fps.
            if str(30 * k) in masks:
                m = rle.decode(masks[str(30 * k)]).astype(bool)
                assert m.shape == tuple(keys.source_hw), (m.shape, keys.source_hw)
                m = binary_erosion(to_recon(m, res, side), iterations=1) & (views.depth[i] > 0)
                pts[oid].append(world[m])
    return {o: np.concatenate(p) for o, p in pts.items() if sum(map(len, p)) >= 10}


def within(a: np.ndarray, b: np.ndarray, r: float = 0.05) -> np.ndarray:
    return cKDTree(b).query(a, distance_upper_bound=r)[0] < r


def match(preds: list[Change], objects: list[dict], gt_pts, res: Result):
    """One-to-one: a change matches a GT object if a quarter of its points lie within 5 cm of it.
    Returns the matches and the overlaps per end, (2, changes, objects)."""
    S = np.zeros((2, len(preds), len(objects)))
    for i, c in enumerate(preds):
        for j, o in enumerate(objects):
            for s, obj in ((0, c.before), (1, c.after)):
                if obj is not None and o["id"] in gt_pts[s]:
                    S[s, i, j] = float(within(res.clouds[s].xyz[obj.idx], gt_pts[s][o["id"]]).mean())
    best = S.max(0)
    rows, cols = linear_sum_assignment(-best) if best.size else ([], [])
    return [(r, c) for r, c in zip(rows, cols) if best[r, c] >= 0.25], S


def score(preds: list[Change], objects: list[dict], gt_pts, res: Result) -> dict:
    m, S = match(preds, objects, gt_pts, res)
    flat = {j for j, o in enumerate(objects) if o["height_m"] < FLAT}
    moves = [(r, c) for r, c in m if preds[r].kind == objects[c]["type"] == "moved"]
    # A found move is placed right only if both of its ends lie on the object.
    placed = [(r, c) for r, c in moves if min(S[0, r, c], S[1, r, c]) >= 0.25]
    return {
        "pred": len(preds),
        "tp": len(m),
        "pred_confirmed": sum(c.status == "confirmed" for c in preds),
        "tp_confirmed": sum(preds[r].status == "confirmed" for r, _ in m),
        "type_ok": sum(preds[r].kind == objects[c]["type"] for r, c in m),
        "tp_flat": sum(c in flat for _, c in m),
        "matches": [(preds[r].id, objects[c]["id"]) for r, c in m],
        "moves": len(moves),
        "moves_placed": len(placed),
        "move_error_m": [
            round(abs(preds[r].motion.distance - objects[c]["shift_m"]), 3)
            for r, c in placed
            if preds[r].motion is not None and objects[c].get("shift_m") is not None
        ],
        # Ends of unmatched changes whose points the other video mostly saw in too few frames.
        "fp_unseen": sum(
            bool(np.mean(res.states[s][o.idx] == UNREACHED) > 0.5)
            for i, c in enumerate(preds)
            if i not in {r for r, _ in m}
            for s, o in ((0, c.before), (1, c.after))
            if o is not None
        ),
    }


def naive(res: Result, cfg: Config) -> list[Change]:
    """Nearest-neighbour differencing as in VGGT-CD: points with nothing of the other cloud within 5 cm,
    clustered like roomdiff's candidates."""
    out = []
    for s, kind in ((0, "removed"), (1, "added")):
        xyz = res.clouds[s].xyz
        lonely = np.nonzero(~within(xyz, res.clouds[1 - s].xyz))[0]
        if len(lonely) == 0:
            continue
        labels = components(xyz[lonely], cfg.cluster_voxel)
        for lab in np.unique(labels):
            idx = lonely[labels == lab]
            if len(idx) >= cfg.min_cluster_points:
                ends = (Obj(s, idx, idx), None) if s == 0 else (None, Obj(s, idx, idx))
                out.append(Change(len(out) + 1, kind, "confirmed", "", *ends))
    return out


def export_official(res: Result, out: Path, device) -> None:
    """Write object_masks.pkl for SceneDiff's evaluator: per change and scored frame, a SAM mask at the
    source resolution, prompted with the change's projected points."""
    jobs, owners = {}, {}
    for c in res.changes:
        for s, o in ((0, c.before), (1, c.after)):
            if o is None:
                continue
            for k, i in scored_frames(res, s).items():
                u, v, ok = visible(res.clouds[s].xyz[o.idx], res.views[s], i)
                if ok.sum() >= 5:
                    jobs.setdefault((s, i), []).append(hires(u[ok], v[ok]))
                    owners.setdefault((s, i), []).append((c, k))
    images = (res.recon.keys[0].images, res.recon.keys[1].images)
    h, w = res.recon.keys[0].source_hw
    pred = {"H": h, "W": w}
    for (s, i), results in sam_masks(jobs, images, device).items():
        keys = res.recon.keys[s]
        x0, y0, x1, y1 = keys.crop
        for (c, k), (px, masks, scores, box) in zip(owners[(s, i)], results):
            m = choose_mask(masks, scores, px, box)
            m = splat(px, masks.shape[1:]) if m is None else m
            full = np.zeros(keys.source_hw, bool)
            full[y0:y1, x0:x1] = np.asarray(
                Image.fromarray(m).resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
            )
            code = rle.encode(np.asfortranarray(full.astype(np.uint8)))
            cost = 1.0 if c.status == "confirmed" else 0.4  # the evaluator ranks objects by mean cost
            pred.setdefault(c.id, {}).setdefault(f"video_{s + 1}", {})[k] = {"mask": code, "cost": cost}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "object_masks.pkl", "wb") as f:
        pickle.dump(pred, f)


def evaluate_pair(pair: str, args: Args) -> dict:
    gt = load_gt(pair)
    v1, v2 = videos(pair)
    res = analyse(v1, v2, args.config, (scored_times(v1), scored_times(v2)))
    if args.write:
        write_outputs(res, args.out / pair, args.config)
    if args.official:
        export_official(res, args.out / "official" / pair, pick_device(args.config.device))
    gt_pts = [lift(gt, res, 0), lift(gt, res, 1)]

    # Share of decided points away from every GT object that came out free.
    static = []
    for s in (0, 1):
        cloud, state = res.clouds[s], res.states[s]
        changed = np.zeros(len(cloud.xyz), bool)
        for o in gt_pts[s].values():
            changed |= within(cloud.xyz, o, 0.1)
        decided = ~changed & ((state == FREE) | (state == SEEN))
        static.append(round(float(np.mean(state[decided] == FREE)), 4))

    objects = []
    for oid, obj in enumerate(gt["objects"]):
        sides = [s for s in (0, 1) if oid in gt_pts[s]]
        if not sides:
            continue
        z = np.concatenate([gt_pts[s][oid][:, 2] for s in sides])
        diag = {"id": oid, "label": obj["label"], "type": object_type(obj)}
        diag["height_m"] = round(float(np.percentile(z, 95) - np.percentile(z, 5)), 3)
        if len(sides) == 2:
            shift = np.median(gt_pts[1][oid], 0) - np.median(gt_pts[0][oid], 0)
            diag["shift_m"] = round(float(np.linalg.norm(shift)), 3)
        for s in sides:
            d, j = cKDTree(res.clouds[s].xyz).query(gt_pts[s][oid], distance_upper_bound=0.03)
            free = res.states[s][j[d < 0.03]] == FREE
            diag[f"free_{'ab'[s]}"] = round(float(np.mean(free)) if free.size else 0.0, 3)
        objects.append(diag)

    ours = score(res.changes, objects, gt_pts, res)
    base = score(naive(res, args.config), objects, gt_pts, res)
    changes = [{"id": c.id, "type": c.kind, "status": c.status, "reason": c.reason} for c in res.changes]
    flat = sum(o["height_m"] < FLAT for o in objects)
    return {"pair": pair, "gt": len(objects), "gt_flat": flat, "ours": ours, "naive": base,
            "static_false_free": static, "objects": objects, "changes": changes}  # fmt: skip


def control_pair(pair: str, args: Args) -> dict:
    """Negative control: the two halves of the pair's first video show the same unchanged room."""
    src, folder = videos(pair)[0], args.out / "control" / pair
    a = reencode(src, folder / "first.mp4", part=(0, 0.5))
    b = reencode(src, folder / "second.mp4", part=(0.5, 1))
    res = analyse(a, b, args.config)
    if args.write:
        write_outputs(res, folder / "out", args.config)
    decided = [(s == FREE) | (s == SEEN) for s in res.states]
    return {
        "pair": pair,
        "changes": sum(c.status == "confirmed" for c in res.changes),
        "unverified": sum(c.status == "unverified" for c in res.changes),
        "naive_changes": len(naive(res, args.config)),
        "false_free": [round(float(np.mean(s[d] == FREE)), 4) for s, d in zip(res.states, decided)],
        "overlap": [round(float(d.mean()), 3) for d in decided],
    }


def summary(rows: list[dict]) -> dict:
    gt, flat = sum(r["gt"] for r in rows), sum(r["gt_flat"] for r in rows)
    out = {"pairs": len(rows), "gt_objects": gt, "gt_flat": flat}
    for who in ("ours", "naive"):
        t = {k: sum(r[who][k] for r in rows) for k in COUNTS}
        errors = [e for r in rows for e in r[who]["move_error_m"]]
        out[who] = {
            "recall": round(t["tp"] / max(gt, 1), 3),
            "precision": round(t["tp"] / max(t["pred"], 1), 3),
            "recall_confirmed": round(t["tp_confirmed"] / max(gt, 1), 3),
            "precision_confirmed": round(t["tp_confirmed"] / max(t["pred_confirmed"], 1), 3),
            "recall_standing": round((t["tp"] - t["tp_flat"]) / max(gt - flat, 1), 3),
            "recall_flat": round(t["tp_flat"] / max(flat, 1), 3),
            "type_accuracy": round(t["type_ok"] / max(t["tp"], 1), 3),
            "false_in_unseen_regions": t["fp_unseen"],
            "moves_placed": f"{t['moves_placed']} of {t['moves']}",
            "median_move_error_m": round(float(np.median(errors)), 3) if errors else None,
        }  # fmt: skip
    out["static_false_free"] = round(float(np.mean([x for r in rows for x in r["static_false_free"]])), 4)
    return out


def main(args: Args) -> None:
    pairs = list(args.pairs) or {"tuning": TUNING, "report": REPORT}[args.set]
    args.out.mkdir(parents=True, exist_ok=True)
    name = ("control_" if args.control else "") + (args.set if not args.pairs else "custom")
    rows = []
    for pair in pairs:
        print(f"== {pair}", flush=True)
        rows.append(control_pair(pair, args) if args.control else evaluate_pair(pair, args))
        print("  ", {k: v for k, v in rows[-1].items() if k not in ("objects", "changes")}, flush=True)
    total = rows if args.control else summary(rows)
    print(json.dumps(total))
    (args.out / f"{name}.json").write_text(json.dumps({"total": total, "pairs": rows}, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Args))
