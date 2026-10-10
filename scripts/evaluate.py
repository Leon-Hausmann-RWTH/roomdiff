"""3D evaluation on SceneDiff pairs: ground-truth masks lifted into 3D with the run's own depth and poses.

Run from the repository root: uv run python scripts/evaluate.py --set tuning
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from PIL import Image
from scenediff import load_gt, object_type, rle_decode, videos
from scipy.ndimage import binary_erosion
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from roomdiff.config import Config
from roomdiff.evidence import FREE, SEEN
from roomdiff.geometry import backproject
from roomdiff.pipeline import Result, analyse, write_outputs
from roomdiff.video import probe

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
SHOWCASE = ["living_room_17_living_room_18"]


@dataclass
class Args:
    set: str = "tuning"
    """tuning, report or showcase."""
    pairs: tuple[str, ...] = ()
    """Explicit pair names; overrides --set."""
    out: Path = Path("runs/eval")
    write: bool = False
    """Also write the normal outputs (viewer, json) per pair."""
    config: Config = field(default_factory=Config)


def scored_times(path: Path) -> tuple[float, ...]:
    duration = probe(path)[2]
    return tuple(float(k) for k in range(int(duration) + 1) if k <= duration - 0.02)


def lift(gt: dict, video: int, res: Result, side: int) -> dict[int, np.ndarray]:
    """World points of each GT object, from the scored keyframes of one video."""
    keys, views = res.recon.keys[side], res.views[side]
    h, w = views.depth.shape[1:]
    x0, y0, x1, y1 = keys.crop
    objs = gt[f"video{video}_objects"]
    pts = defaultdict(list)
    for k in range(int(round(keys.times.max())) + 1):
        i = int(np.abs(keys.times - k).argmin())
        if abs(keys.times[i] - k) > 0.05:
            continue
        world = backproject(views.depth[i], views.K[i], views.c2w[i])
        for oid, masks in objs.items():
            rle = masks.get(str(30 * k))
            if rle is None:
                continue
            m = rle_decode(rle)
            assert m.shape == tuple(keys.source_hw), (m.shape, keys.source_hw)
            m = np.asarray(Image.fromarray(m[y0:y1, x0:x1]).resize((w, h), Image.Resampling.NEAREST))
            m = binary_erosion(m, iterations=1) & (views.depth[i] > 0)
            pts[oid].append(world[m])
    return {o: np.concatenate(p) for o, p in pts.items() if sum(map(len, p)) >= 10}


def frac_within(a: np.ndarray, b: np.ndarray, r: float) -> float:
    return float(np.mean(cKDTree(b).query(a, distance_upper_bound=r)[0] < r))


def evaluate_pair(pair: str, args: Args) -> dict:
    gt = load_gt(pair)
    v1, v2 = videos(pair)
    res = analyse(v1, v2, args.config, (scored_times(v1), scored_times(v2)))
    if args.write:
        write_outputs(res, args.out / pair)
    (ca, cb), (sa, sb) = res.clouds, res.states
    gt_pts = [lift(gt, 1, res, 0), lift(gt, 2, res, 1)]

    static = []
    for side, cloud, state in ((0, ca, sa), (1, cb, sb)):
        # Points that no scored frame marks as changed, and that the other video decided on.
        changed = np.zeros(len(cloud.xyz), bool)
        for o in gt_pts[side].values():
            changed |= cKDTree(o).query(cloud.xyz, distance_upper_bound=0.1)[0] < 0.1
        decided = (~changed) & ((state == FREE) | (state == SEEN))
        static.append(float(np.mean(state[decided] == FREE)))

    objects = []
    for oid, obj in enumerate(gt["objects"]):
        sides = [s for s in (0, 1) if oid in gt_pts[s]]
        if not sides:
            continue
        diag = {"id": oid, "label": obj["label"], "type": object_type(obj)}
        for s, cloud, state in ((0, ca, sa), (1, cb, sb)):
            if s in sides:
                d, j = cKDTree(cloud.xyz).query(gt_pts[s][oid], distance_upper_bound=0.03)
                hit = d < 0.03
                diag[f"free_{'ab'[s]}"] = round(
                    float(np.mean(state[j[hit]] == FREE)) if hit.any() else 0.0, 3
                )
                z = gt_pts[s][oid][:, 2]
                diag["height_m"] = round(float(np.percentile(z, 95) - np.percentile(z, 5)), 3)
        objects.append(diag)

    preds = [c for c in res.changes if c.status != "dismissed"]
    S = np.zeros((len(preds), len(objects)))
    for i, c in enumerate(preds):
        for j, o in enumerate(objects):
            for s, idx, cloud in ((0, c.before, ca), (1, c.after, cb)):
                if idx is not None and o["id"] in gt_pts[s]:
                    S[i, j] = max(S[i, j], frac_within(cloud.xyz[idx], gt_pts[s][o["id"]], 0.05))
    rows, cols = linear_sum_assignment(-S) if S.size else ([], [])
    matches = [(r, c) for r, c in zip(rows, cols) if S[r, c] >= 0.25]
    for r, c in matches:
        objects[c]["matched"] = {"change": preds[r].id, "type": preds[r].kind, "status": preds[r].status}
    out = {
        "pair": pair,
        "gt": len(objects),
        "pred": len(preds),
        "tp": len(matches),
        "type_ok": sum(preds[r].kind == objects[c]["type"] for r, c in matches),
        "static_false_free": [round(x, 4) for x in static],
        "objects": objects,
        "unmatched_pred": [
            {
                "id": preds[i].id,
                "type": preds[i].kind,
                "best_overlap": round(float(S[i].max()) if S.shape[1] else 0, 2),
            }
            for i in range(len(preds))
            if i not in {r for r, _ in matches}
        ],
    }
    return out


def main(args: Args) -> None:
    pairs = list(args.pairs) or {"tuning": TUNING, "report": REPORT, "showcase": SHOWCASE}[args.set]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in pairs:
        print(f"== {pair}")
        r = evaluate_pair(pair, args)
        rows.append(r)
        print(f"   gt {r['gt']} pred {r['pred']} tp {r['tp']} type ok {r['type_ok']}", flush=True)
    gt, pred, tp = (sum(r[k] for r in rows) for k in ("gt", "pred", "tp"))
    total = {
        "recall": round(tp / max(gt, 1), 3),
        "precision": round(tp / max(pred, 1), 3),
        "type_accuracy": round(sum(r["type_ok"] for r in rows) / max(tp, 1), 3),
        "static_false_free": round(float(np.mean([x for r in rows for x in r["static_false_free"]])), 4),
    }
    print(json.dumps(total))
    name = args.set if not args.pairs else "custom"
    (args.out / f"{name}.json").write_text(json.dumps({"total": total, "pairs": rows}, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Args))
