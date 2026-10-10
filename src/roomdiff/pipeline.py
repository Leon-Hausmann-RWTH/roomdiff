import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .changes import Change, assemble, candidates
from .config import Config, pick_device
from .describe import name_changes
from .evidence import (
    FREE,
    OCCLUDED,
    SEEN,
    UNREACHED,
    count_evidence,
    frame_conflicts,
    point_states,
    self_consistent,
)
from .geometry import Cloud, Views, cloud_from_views, subset
from .moves import match
from .objects import Obj, embed, lift, merge, segment, verdict
from .outputs import change_json, write_ply
from .reconstruct import Recon, reconstruct, to_views
from .report import write_cards, write_floorplan, write_report
from .search import find_other_ends
from .viewer import write_viewer
from .world import Frame, transform, world_frame


@dataclass
class Result:
    paths: tuple[Path, Path]
    recon: Recon
    frame: Frame
    views: tuple[Views, Views]
    clouds: tuple[Cloud, Cloud]
    states: tuple[np.ndarray, np.ndarray]  # per-point evidence state of each cloud against the other video
    changes: list[Change]
    dismissed: list[Obj]


def analyse(before: Path, after: Path, cfg: Config, required=((), ())) -> Result:
    recon = reconstruct(before, after, cfg, required)
    t = time.time()
    va, vb = to_views(recon, cfg)
    frame = world_frame(va, vb)
    va, vb = transform(va, frame.T), transform(vb, frame.T)
    print(f"world frame: camera {frame.camera_height:.2f} m above the floor ({time.time() - t:.1f} s)")
    if not frame.floor_found or not 0.5 < frame.camera_height < 2.2:
        print("warning: no plausible floor plane; heights and 'on the floor' statements may be wrong")

    t = time.time()
    for own, name in ((va, "before"), (vb, "after")):
        bad = np.nonzero(frame_conflicts(own, cfg) > cfg.max_frame_conflict)[0]
        own.depth[bad] = 0
        if len(bad):
            print(f"warning: dropped {name} keyframes {bad.tolist()}: the rest of the video contradicts them")
    clouds = []
    for own in (va, vb):
        c = cloud_from_views(own, cfg.voxel)
        clouds.append(subset(c, self_consistent(c.xyz, c.depth, own, cfg)))
    ca, cb = clouds
    ea, eb = count_evidence(ca.xyz, ca.depth, vb, cfg), count_evidence(cb.xyz, cb.depth, va, cfg)
    sa, sb = point_states(ea, cfg), point_states(eb, cfg)
    print(f"ray evidence: {len(ca.xyz)} + {len(cb.xyz)} points ({time.time() - t:.1f} s)")

    t = time.time()
    clouds, states, views, evidence = (ca, cb), (sa, sb), (va, vb), (ea, eb)
    objs = [Obj(s, idx, idx) for s in (0, 1) for idx in candidates(clouds[s], states[s], evidence[s], cfg)]
    device = pick_device(cfg.device)
    images = (recon.keys[0].images, recon.keys[1].images)
    segment(objs, clouds, views, images, device)
    for o in objs:
        lift(o, clouds[o.side], views[o.side])
    objs = merge(objs)
    for o in objs:
        verdict(o, clouds[o.side], states[o.side], cfg)
    print(
        f"objects: {len(objs)} from free-space candidates, segmented with SAM 2.1 ({time.time() - t:.1f} s)"
    )

    t = time.time()
    live = [o for o in objs if o.status != "dismissed"]
    embed(live, images, device)
    sides = [o for o in live if o.side == 0], [o for o in live if o.side == 1]
    pairs = match(*sides, (ca.xyz, cb.xyz), cfg)
    paired = {id(o) for p in pairs for o in p[:2]}
    lonely = [o for o in live if o.status == "confirmed" and id(o) not in paired]
    taken = [np.zeros(len(c.xyz), bool) for c in clouds]
    for o in live:
        taken[o.side][o.idx] = True
    found = find_other_ends(lonely, clouds, states, views, images, taken, cfg, device)
    changes, dismissed = assemble(objs, pairs + found, cfg)
    print(
        f"changes: {len(changes)}, {len(pairs)} matched as moves, {len(found)} more found in the other "
        f"video ({time.time() - t:.1f} s)"
    )
    return Result((before, after), recon, frame, views, clouds, states, changes, dismissed)


def write_outputs(r: Result, out: Path, cfg: Config) -> dict:
    t = time.time()
    out.mkdir(parents=True, exist_ok=True)
    ca, cb = r.clouds
    images = (r.recon.keys[0].images, r.recon.keys[1].images)
    times = (r.recon.keys[0].times, r.recon.keys[1].times)
    name_changes(r.changes, r.clouds, r.states, r.views, images, times, r.frame.floor_found, cfg)
    print(f"labels: SigLIP 2 names and sentences for {len(r.changes)} changes ({time.time() - t:.1f} s)")
    t = time.time()
    ids = [np.full(len(ca.xyz), -1, np.int16), np.full(len(cb.xyz), -1, np.int16)]
    for c in r.changes:
        for side, o in ((0, c.before), (1, c.after)):
            if o is not None:
                ids[side][o.idx] = c.id
    changes = [change_json(c, ca, cb) for c in r.changes]
    result = {
        "before": r.paths[0].name,
        "after": r.paths[1].name,
        "units": "metres, z up, floor at z = 0",
        "camera_height_m": round(r.frame.camera_height, 2),
        "counts": {k: sum(c.kind == k for c in r.changes) for k in ("moved", "added", "removed")}
        | {"unverified": sum(c.status == "unverified" for c in r.changes)},
        "changes": changes,
        "coverage": coverage(r.states),
        "ledger": ledger(r),
    }
    objects = {
        c.id: tuple(None if o is None else r.clouds[s].xyz[o.idx] for s, o in ((0, c.before), (1, c.after)))
        for c in r.changes
    }
    write_cards(out, changes, objects, r.views, images)
    (out / "changes.json").write_text(json.dumps(result, indent=1))
    write_ply(out / "scene.ply", ca, cb, ids)
    cameras = (r.views[0].c2w, r.views[1].c2w)
    write_floorplan(out / "floorplan.png", ca, cb, r.states, result, cameras)
    write_report(out / "report.md", result)
    write_viewer(out / "viewer.html", ca, cb, ids, r.states, result, out / "cards", cameras)
    print(f"outputs: {len(changes)} changes written to {out} ({time.time() - t:.1f} s)")
    return result


def coverage(states: tuple[np.ndarray, np.ndarray]) -> dict:
    """Shares of all surface points: checked by the other video (seen or free), hidden from it, or
    seen by one video only. Hidden points carry no information either way."""
    total = sum(len(s) for s in states)
    share = lambda keep: round(float(sum(int(keep(s).sum()) for s in states)) / total, 3)  # noqa: E731
    out = {"checked": share(lambda s: np.isin(s, (SEEN, FREE))), "hidden": share(lambda s: s == OCCLUDED)}
    return out | {
        k: round(float((s == UNREACHED).sum()) / total, 3)
        for k, s in zip(("only_before", "only_after"), states)
    }


def ledger(r: Result) -> dict:
    """Where the evidence of absence went: every free point is counted once, in this order of buckets."""
    free = [s == FREE for s in r.states]
    total = max(sum(int(f.sum()) for f in free), 1)
    claimed = [np.zeros(len(f), bool) for f in free]
    buckets = {
        "confirmed": [o for c in r.changes if c.status == "confirmed" for o in (c.before, c.after) if o],
        "unverified": [o for c in r.changes if c.status == "unverified" for o in (c.before, c.after) if o],
        "dismissed": r.dismissed,
    }
    out = {}
    for name, objs in buckets.items():
        hit = [np.zeros(len(f), bool) for f in free]
        for o in objs:
            hit[o.side][o.idx] = True
        out[name] = round(sum(int((h & f & ~c).sum()) for h, f, c in zip(hit, free, claimed)) / total, 3)
        claimed = [c | h for c, h in zip(claimed, hit)]
    return out | {"residual": round(1 - sum(out.values()), 3)}
