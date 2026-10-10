import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .changes import Change, assemble, candidates
from .config import Config, pick_device
from .evidence import STATE_NAMES, count_evidence, frame_conflicts, point_states, self_consistent
from .geometry import Cloud, Views, cloud_from_views, subset
from .moves import match
from .objects import Obj, describe, lift, merge, segment, verdict
from .outputs import change_json, write_ply, write_viewer
from .reconstruct import Recon, reconstruct, to_views
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
        verdict(o, clouds[o.side], evidence[o.side], cfg)
    print(
        f"objects: {len(objs)} from free-space candidates, segmented with SAM 2.1 ({time.time() - t:.1f} s)"
    )

    t = time.time()
    live = [o for o in objs if o.status != "dismissed"]
    describe(live, images, device)
    sides = [o for o in live if o.side == 0], [o for o in live if o.side == 1]
    pairs = match(*sides, (ca.xyz, cb.xyz), cfg)
    changes, dismissed = assemble(objs, pairs, *sides, cfg)
    print(f"changes: {len(changes)}, {len(pairs)} matched as moves ({time.time() - t:.1f} s)")
    return Result((before, after), recon, frame, views, clouds, states, changes, dismissed)


def write_outputs(r: Result, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (ca, cb), (sa, sb) = r.clouds, r.states
    ids = [np.full(len(ca.xyz), -1, np.int16), np.full(len(cb.xyz), -1, np.int16)]
    for c in r.changes:
        for side, o in ((0, c.before), (1, c.after)):
            if o is not None:
                ids[side][o.idx] = c.id
    shares = {k: np.bincount(s, minlength=4) / len(s) for k, s in (("before", sa), ("after", sb))}
    result = {
        "before": r.paths[0].name,
        "after": r.paths[1].name,
        "camera_height_m": round(r.frame.camera_height, 2),
        "point_states": {k: dict(zip(STATE_NAMES, v.round(3).tolist())) for k, v in shares.items()},
        "changes": [change_json(c, ca, cb) for c in r.changes],
    }
    (out / "changes.json").write_text(json.dumps(result, indent=1))
    write_ply(out / "scene.ply", ca, cb, ids)
    summary = f"{len(r.changes)} changes between {r.paths[0].name} and {r.paths[1].name}"
    write_viewer(out / "viewer.html", ca, cb, {"summary": summary, "changes": result["changes"]})
    return result
