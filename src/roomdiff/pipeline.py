import json
import time
from pathlib import Path

import numpy as np

from .changes import Change, candidates
from .config import Config
from .evidence import STATE_NAMES, count_evidence, point_states, self_consistent
from .geometry import cloud_from_views, subset
from .outputs import change_json, write_ply, write_viewer
from .reconstruct import reconstruct, to_views
from .world import transform, world_frame


def run(before: Path, after: Path, out: Path, cfg: Config, required=((), ())) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    recon = reconstruct(before, after, cfg, required)
    t = time.time()
    va, vb = to_views(recon, cfg)
    frame = world_frame(va, vb)
    va, vb = transform(va, frame.T), transform(vb, frame.T)
    print(f"world frame: camera {frame.camera_height:.2f} m above the floor ({time.time() - t:.1f} s)")
    if not frame.floor_found or not 0.5 < frame.camera_height < 2.2:
        print("warning: no plausible floor plane; heights and 'on the floor' statements may be wrong")

    t = time.time()
    clouds = []
    for own in (va, vb):
        c = cloud_from_views(own, cfg.voxel)
        clouds.append(subset(c, self_consistent(c.xyz, c.depth, own, cfg)))
    ca, cb = clouds
    sa = point_states(count_evidence(ca.xyz, ca.depth, vb, cfg), cfg)
    sb = point_states(count_evidence(cb.xyz, cb.depth, va, cfg), cfg)
    share = {k: np.round(np.bincount(s, minlength=4) / len(s), 3) for k, s in (("before", sa), ("after", sb))}
    print(f"ray evidence: {len(ca.xyz)} + {len(cb.xyz)} points ({time.time() - t:.1f} s)")

    t = time.time()
    changes = []
    for idx in candidates(ca.xyz, sa, cfg):
        changes.append(Change(len(changes), "removed", "confirmed", "seen through after", before=idx))
    for idx in candidates(cb.xyz, sb, cfg):
        changes.append(Change(len(changes), "added", "confirmed", "seen through before", after=idx))
    print(f"changes: {len(changes)} candidates ({time.time() - t:.1f} s)")

    ids = [np.full(len(ca.xyz), -1, np.int16), np.full(len(cb.xyz), -1, np.int16)]
    for c in changes:
        if c.before is not None:
            ids[0][c.before] = c.id
        if c.after is not None:
            ids[1][c.after] = c.id
    result = {
        "before": str(before),
        "after": str(after),
        "camera_height_m": round(frame.camera_height, 2),
        "point_states": {k: dict(zip(STATE_NAMES, v.tolist())) for k, v in share.items()},
        "changes": [change_json(c, ca, cb) for c in changes],
    }
    (out / "changes.json").write_text(json.dumps(result, indent=1))
    write_ply(out / "scene.ply", ca, cb, ids)
    summary = f"{len(changes)} changes between {before.name} and {after.name}"
    write_viewer(out / "viewer.html", ca, cb, {"summary": summary, "changes": result["changes"]})
    return result
