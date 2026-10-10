import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .changes import Change, candidates
from .config import Config
from .evidence import STATE_NAMES, count_evidence, point_states, self_consistent
from .geometry import Cloud, Views, cloud_from_views, subset
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
    clouds = []
    for own in (va, vb):
        c = cloud_from_views(own, cfg.voxel)
        clouds.append(subset(c, self_consistent(c.xyz, c.depth, own, cfg)))
    ca, cb = clouds
    sa = point_states(count_evidence(ca.xyz, ca.depth, vb, cfg), cfg)
    sb = point_states(count_evidence(cb.xyz, cb.depth, va, cfg), cfg)
    print(f"ray evidence: {len(ca.xyz)} + {len(cb.xyz)} points ({time.time() - t:.1f} s)")

    t = time.time()
    changes = []
    for idx in candidates(ca.xyz, sa, cfg):
        changes.append(Change(len(changes), "removed", "confirmed", "seen through after", before=idx))
    for idx in candidates(cb.xyz, sb, cfg):
        changes.append(Change(len(changes), "added", "confirmed", "seen through before", after=idx))
    print(f"changes: {len(changes)} candidates ({time.time() - t:.1f} s)")
    return Result((before, after), recon, frame, (va, vb), (ca, cb), (sa, sb), changes)


def write_outputs(r: Result, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (ca, cb), (sa, sb) = r.clouds, r.states
    ids = [np.full(len(ca.xyz), -1, np.int16), np.full(len(cb.xyz), -1, np.int16)]
    for c in r.changes:
        if c.before is not None:
            ids[0][c.before] = c.id
        if c.after is not None:
            ids[1][c.after] = c.id
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
