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

FRAME = (
    "One frame for both videos, new for each run: metres, right-handed, z up, x along one of the main wall "
    "directions, origin below the middle of the scene,"
)
NO_FLOOR = (
    " z = 0 at the lowest large level surface, or at the lowest 1 % of points if there is none (no plausible"
    " floor found)."
)


@dataclass
class Result:
    paths: tuple[Path, Path]
    recon: Recon
    frame: Frame
    views: tuple[Views, Views]
    clouds: tuple[Cloud, Cloud]
    states: tuple[np.ndarray, np.ndarray]  # (N,) evidence state of each point against the other video
    changes: list[Change]
    warnings: list[str]


def analyse(before: Path, after: Path, cfg: Config, required=((), ())) -> Result:
    warnings = []

    def warn(text: str) -> None:
        print(f"warning: {text}")
        warnings.append(text)

    recon = reconstruct(before, after, cfg, required)
    if len({h > w for h, w in (k.source_hw for k in recon.keys)}) > 1:
        warn("one video is portrait and the other landscape; both were cropped to one shape")
    if max(recon.fit_px) > 2:
        warn("the predicted rays fit a pinhole camera poorly (a wide-angle lens?)")
    t = time.time()
    va, vb = to_views(recon, cfg)
    frame = world_frame(va, vb)
    va, vb = transform(va, frame.T), transform(vb, frame.T)
    ground = "the floor" if frame.floor_found else "z = 0"
    print(f"world frame: camera {frame.camera_height:.2f} m above {ground} ({time.time() - t:.1f} s)")
    if not frame.up_from_cameras:
        warn("both videos look in almost one direction; gravity comes from surface normals")
    if not frame.floor_found:
        warn("no plausible floor plane: heights count from z = 0 (see frame); nothing is put on the floor")

    t = time.time()
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
    changes = assemble(objs, pairs + found, cfg)
    print(
        f"changes: {len(changes)}, {len(pairs)} matched as moves, {len(found)} more found in the other "
        f"video ({time.time() - t:.1f} s)"
    )
    return Result((before, after), recon, frame, views, clouds, states, changes, warnings)


def write_outputs(r: Result, out: Path, cfg: Config) -> dict:
    t = time.time()
    out.mkdir(parents=True, exist_ok=True)
    ca, cb = r.clouds
    images = (r.recon.keys[0].images, r.recon.keys[1].images)
    ids = [np.full(len(ca.xyz), -1, np.int16), np.full(len(cb.xyz), -1, np.int16)]  # change id per point
    for c in r.changes:
        for side, o in ((0, c.before), (1, c.after)):
            if o is not None:
                ids[side][o.idx] = c.id
    taken = [i >= 0 for i in ids]
    name_changes(r.changes, r.clouds, r.states, taken, r.views, images, r.frame.floor_found, cfg)
    print(f"labels: SigLIP 2 names and sentences for {len(r.changes)} changes ({time.time() - t:.1f} s)")
    t = time.time()
    changes = [change_json(c, ca, cb) for c in r.changes]
    result = {
        "before": r.paths[0].name,
        "after": r.paths[1].name,
        "frame": FRAME + (" floor at z = 0." if r.frame.floor_found else NO_FLOOR),
        "camera_height_m": round(r.frame.camera_height, 2),
        "floor_found": r.frame.floor_found,
        "warnings": r.warnings,
        "counts": {k: sum(c.kind == k for c in r.changes) for k in ("moved", "added", "removed")}
        | {"unverified": sum(c.status == "unverified" for c in r.changes)},
        "changes": changes,
        "coverage": coverage(r.states),
    }
    objects = {
        c.id: tuple(None if o is None else r.clouds[s].xyz[o.idx] for s, o in ((0, c.before), (1, c.after)))
        for c in r.changes
    }
    write_cards(out, changes, objects, r.views, images)
    (out / "changes.json").write_text(json.dumps(result, indent=1))
    write_ply(out / "scene.ply", ca, cb, ids, r.states)
    cameras = (r.views[0].c2w, r.views[1].c2w)
    write_floorplan(out / "floorplan.png", ca, cb, r.states, result, cameras)
    write_report(out / "report.md", result)
    write_viewer(out / "viewer.html", ca, cb, ids, r.states, result, out / "cards", cameras)
    print(f"outputs: {len(changes)} changes written to {out} ({time.time() - t:.1f} s)")
    return result


def coverage(states: tuple[np.ndarray, np.ndarray]) -> dict:
    """Shares of all points the other video checked (seen or free), had hidden, or saw in too few frames."""
    s = np.concatenate(states)
    masks = {"checked": np.isin(s, (SEEN, FREE)), "hidden": s == OCCLUDED, "unchecked": s == UNREACHED}
    return {k: round(float(m.mean()), 3) for k, m in masks.items()}
