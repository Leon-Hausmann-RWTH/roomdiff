import base64
import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from .changes import Box, Change, fit_box
from .geometry import Cloud


def box_json(b: Box | None) -> dict | None:
    if b is None:
        return None
    return {
        "center": b.center.round(3).tolist(),
        "size": b.size.round(3).tolist(),
        "yaw": round(float(b.yaw), 3),
    }


def change_json(c: Change, before: Cloud, after: Cloud) -> dict:
    m = c.motion
    obj = c.before if c.before is not None else c.after
    return {
        "id": c.id,
        "type": c.kind,
        "status": c.status,
        "reason": c.reason,
        "label": c.label,
        "sentence": c.sentence,
        "box_before": box_json(fit_box(before.xyz[c.before.idx])) if c.before is not None else None,
        "box_after": box_json(fit_box(after.xyz[c.after.idx])) if c.after is not None else None,
        "motion": None
        if m is None
        else {
            "distance_m": round(m.distance, 2),
            "turn_deg": None if m.turn is None else round(m.turn),
            "fit": round(m.fit, 2),
        },
        "transform": None if m is None else {"yaw": round(m.yaw, 4), "t": np.round(m.t, 4).tolist()},
        "evidence": obj.shares,
        **c.facts,
    }


def write_ply(path: Path, before: Cloud, after: Cloud, change_ids: tuple[np.ndarray, np.ndarray]) -> None:
    xyz = np.concatenate([before.xyz, after.xyz]).astype("<f4")
    rgb = np.concatenate([before.rgb, after.rgb]).astype("u1")
    video = np.concatenate([np.zeros(len(before.xyz)), np.ones(len(after.xyz))]).astype("u1")
    change = np.concatenate(change_ids).astype("<i2")
    rec = np.empty(
        len(xyz),
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("video", "u1"),
            ("change", "<i2"),
        ],
    )
    for i, k in enumerate("xyz"):
        rec[k] = xyz[:, i]
    for i, k in enumerate(("red", "green", "blue")):
        rec[k] = rgb[:, i]
    rec["video"], rec["change"] = video, change
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment video: 0 before, 1 after; change: id in changes.json, -1 unchanged\n"
        f"element vertex {len(rec)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar video\nproperty short change\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def write_viewer(path: Path, before: Cloud, after: Cloud, data: dict, max_points: int = 200_000) -> None:
    rng = np.random.default_rng(0)
    parts = {}
    for name, c in (("before", before), ("after", after)):
        keep = np.sort(rng.permutation(len(c.xyz))[:max_points])
        parts[name] = {"xyz": b64(c.xyz[keep].astype("<f4")), "rgb": b64(c.rgb[keep].astype("u1"))}
    allxyz = np.concatenate([before.xyz, after.xyz])
    lo, hi = np.percentile(allxyz, 2, 0), np.percentile(allxyz, 98, 0)
    payload = {
        **data,
        "points": parts,
        "view_center": ((lo + hi) / 2).tolist(),
        "view_radius": float(np.linalg.norm(hi - lo)),
    }
    assets = files("roomdiff") / "assets"
    html = (assets / "viewer.html").read_text()
    html = html.replace("/*THREE*/", (assets / "three.min.js").read_text())
    html = html.replace("/*ORBIT*/", (assets / "OrbitControls.js").read_text())
    html = html.replace("/*DATA*/", json.dumps(payload))
    path.write_text(html)
