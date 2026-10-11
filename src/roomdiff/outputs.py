from pathlib import Path

import numpy as np

from .changes import Change, fit_box
from .geometry import Cloud


def grounded_box(xyz: np.ndarray, support: dict | None) -> dict:
    """Box around the points, extended down to the support if it floats less than 15 cm above it."""
    b = fit_box(xyz)
    bottom, top = b.center[2] - b.size[2] / 2, b.center[2] + b.size[2] / 2
    if support and 0 < bottom - support["height_m"] < 0.15:
        bottom = support["height_m"]
    b.center[2], b.size[2] = (bottom + top) / 2, top - bottom
    return {
        "center": b.center.round(3).tolist(),
        "size": b.size.round(3).tolist(),
        "yaw": round(float(b.yaw), 3),
    }


def change_json(c: Change, before: Cloud, after: Cloud) -> dict:
    m = c.motion
    support = c.facts.get("support", {})
    box_before = grounded_box(before.xyz[c.before.idx], support.get("before")) if c.before else None
    box_after = grounded_box(after.xyz[c.after.idx], support.get("after")) if c.after else None
    return {
        "id": c.id,
        "type": c.kind,
        "status": c.status,
        "reason": c.reason,
        "label": c.label,
        "label_score": c.facts.get("label_score"),
        "title": c.facts.get("title"),
        "sentence": c.sentence,
        "box_before": box_before,
        "box_after": box_after,
        "motion": None
        if m is None
        else {
            "distance_m": round(m.distance, 2),
            "turn_deg": None if m.turn is None else round(m.turn),
            "fit": round(m.fit, 2),
            "in_place": c.facts.get("in_place", False),
        },
        "transform": None if m is None else {"yaw": round(m.yaw, 4), "t": np.round(m.t, 4).tolist()},
        "support": support,
        "floor_blocked_m2": c.facts.get("floor_blocked_m2"),
        "evidence": c.lead.shares,
        "cards": {},
    }


def write_ply(path: Path, before: Cloud, after: Cloud, change_ids: tuple, states: tuple) -> None:
    xyz = np.concatenate([before.xyz, after.xyz]).astype("<f4")
    rgb = np.concatenate([before.rgb, after.rgb]).astype("u1")
    video = np.concatenate([np.zeros(len(before.xyz)), np.ones(len(after.xyz))]).astype("u1")
    change = np.concatenate(change_ids).astype("<i2")
    state = np.concatenate(states).astype("u1")
    rec = np.rec.fromarrays(
        [*xyz.T, *rgb.T, video, change, state], names="x,y,z,red,green,blue,video,change,state"
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment metres, in the frame described by the frame field of changes.json\n"
        "comment video: 0 before, 1 after; change: id in changes.json, -1 unchanged\n"
        "comment state against the other video: 0 surface seen, 1 looked through, 2 hidden, 3 too few views\n"
        f"element vertex {len(rec)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar video\nproperty short change\nproperty uchar state\nend_header\n"
    )
    path.write_bytes(header.encode() + rec.tobytes())
