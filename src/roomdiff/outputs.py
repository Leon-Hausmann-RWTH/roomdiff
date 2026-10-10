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


def grounded_box(xyz: np.ndarray, support: dict | None) -> dict:
    # Free evidence stops a few centimetres above the support (the contact zone tests as seen), so a
    # box fitted to the object's points floats; extend it down to what it stands on.
    b = fit_box(xyz)
    bottom, top = b.center[2] - b.size[2] / 2, b.center[2] + b.size[2] / 2
    if support and support["height_m"] is not None and 0 < bottom - support["height_m"] < 0.15:
        bottom = support["height_m"]
    b.center[2], b.size[2] = (bottom + top) / 2, top - bottom
    return box_json(b)


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
        # Of the box where the object is now, or where it was if it is gone.
        "height_m": round((box_after or box_before)["size"][2], 2),
        "floor_blocked_m2": c.facts.get("floor_blocked_m2"),
        "check_from": c.facts.get("check_from"),
        "evidence": c.lead.shares,
        "cards": {},
    }


def write_ply(path: Path, before: Cloud, after: Cloud, change_ids: tuple, states: tuple) -> None:
    xyz = np.concatenate([before.xyz, after.xyz]).astype("<f4")
    rgb = np.concatenate([before.rgb, after.rgb]).astype("u1")
    video = np.concatenate([np.zeros(len(before.xyz)), np.ones(len(after.xyz))]).astype("u1")
    change = np.concatenate(change_ids).astype("<i2")
    state = np.concatenate(states).astype("u1")
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
            ("state", "u1"),
        ],
    )
    for i, k in enumerate("xyz"):
        rec[k] = xyz[:, i]
    for i, k in enumerate(("red", "green", "blue")):
        rec[k] = rgb[:, i]
    rec["video"], rec["change"], rec["state"] = video, change, state
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
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())
