import base64
import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from .geometry import Cloud


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def inside(xyz: np.ndarray, box: dict, margin: float) -> np.ndarray:
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    local = (xyz - box["center"]) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # (M, 3) in the box's axes
    return (np.abs(local) <= np.asarray(box["size"]) / 2 + margin).all(1)


def write_viewer(
    path: Path,
    before: Cloud,
    after: Cloud,
    change_ids: tuple[np.ndarray, np.ndarray],
    states: tuple[np.ndarray, np.ndarray],
    result: dict,
    cards_dir: Path,
    cameras: tuple[np.ndarray, np.ndarray],
    max_points: int = 200_000,
) -> None:
    rng = np.random.default_rng(0)
    # An object that left hid a surface the before video never saw. The after video saw it, so those after
    # points (-2) fill the hole as the edits apply.
    ids = [change_ids[0], change_ids[1].copy()]
    for c in result["changes"]:
        if c["box_before"] is not None:
            ids[1][inside(after.xyz, c["box_before"], 0.05) & (ids[1] == -1)] = -2
    clouds = {}
    for name, c, cid, state in (("before", before, ids[0], states[0]), ("after", after, ids[1], states[1])):
        # Thin out only the unchanged points; object and revealed points are few and carry the result.
        static = rng.permutation(np.nonzero(cid == -1)[0])[:max_points]
        keep = np.sort(np.concatenate([static, np.nonzero(cid != -1)[0]]))
        clouds[name] = {
            "xyz": b64(c.xyz[keep].astype("<f4")),
            "rgb": b64(c.rgb[keep].astype("u1")),
            "cid": b64(cid[keep].astype("<i2")),  # id in result["changes"]; -1 unchanged, -2 revealed
            "st": b64(state[keep].astype("u1")),  # evidence state against the other video, 3 = never reached
        }
    cards = {
        c["id"]: {
            side: "data:image/jpeg;base64,"
            + base64.b64encode((cards_dir / Path(rel).name).read_bytes()).decode()
            for side, rel in c["cards"].items()
            if rel
        }
        for c in result["changes"]
    }
    allxyz = np.concatenate([before.xyz, after.xyz])
    lo, hi = np.percentile(allxyz, 2, 0), np.percentile(allxyz, 98, 0)
    payload = {
        "result": result,
        "clouds": clouds,
        "cards": cards,
        "cameras": {k: c2w[:, :3, 3].round(3).tolist() for k, c2w in zip(("before", "after"), cameras)},
        "center": ((lo + hi) / 2).tolist(),
        "radius": float(np.linalg.norm(hi - lo)),
    }
    assets = files("roomdiff") / "assets"
    html = (assets / "viewer.html").read_text()
    html = html.replace("/*THREE*/", (assets / "three.min.js").read_text())
    html = html.replace("/*ORBIT*/", (assets / "OrbitControls.js").read_text())
    # A "</script>" inside a label or sentence would end the script block early.
    html = html.replace("/*DATA*/", json.dumps(payload).replace("</", "<\\/"))
    path.write_text(html)
