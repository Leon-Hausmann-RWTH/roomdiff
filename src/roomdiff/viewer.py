import base64
import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from .geometry import Cloud, rz


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def inside(xyz: np.ndarray, box: dict, margin: float) -> np.ndarray:
    local = (xyz - box["center"]) @ rz(box["yaw"])  # (M, 3) in the box's axes
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
) -> None:
    rng = np.random.default_rng(0)
    # After points inside an old box (-2) fill the hole the object leaves in the before cloud.
    ids = [change_ids[0], change_ids[1].copy()]
    for c in result["changes"]:
        if c["box_before"] is not None:
            ids[1][inside(after.xyz, c["box_before"], 0.05) & (ids[1] == -1)] = -2
    clouds = {}
    for name, c, cid, state in zip(("before", "after"), (before, after), ids, states):
        # Keep all changed and revealed points and at most 200,000 unchanged ones.
        static = rng.permutation(np.nonzero(cid == -1)[0])[:200_000]
        keep = np.sort(np.concatenate([static, np.nonzero(cid != -1)[0]]))
        clouds[name] = {
            "xyz": b64(c.xyz[keep].astype("<f4")),
            "rgb": b64(c.rgb[keep].astype("u1")),
            "cid": b64(cid[keep].astype("<i2")),  # change id; -1 unchanged, -2 revealed
            "st": b64(state[keep].astype("u1")),  # evidence state, as in evidence.py
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
        "cameras": {
            k: c2w[:, :3, 3].astype(float).round(3).tolist() for k, c2w in zip(("before", "after"), cameras)
        },
        "center": ((lo + hi) / 2).tolist(),
        "radius": float(np.linalg.norm(hi - lo)),
    }
    assets = files("roomdiff") / "assets"
    html = (assets / "viewer.html").read_text()
    html = html.replace("/*THREE*/", (assets / "three.min.js").read_text())
    html = html.replace("/*ORBIT*/", (assets / "OrbitControls.js").read_text())
    # Escape "</" so no label or sentence can close the script tag.
    html = html.replace("/*DATA*/", json.dumps(payload).replace("</", "<\\/"))
    path.write_text(html)
