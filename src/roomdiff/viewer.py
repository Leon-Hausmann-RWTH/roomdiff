import base64
import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from .geometry import Cloud


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def write_viewer(
    path: Path,
    before: Cloud,
    after: Cloud,
    change_ids: tuple[np.ndarray, np.ndarray],
    result: dict,
    cards_dir: Path,
    cameras: tuple[np.ndarray, np.ndarray],
    max_points: int = 200_000,
) -> None:
    rng = np.random.default_rng(0)
    clouds = {}
    for name, c, ids in (("before", before, change_ids[0]), ("after", after, change_ids[1])):
        # Thin out only the unchanged points; object points are few and carry the result.
        static = rng.permutation(np.nonzero(ids < 0)[0])[:max_points]
        keep = np.sort(np.concatenate([static, np.nonzero(ids >= 0)[0]]))
        clouds[name] = {
            "xyz": b64(c.xyz[keep].astype("<f4")),
            "rgb": b64(c.rgb[keep].astype("u1")),
            "cid": b64(ids[keep].astype("<i2")),  # id in result["changes"], -1 unchanged
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
