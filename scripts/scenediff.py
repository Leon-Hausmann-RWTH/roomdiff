"""SceneDiff benchmark helpers (huggingface.co/datasets/yuqun/SceneDiff, MIT).

Ground-truth masks are keyed by frame index j of the video resampled to 30 fps (ffmpeg -r 30),
and the official evaluator scores j = 30 k, one frame per second.
"""

import io
import pickle
from pathlib import Path

import numpy as np

DATA = Path("data/scenediff_benchmark/data")
SAFE = (
    {("builtins", n) for n in ("dict", "list", "tuple", "set", "str", "int", "float", "bool", "bytes")}
    | {
        (m, n)
        for m in ("numpy.core.multiarray", "numpy._core.multiarray")
        for n in ("_reconstruct", "scalar")
    }
    | {("numpy", "ndarray"), ("numpy", "dtype")}
)


class SafeUnpickler(pickle.Unpickler):
    # The pickles come from the internet; refuse anything but containers and numpy arrays.
    def find_class(self, module, name):
        if (module, name) not in SAFE:
            raise pickle.UnpicklingError(f"refusing {module}.{name}")
        return super().find_class(module, name)


def load_gt(pair: str) -> dict:
    return SafeUnpickler(io.BytesIO((DATA / pair / "segments.pkl").read_bytes())).load()


def videos(pair: str) -> tuple[Path, Path]:
    found = sorted(p for p in (DATA / pair).iterdir() if p.name.startswith("original_video"))
    return found[0], found[1]


def object_type(obj: dict) -> str:
    return "moved" if obj["in_video1"] and obj["in_video2"] else "removed" if obj["in_video1"] else "added"


def rle_decode(rle: dict) -> np.ndarray:
    """COCO compressed RLE to a boolean (H, W) mask (port of pycocotools' rleFrString)."""
    s, counts, p = rle["counts"], [], 0
    s = s.decode() if isinstance(s, bytes) else s
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p, k = p + 1, k + 1
            if not more and c & 0x10:
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    h, w = rle["size"]
    flat = np.zeros(h * w, bool)
    ends = np.cumsum(counts)
    for start, stop in zip(ends[0::2], ends[1::2]):
        flat[start:stop] = True
    return flat.reshape(w, h).T


def rle_encode(mask: np.ndarray) -> dict:
    """Boolean (H, W) mask to COCO compressed RLE (port of pycocotools' rleToString)."""
    flat = np.asarray(mask, bool).ravel(order="F")
    bounds = np.concatenate([[0], np.flatnonzero(flat[1:] != flat[:-1]) + 1, [flat.size]])
    runs = ([0] if flat[0] else []) + np.diff(bounds).tolist()
    out = []
    for i, x in enumerate(runs):
        x -= runs[i - 2] if i > 2 else 0
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = x != -1 if c & 0x10 else x != 0
            out.append(chr(c + 48 + (0x20 if more else 0)))
    return {"size": list(mask.shape), "counts": "".join(out)}
