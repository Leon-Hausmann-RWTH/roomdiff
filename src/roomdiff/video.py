from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image
from scipy.ndimage import laplace


@dataclass
class Keyframes:
    images: np.ndarray  # (N, H, W, 3) uint8 at twice the reconstruction size, for crops and masks
    times: np.ndarray  # (N,) seconds from the start of the video
    frame_ids: np.ndarray  # (N,) decoded frame index
    crop: tuple[int, int, int, int]  # (x0, y0, x1, y1) kept from the upright source frame, source pixels
    source_hw: tuple[int, int]  # upright source frame size


def upright_frames(path: Path, long_side: int, keep=None):
    """Yield (time, upright RGB frame scaled to long_side) without holding the video in memory."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        t0 = None
        for i, frame in enumerate(container.decode(stream)):
            t0 = frame.time if t0 is None else t0
            if keep is not None and i not in keep:
                continue
            s = long_side / max(frame.width, frame.height)
            img = frame.to_ndarray(
                width=round(frame.width * s), height=round(frame.height * s), format="rgb24"
            )
            # Apply the display rotation; iPhones store portrait video as landscape.
            yield frame.time - t0, np.ascontiguousarray(np.rot90(img, k=round(frame.rotation / 90)))


def probe(path: Path) -> tuple[float, tuple[int, int], float]:
    """Aspect ratio (W/H) and size of the upright frame, and the duration in seconds."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
        hw = (
            (frame.height, frame.width)
            if round(frame.rotation / 90) % 2 == 0
            else (frame.width, frame.height)
        )
        duration = (
            float(stream.duration * stream.time_base) if stream.duration else float(container.duration / 1e6)
        )
    return hw[1] / hw[0], hw, duration


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize(size, Image.Resampling.BOX))


def recon_size(aspect: float, pixels: int) -> tuple[int, int]:
    """Largest (W, H) in multiples of 14 (the ViT patch) with W/H close to aspect and W*H <= pixels."""
    h = np.sqrt(pixels / aspect)
    w, h = int(round(aspect * h / 14)), int(round(h / 14))
    while w * h * 196 > pixels:
        if w / h > aspect:
            w -= 1
        else:
            h -= 1
    return w * 14, h * 14


def select(path: Path, n: int, size: tuple[int, int], required_times=()) -> Keyframes:
    """Keyframes: the frame nearest each required time, then the sharpest of n equal windows that lack one."""
    times, sharp = [], []
    for t, img in upright_frames(path, 320):
        times.append(t)
        sharp.append(laplace(img.mean(-1)).var())
    times, sharp = np.array(times), np.array(sharp)
    picked = {int(np.abs(times - t).argmin()) for t in required_times}
    for win in np.array_split(np.arange(len(times)), n):
        if not picked.intersection(win.tolist()):
            picked.add(int(win[np.argmax(sharp[win])]))
    keep = sorted(picked)
    w, h = size
    source_hw = probe(path)[1]
    frames = [img for _, img in upright_frames(path, 2 * max(w, h) * 1.25, set(keep))]
    fh, fw = frames[0].shape[:2]
    # Centre-crop to the reconstruction aspect ratio.
    cw, ch = (round(fh * w / h), fh) if fw / fh > w / h else (fw, round(fw * h / w))
    x0, y0 = (fw - cw) // 2, (fh - ch) // 2
    images = np.stack(
        [resize(np.ascontiguousarray(f[y0 : y0 + ch, x0 : x0 + cw]), (2 * w, 2 * h)) for f in frames]
    )
    s = source_hw[0] / fh
    crop = (round(x0 * s), round(y0 * s), round((x0 + cw) * s), round((y0 + ch) * s))
    return Keyframes(images, times[keep], np.array(keep), crop, source_hw)
