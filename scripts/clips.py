from pathlib import Path

import av
import numpy as np


def reencode(src: Path, dst: Path, part: tuple[float, float] = (0.0, 1.0)) -> Path:
    """Upright H.264 copy of a video or of its fraction `part`, long side at most 1280 px, no audio."""
    with av.open(str(src)) as container:
        n = sum(1 for _ in container.decode(video=0))
    first, last = int(part[0] * n), int(part[1] * n)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first, so src == dst works.
    tmp = dst.with_name(dst.stem + ".part.mp4")
    with av.open(str(src)) as container, av.open(str(tmp), "w") as out:
        stream = container.streams.video[0]
        enc = None
        for i, f in enumerate(container.decode(stream)):
            if not first <= i < last:
                continue
            s = min(1.0, 1280 / max(f.width, f.height))
            img = f.to_ndarray(
                width=2 * round(f.width * s / 2), height=2 * round(f.height * s / 2), format="rgb24"
            )
            img = np.ascontiguousarray(np.rot90(img, k=round(f.rotation / 90)))
            if enc is None:
                enc = out.add_stream("libx264", rate=stream.average_rate)
                enc.height, enc.width = img.shape[:2]
                enc.pix_fmt, enc.options = "yuv420p", {"crf": "23", "preset": "slow"}
            out.mux(enc.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
        out.mux(enc.encode())
    tmp.replace(dst)
    return dst
