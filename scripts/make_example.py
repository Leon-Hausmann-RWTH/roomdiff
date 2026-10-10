"""Rebuild the committed example and the README media from two videos.

Writes examples/<name>/{before.mp4, after.mp4} (small upright copies), examples/<name>/output/ (the
normal run on those copies), docs/index.html (the viewer, for GitHub Pages) and docs/media/.

    uv run python scripts/make_example.py before.mov after.mov --name living_room
"""

import io
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from clips import reencode
from PIL import Image
from playwright.sync_api import sync_playwright

from roomdiff.config import Config
from roomdiff.pipeline import analyse, write_outputs

GL = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
SET_SLIDER = (
    "t => { const e = document.getElementById('edit'); e.value = t; e.dispatchEvent(new Event('input')); }"
)


@dataclass
class Args:
    before: tyro.conf.Positional[Path]
    after: tyro.conf.Positional[Path]
    name: str = "living_room"


def slider_gif(viewer: Path, path: Path, frames: int = 48) -> None:
    """The edit slider played on the before scene while the camera turns slowly, as an animated GIF."""
    shots = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=GL)
        page = browser.new_page(viewport={"width": 1180, "height": 700})
        page.goto(viewer.resolve().as_uri())
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="Before", exact=True).click()
        page.evaluate("controls.autoRotate = true; controls.autoRotateSpeed = 0.6")
        for k in range(frames):
            t = min(1.0, max(0.0, (k - 6) / (frames - 18)))  # hold the before scene, sweep, hold the result
            page.evaluate(SET_SLIDER, t)
            page.wait_for_timeout(120)
            png = page.locator("canvas").screenshot()
            shots.append(Image.open(io.BytesIO(png)).convert("RGB"))
        browser.close()
    w = 640
    small = [s.resize((w, round(s.height * w / s.width)), Image.Resampling.LANCZOS) for s in shots]
    palette = small[len(small) // 2].quantize(colors=200)
    frames_p = [s.quantize(palette=palette, dither=Image.Dither.NONE) for s in small]
    frames_p[0].save(path, save_all=True, append_images=frames_p[1:], duration=110, loop=0, optimize=True)
    print(f"{path}: {path.stat().st_size / 1e6:.1f} MB")


def card_strip(output: Path, path: Path, n: int = 3) -> None:
    """Before and after cards of the first changes side by side: the evidence at a glance."""
    rows = []
    for card in sorted((output / "cards").glob("*_before.jpg"), key=lambda p: int(p.name.split("_")[0]))[:n]:
        pair = [Image.open(card), Image.open(card.with_name(card.name.replace("before", "after")))]
        h = 300
        pair = [im.resize((round(im.width * h / im.height), h)) for im in pair]
        rows.append(np.concatenate([np.asarray(im) for im in pair], 1))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 8), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    Image.fromarray(np.concatenate(rows)).save(path, quality=85)


def main(args: Args) -> None:
    root = Path("examples") / args.name
    before = reencode(args.before, root / "before.mp4")
    after = reencode(args.after, root / "after.mp4")
    output = root / "output"
    write_outputs(analyse(before, after, Config()), output, Config())
    media = Path("docs/media")
    media.mkdir(parents=True, exist_ok=True)
    shutil.copy(output / "viewer.html", "docs/index.html")
    slider_gif(output / "viewer.html", media / "edit_slider.gif")
    Image.open(output / "floorplan.png").save(media / "floorplan.png")
    card_strip(output, media / "cards.jpg")


if __name__ == "__main__":
    main(tyro.cli(Args))
