"""Turn a video pair into a committed example: examples/<name>/, docs/<name>.html and the README media.

Needs Playwright's browser once: uv run playwright install chromium.

    uv run python scripts/make_example.py before.MOV after.MOV --name my_room
"""

import io
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from clips import copy_video
from evaluate import naive
from PIL import Image, ImageColor, ImageDraw
from playwright.sync_api import sync_playwright

from roomdiff.cli import check_video
from roomdiff.config import Config
from roomdiff.evidence import FREE, OCCLUDED, SEEN, UNREACHED
from roomdiff.pipeline import Result, analyse, write_outputs
from roomdiff.report import COLORS, corners, font, kind, place_tag, ring

# Software WebGL for headless Chromium.
GL = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
# Aim at the middle of all change boxes from the default direction, close enough to show them all.
FRAME_CHANGES = """() => {
  const ps = R.changes.flatMap(c => [c.box_before, c.box_after]).filter(Boolean)
    .map(b => new THREE.Vector3(...b.center));
  const mid = ps.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(ps.length);
  const r = Math.max(1, ...ps.map(p => p.distanceTo(mid)));
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(mid);
  camera.position.copy(mid).addScaledVector(dir, 2.6 * r);
}"""
SET_SLIDER = (
    "t => { const e = document.getElementById('edit'); e.value = t; e.dispatchEvent(new Event('input')); }"
)


@dataclass
class Args:
    before: tyro.conf.Positional[Path]
    after: tyro.conf.Positional[Path]
    name: str
    """Folder name under examples/."""
    docs: bool = True
    """Make this the main example: docs/index.html and the README media."""
    config: Config = field(default_factory=Config)


def slider_gif(viewer: Path, path: Path, frames: int = 40) -> None:
    """The edit slider played on the before scene while the camera turns slowly, as an animated GIF."""
    shots = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=GL)
        page = browser.new_page(viewport={"width": 1180, "height": 700})
        page.goto(viewer.resolve().as_uri())
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="Before", exact=True).click()
        page.locator("#showUnchecked").uncheck()
        page.evaluate(FRAME_CHANGES)
        page.evaluate("controls.autoRotate = true; controls.autoRotateSpeed = 0.25")
        for k in range(frames):
            t = min(1.0, max(0.0, (k - 6) / (frames - 18)))  # hold the before scene, sweep, hold the result
            page.evaluate(SET_SLIDER, t)
            page.wait_for_timeout(120)
            png = page.locator("canvas").screenshot()
            shots.append(Image.open(io.BytesIO(png)).convert("RGB"))
        browser.close()
    w = 560
    small = [s.resize((w, round(s.height * w / s.width)), Image.Resampling.LANCZOS) for s in shots]
    palette = small[len(small) // 2].quantize(colors=200)
    frames_p = [s.quantize(palette=palette, dither=Image.Dither.NONE) for s in small]
    frames_p[0].save(path, save_all=True, append_images=frames_p[1:], duration=110, loop=0, optimize=True)
    print(f"{path}: {path.stat().st_size / 1e6:.1f} MB")


def card_strip(output: Path, result: dict, path: Path, n: int = 3) -> None:
    """Before and after cards side by side, one row for each of the first n changes that have both."""
    rows = []
    for c in [c for c in result["changes"] if all(c["cards"].values())][:n]:
        pair = [Image.open(output / c["cards"][side]) for side in ("before", "after")]
        pair = [im.resize((round(im.width * 300 / im.height), 300)) for im in pair]
        rows.append(np.concatenate([np.asarray(im) for im in pair], 1))
    if not rows:
        print("no change with both cards; no card strip written")
        return
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 8), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    Image.fromarray(np.concatenate(rows)).save(path, quality=85)


# docs/index.html, the GitHub Pages root, forwards to the main example's viewer.
INDEX = """<!doctype html>
<meta charset="utf-8">
<title>roomdiff</title>
<meta http-equiv="refresh" content="0; url={name}.html">
<p><a href="{name}.html">Open the {name} viewer</a></p>
"""


# Ray-test state, colour and legend text of the evidence figure.
VERDICTS = [
    (FREE, (29, 29, 31), "looked through by the other video"),
    (OCCLUDED, (150, 120, 230), "hidden from the other video"),
    (UNREACHED, (240, 175, 60), "seen by it in fewer than 3 frames"),
    (SEEN, (150, 150, 155), "seen by it as a surface"),
]


def evidence_figure(res: Result, result: dict, cfg: Config, path: Path) -> None:
    """Top view of every region where the two clouds differ by more than 5 cm, coloured by its ray-test
    state, with the changes roomdiff reports drawn on top."""
    regions = naive(res, cfg)
    xyz = np.concatenate([c.xyz for c in res.clouds])
    lo, hi = np.percentile(xyz[:, :2], 0.5, 0) - 0.2, np.percentile(xyz[:, :2], 99.5, 0) + 0.2
    s, m = 900 / (hi - lo).max(), 40  # pixels per metre, margin
    pw, ph = np.ceil((hi - lo) * s).astype(int)
    canvas = np.full((ph + 2 * m, pw + 2 * m + 660, 3), 255, np.uint8)

    def px(p: np.ndarray) -> np.ndarray:
        return np.stack([m + (p[..., 0] - lo[0]) * s, m + (hi[1] - p[..., 1]) * s], -1)

    def dots(p: np.ndarray, color, r: int) -> None:
        q = np.round(px(p)).astype(int)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = (q[:, 0] + dx).clip(0, pw + 2 * m - 1), (q[:, 1] + dy).clip(0, ph + 2 * m - 1)
                canvas[y, x] = color

    dots(xyz[:, :2], (222, 222, 222), 1)
    colors, counts = {code: col for code, col, _ in VERDICTS}, dict.fromkeys(range(4), 0)
    for c in regions:
        o = c.before or c.after
        code = int(np.bincount(res.states[o.side][o.idx], minlength=4).argmax())
        counts[code] += 1
        dots(res.clouds[o.side].xyz[o.idx, :2], colors[code], 2)
    img = Image.fromarray(canvas)
    d, taken = ImageDraw.Draw(img, "RGBA"), []
    for c in result["changes"]:
        col = ImageColor.getrgb(COLORS[kind(c)])
        for box in (b for b in (c["box_before"], c["box_after"]) if b):
            fp = px(corners(box)[[0, 1, 3, 2], :2])
            ring(d, fp, col, 4)
            taken.append((*fp.min(0), *fp.max(0)))
    for c in result["changes"]:
        place_tag(d, px(corners(c["box_after"] or c["box_before"])[:, :2]), taken, c, 34)
    x, y, f = pw + 2 * m + 20, m, font(26)
    d.text((x, y), "Where the two videos disagree", font=font(32), fill="black")
    d.text((x, y + 44), f"{len(regions)} regions have no point of the other video", font=f, fill=(90, 90, 90))
    d.text((x, y + 78), "within 5 cm: naive differencing reports each one", font=f, fill=(90, 90, 90))
    y += 140
    for code, col, text in VERDICTS:
        d.rectangle((x, y, x + 40, y + 30), fill=col)
        d.text((x + 56, y + 15), f"{counts[code]} {text}", font=f, fill="black", anchor="lm")
        y += 48
    ring(d, np.array([[x, y], [x + 40, y], [x + 40, y + 30], [x, y + 30]], float), (60, 60, 60), 4)
    n = len(result["changes"])
    d.text(
        (x + 56, y + 15), f"{n} change{'s' * (n != 1)} roomdiff reports", font=f, fill="black", anchor="lm"
    )
    img.save(path)


def main(args: Args) -> None:
    # The CLI's input checks, on the original files (the copies lose the HDR tag).
    notes = [n for p in (args.before, args.after) for n in check_video(p, args.config)]
    root = Path("examples") / args.name
    before = copy_video(args.before, root / "before.mp4")
    after = copy_video(args.after, root / "after.mp4")
    output = root / "output"
    res = analyse(before, after, args.config)
    res.warnings[:0] = notes
    result = write_outputs(res, output, args.config)
    media = Path("docs/media")
    if args.docs:
        # The GIF first, so a missing browser fails before docs/ changes.
        slider_gif(output / "viewer.html", media / "edit_slider.gif")
        Image.open(output / "floorplan.png").save(media / "floorplan.png")
        evidence_figure(res, result, args.config, media / "evidence_example.png")
        Path("docs/index.html").write_text(INDEX.format(name=args.name))
        card_strip(output, result, media / "cards.jpg")
    shutil.copy(output / "viewer.html", f"docs/{args.name}.html")


if __name__ == "__main__":
    main(tyro.cli(Args))
