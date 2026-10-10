from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from .geometry import Cloud, Views, project

COLORS = {"removed": "#D55E00", "added": "#0072B2", "moved": "#CC79A7", "unverified": "#8A8A8A"}
GLYPHS = {"removed": "−", "added": "+", "moved": "→", "unverified": "?"}
VIDEO_RGB = {"before": (0, 158, 115), "after": (230, 159, 0)}
SIDES = ("before", "after")
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
NEAR = 0.05  # metres; box edges are clipped at this camera depth
CARD = (640, 480)
PLAN = (940, 1320)  # largest plan area in pixels, left of the legend
HEIGHT_TOP = 1.5  # metres; the floor plan grey saturates here
GREY = (110, 110, 110)


def kind(change: dict) -> str:
    return "unverified" if change["status"] == "unverified" else change["type"]


def font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow bundles this font, so images look the same everywhere; it lacks minus and arrow, hence glyph().
    return ImageFont.load_default(size)


def corners(box: dict) -> np.ndarray:
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    signs = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)])
    return np.asarray(box["center"]) + (signs * np.asarray(box["size"]) / 2) @ R.T  # (8, 3)


def glyph(d: ImageDraw.ImageDraw, k: str, cx: float, cy: float, s: float) -> None:
    if k == "unverified":
        d.text((cx, cy), "?", font=font(round(s * 1.15)), fill="white", anchor="mm")
        return
    w, r = max(2, round(s / 6)), s * 0.38
    d.line([(cx - r, cy), (cx + (s * 0.2 if k == "moved" else r), cy)], fill="white", width=w)
    if k == "added":
        d.line([(cx, cy - r), (cx, cy + r)], fill="white", width=w)
    if k == "moved":
        d.polygon(
            [(cx + s * 0.46, cy), (cx + s * 0.1, cy - s * 0.3), (cx + s * 0.1, cy + s * 0.3)], fill="white"
        )


def tag(d: ImageDraw.ImageDraw, x: float, y: float, k: str, text: str, h: int) -> tuple:
    f = font(round(h * 0.66))
    w = h + (d.textlength(text, font=f) + h * 0.3 if text else 0)
    d.rounded_rectangle((x, y, x + w, y + h), radius=h * 0.22, fill=COLORS[k])
    glyph(d, k, x + h / 2, y + h / 2, h * 0.62)
    d.text((x + h * 0.95, y + h / 2), text, font=f, fill="white", anchor="lm")
    return (x, y, x + w, y + h)


def ring(
    d: ImageDraw.ImageDraw, pts: np.ndarray, fill, width: int, dash: float = 0, closed: bool = True
) -> None:
    for p, q in zip(pts, np.roll(pts, -1, 0)[: len(pts) - (not closed)]):
        n = max(np.linalg.norm(q - p), 1e-6)
        for a in np.arange(0, n, 2 * dash) if dash else [0]:
            b = min(a + dash, n) if dash else n
            d.line([tuple(p + (q - p) * a / n), tuple(p + (q - p) * b / n)], fill=fill, width=width)


def sharpness(images: np.ndarray) -> np.ndarray:
    g = images[:, ::2, ::2].astype(np.float32).mean(-1)
    lap = 4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
    return lap.var((1, 2))  # (N,) variance of the Laplacian, higher is sharper


def pick_frame(xyz: np.ndarray, box: dict, views: Views, sharp: np.ndarray, present: bool) -> int | None:
    h, w = views.depth.shape[1:]
    good, inside = np.zeros(len(views.depth)), np.zeros(len(views.depth))
    for i in range(len(views.depth)):
        u, v, z = project(xyz, views.K[i], views.c2w[i])
        ok = (z > NEAR) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d, z = views.depth[i][v[ok].astype(int), u[ok].astype(int)], z[ok]
        # A present object must be the visible surface; an empty place must be seen or seen through.
        hit = (np.abs(d - z) < 0.05 + 0.03 * z if present else d > z - 0.05 - 0.03 * z) & (d > 0)
        cu, cv, cz = project(corners(box), views.K[i], views.c2w[i])
        # Frames that cut the box off count less, so the card shows the whole object where it can.
        whole = ((cz > NEAR) & (cu >= 0) & (cu < w) & (cv >= 0) & (cv < h)).mean() ** 2
        good[i], inside[i] = hit.sum() * whole, ok.sum()
    score = good if good.max() > 0 else inside
    if score.max() == 0:
        return None
    # Among frames that show nearly as much of the object as the best one, motion blur decides.
    return int(np.argmax(np.where(score >= 0.8 * score.max(), sharp, -1)))


def card(
    image: np.ndarray, K: np.ndarray, c2w: np.ndarray, box: dict, change: dict, caption: str
) -> Image.Image:
    w2c = np.linalg.inv(c2w)
    cam = corners(box) @ w2c[:3, :3].T + w2c[:3, 3]
    segs = []
    for p, q in (cam[list(e)] for e in EDGES):
        if p[2] < NEAR and q[2] < NEAR:
            continue
        # Clip at the near plane so edges that pass behind the camera do not wrap across the image.
        if p[2] < NEAR:
            p = p + (q - p) * (NEAR - p[2]) / (q[2] - p[2])
        if q[2] < NEAR:
            q = q + (p - q) * (NEAR - q[2]) / (p[2] - q[2])
        segs.append([K[:2, :2] @ (x[:2] / x[2]) + K[:2, 2] for x in (p, q)])
    H, W = image.shape[:2]
    segs = np.array(segs).reshape(-1, 2, 2)  # (E, 2, 2) pixels
    lo, hi = segs.reshape(-1, 2).min(0).clip(0, [W, H]), segs.reshape(-1, 2).max(0).clip(0, [W, H])
    cw = min(max((hi - lo)[0] * 2.2, (hi - lo)[1] * 2.2 * 4 / 3, 0.4 * W), W, H * 4 / 3)
    x0 = float(np.clip((lo + hi)[0] / 2 - cw / 2, 0, W - cw))
    y0 = float(np.clip((lo + hi)[1] / 2 - cw * 3 / 8, 0, H - cw * 3 / 4))
    img = Image.fromarray(image).crop((x0, y0, x0 + cw, y0 + cw * 3 / 4)).resize(CARD, Image.LANCZOS)
    segs = (segs - [x0, y0]) * CARD[0] / cw
    d, k = ImageDraw.Draw(img, "RGBA"), kind(change)
    for width, color in ((8, (0, 0, 0, 110)), (4, COLORS[k])):
        for seg in segs:
            ring(d, seg, color, width, dash=12 if k == "unverified" else 0, closed=False)
    top = segs.reshape(-1, 2).min(0)
    tag(d, np.clip(top[0], 6, CARD[0] - 90), np.clip(top[1] - 44, 6, CARD[1] - 44), k, str(change["id"]), 36)
    f = font(20)
    d.rounded_rectangle(
        (6, CARD[1] - 36, 18 + d.textlength(caption, font=f), CARD[1] - 6), 6, fill=(0, 0, 0, 150)
    )
    d.text((12, CARD[1] - 21), caption, font=f, fill="white", anchor="lm")
    return img


def write_cards(
    out_dir: Path,
    changes: list[dict],
    objects: dict[int, tuple[np.ndarray | None, np.ndarray | None]],
    views: tuple[Views, Views],
    hires: tuple[np.ndarray, np.ndarray],
) -> None:
    (out_dir / "cards").mkdir(parents=True, exist_ok=True)
    sharp = [sharpness(h) for h in hires]
    for c in changes:
        pts, c["cards"] = objects[c["id"]], {}
        for s, side in enumerate(SIDES):
            # Where the object is absent, the card shows its box from the other video over the empty place.
            present = pts[s] is not None
            box = c[f"box_{side}"] or c[f"box_{SIDES[1 - s]}"]
            i = pick_frame(pts[s] if present else pts[1 - s], box, views[s], sharp[s], present)
            c["cards"][side] = None if i is None else f"cards/{c['id']}_{side}.jpg"
            if i is None:
                continue
            K = views[s].K[i].copy()
            K[:2] *= hires[s].shape[2] / views[s].depth.shape[2]
            empty = "" if present else ", same place" if c["status"] == "unverified" else ", place empty"
            caption = f"{side} video, frame {views[s].frame_ids[i]}{empty}"
            card(hires[s][i], K, views[s].c2w[i], box, c, caption).save(
                out_dir / c["cards"][side], quality=88
            )


def count_text(changes: list[dict]) -> str:
    parts = [f"{n} {t}" for t in ("moved", "added", "removed") if (n := sum(c["type"] == t for c in changes))]
    if not parts:
        return "no changes"
    listed = " and ".join([", ".join(parts[:-1]), parts[-1]] if len(parts) > 1 else parts)
    unverified = sum(c["status"] == "unverified" for c in changes)
    return f"{len(changes)} change{'s' * (len(changes) != 1)} ({listed}), {unverified} of them unverified"


def write_floorplan(
    path: Path,
    before: Cloud,
    after: Cloud,
    states: tuple[np.ndarray, np.ndarray],
    result: dict,
    cameras: tuple[np.ndarray, np.ndarray],
) -> None:
    xyz, state = np.concatenate([before.xyz, after.xyz]), np.concatenate(states)
    paths = [c[:, :2, 3] for c in cameras]  # (N, 2) camera centres on the floor, in keyframe order
    boxes = [c[f"box_{s}"] for c in result["changes"] for s in SIDES if c[f"box_{s}"]]
    boxes += [u["box"] for u in result["unchecked"]]
    ext = np.concatenate(
        [np.percentile(xyz[:, :2], [0.5, 99.5], 0), *paths, *(corners(b)[:, :2] for b in boxes)]
    )
    lo = ext.min(0) - 0.3
    # Round the extent to 10 cm so the 2 cm and 5 cm grids cover exactly the plan area.
    hi = lo + np.ceil((ext.max(0) + 0.3 - lo) / 0.1) * 0.1
    s = min(PLAN[0] / (hi - lo)[0], PLAN[1] / (hi - lo)[1])  # pixels per metre
    pw, ph, m = int((hi - lo)[0] * s), int((hi - lo)[1] * s), 40
    img = Image.new("RGB", (pw + 510 + 3 * m, max(ph, 900) + 2 * m), "white")

    def px(p) -> np.ndarray:
        p = np.asarray(p)
        return np.stack([m + (p[..., 0] - lo[0]) * s, m + (hi[1] - p[..., 1]) * s], -1)

    def cells(cell: float) -> tuple[np.ndarray, tuple[int, int]]:
        shape = tuple(np.round((hi - lo)[::-1] / cell).astype(int))
        ij = ((xyz[:, :2] - lo) / cell).astype(int).clip(0, [shape[1] - 1, shape[0] - 1])
        return (shape[0] - 1 - ij[:, 1]) * shape[1] + ij[:, 0], shape  # flat index, rows run north to south

    flat, shape = cells(0.02)
    top = np.full(shape[0] * shape[1], -1.0)
    np.maximum.at(top, flat, xyz[:, 2].clip(0, HEIGHT_TOP))
    shade = np.where(top < 0, 255, 236 - 116 * top / HEIGHT_TOP).reshape(shape).astype(np.uint8)
    grey = Image.fromarray(shade, "L").filter(ImageFilter.MinFilter(3))
    img.paste(grey.resize((pw, ph), Image.NEAREST), (m, m))
    # Tinted cells were seen by one video only; the other never reached them, so nothing there was checked.
    flat, shape = cells(0.05)
    n, split = shape[0] * shape[1], len(before.xyz)
    reached = np.bincount(flat, state != 3, n) > 0
    nb, na = np.bincount(flat[:split], minlength=n), np.bincount(flat[split:], minlength=n)
    for side, only in (("before", nb >= na), ("after", na > nb)):
        mask = Image.fromarray((~reached & only & (nb + na > 0)).reshape(shape).astype(np.uint8) * 60, "L")
        img.paste(VIDEO_RGB[side], (m, m, m + pw, m + ph), mask.resize((pw, ph), Image.BILINEAR))
    # The RGBA draw mode blends translucent fills into this RGB image.
    d = ImageDraw.Draw(img, "RGBA")
    for side, p in zip(SIDES, paths):
        ring(d, px(p), VIDEO_RGB[side], 3, closed=False)
    for u in result["unchecked"]:
        fp = px(corners(u["box"])[[0, 1, 3, 2], :2])
        ring(d, fp, GREY, 3, dash=10)
        d.text(tuple(fp[np.lexsort(fp.T)[0]] + [6, 4]), u["id"], font=font(26), fill=GREY)
    taken, anchors = [], []
    for c in result["changes"]:
        k, col = kind(c), ImageColor.getrgb(COLORS[kind(c)])
        for side in (s for s in SIDES if c[f"box_{s}"]):
            fp = px(corners(c[f"box_{side}"])[[0, 1, 3, 2], :2])
            ghost = c["type"] == "moved" and side == "before"
            d.polygon([tuple(q) for q in fp], fill=(*col, 40 if ghost or k == "unverified" else 110))
            ring(d, fp, col, 2 if ghost else 4, dash=10 if k == "unverified" else 0)
            taken.append((*fp.min(0), *fp.max(0)))
        anchors.append(taken[-1])
        if c["type"] == "moved":
            a, b = px(c["box_before"]["center"][:2]), px(c["box_after"]["center"][:2])
            v = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            b = b - v * (min(c["box_after"]["size"][:2]) * s / 2 + 4)  # end the arrow at the outline, roughly
            n = np.array([-v[1], v[0]]) * 14
            d.line([tuple(a), tuple(b - v * 20)], fill=col, width=5)
            d.polygon([tuple(b), tuple(b - v * 30 + n), tuple(b - v * 30 - n)], fill=col)
        if c.get("check_from"):
            camera_glyph(d, px(c["check_from"]["position"][:2]), px(c["check_from"]["look_at"][:2]))
    # Tags go on top, at the first corner of their footprint that is clear of footprints and other tags.
    for c, (x0, y0, x1, y1) in zip(result["changes"], anchors):
        w = 53 + d.textlength(str(c["id"]), font=font(26))
        spots = [(x1 + 6, y0 - 46), (x0 - w - 6, y0 - 46), (x1 + 6, y1 + 6), (x0 - w - 6, y1 + 6)]
        clear = [
            (x, y)
            for x, y in spots
            if not any(x < r[2] and r[0] < x + w and y < r[3] and r[1] < y + 40 for r in taken)
        ]
        taken.append(tag(d, *(clear or spots)[0], kind(c), str(c["id"]), 40))
    legend(d, pw + 2 * m, m, result, s)
    img.save(path)


def camera_glyph(d: ImageDraw.ImageDraw, p: np.ndarray, target: np.ndarray) -> None:
    # A top view of a camera: its field of view as a wedge, with a dashed sight line to the spot to film.
    v = (target - p) / max(np.linalg.norm(target - p), 1e-6)
    ring(d, np.array([p, target]), GREY, 2, dash=7, closed=False)
    wedge = [p, p + 36 * v + 22 * np.array([-v[1], v[0]]), p + 36 * v - 22 * np.array([-v[1], v[0]])]
    d.polygon([tuple(q) for q in wedge], fill=(*GREY, 110), outline=GREY, width=3)


def legend(d: ImageDraw.ImageDraw, x: int, y: int, result: dict, s: float) -> None:
    f = font(26)
    d.text((x, y), "Top view of the changes", font=font(32), fill="black")
    d.text((x, y + 44), f"{result['before']} and {result['after']}", font=font(23), fill=(100, 100, 100))
    texts = {"moved": "moved, arrow from old to new place", "unverified": "unverified, dashed outline"}
    rows = [(k, texts.get(k, k)) for k in COLORS] + [("film", "where to film next (unverified)")]
    rows += [(None, "Not checked")] + [(v, f"seen only in the {v} video") for v in SIDES]
    rows += [("unchecked", "unchecked area (U1, U2, ...)"), (None, "Context")]
    rows += [("height", f"structure, darker is higher (0 to {HEIGHT_TOP} m)")]
    rows += [("path", "camera paths of both videos")]
    y += 104
    for sample, text in rows + [("scale", "1 m")]:
        if sample is None:
            d.text((x, y + 24), text, font=f, fill="black")
            y += 70
            continue
        box = np.array([[x, y], [x + 48, y], [x + 48, y + 34], [x, y + 34]])
        if sample in COLORS:
            tag(d, x, y, sample, "", 34)
        elif sample == "film":
            camera_glyph(d, np.array([x + 2.0, y + 17]), np.array([x + 54.0, y + 17]))
        elif sample in SIDES:
            d.rectangle((x, y, x + 48, y + 34), fill=(*VIDEO_RGB[sample], 60))
        elif sample == "unchecked":
            ring(d, box, GREY, 3, dash=8)
        elif sample == "height":
            for i in range(48):
                d.line([(x + i, y), (x + i, y + 34)], fill=(int(236 - 116 * i / 47),) * 3)
        elif sample == "path":
            for k, side in enumerate(SIDES):
                d.line([(x, y + 10 + 14 * k), (x + 48, y + 10 + 14 * k)], fill=VIDEO_RGB[side], width=3)
        else:
            d.rectangle((x, y + 12, x + s, y + 22), fill="black")
        d.text((x + (s + 14 if sample == "scale" else 66), y + 17), text, font=f, fill="black", anchor="lm")
        y += 50


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def write_report(path: Path, result: dict) -> None:
    changes, cov, led = result["changes"], result["coverage"], result["ledger"]
    out = [
        f"""# What changed in the room

Between {result["before"]} and {result["after"]} roomdiff found {count_text(changes)}. An unverified change is
one whose place the other video did not see well enough to decide. The camera was held
{result["camera_height_m"]:.2f} m above the floor; positions are in metres with z up and the floor at z = 0.

![Top view of the changes](floorplan.png)

| id | | type | label | status | sentence |
|---|---|---|---|---|---|"""
    ]
    for c in changes:
        out.append(f"| {c['id']} | {GLYPHS[kind(c)]} | {c['type']} | {c['label']} | {c['status']} | ")
        out[-1] += f"{c['sentence']} |"
    for c in changes:
        e, mo, w = c["evidence"], c["motion"], c.get("check_from")
        facts = [f"Status: {c['status']}, because {c['reason']}."]
        if mo:
            turn = f", turned {mo['turn_deg']:.0f} degrees" if mo["turn_deg"] is not None else ""
            facts.append(f"Motion: {mo['distance_m']:.2f} m{turn}.")
        if c.get("floor_blocked_m2"):
            facts.append(f"Floor now blocked: {c['floor_blocked_m2']:.2f} m².")
        if w:
            p, q = (", ".join(f"{v:.2f}" for v in w[k]) for k in ("position", "look_at"))
            facts.append(f"Where to film next: where the {w['video']} video stood at {w['time_s']:.1f} s, ")
            facts[-1] += f"at ({p}), looking at ({q})."
        facts.append(
            f"Evidence: of its {e['points']} points, the other video looked through {pct(e['free'])}, "
        )
        facts[-1] += (
            f"saw {pct(e['seen'])}, had {pct(e['occluded'])} hidden, never had {pct(e['unreached'])} in view."
        )
        cards = " ".join(
            f'<img src="{c["cards"][s]}" width="49%">' if c["cards"][s] else f"(no {s} view)" for s in SIDES
        )
        bullets = "\n".join(f"- {f}" for f in facts)
        out.append(f"\n## {GLYPHS[kind(c)]} {c['id']}: {c['label']}, {c['type']}\n\n{c['sentence']}")
        out.append(f"\n{bullets}\n\n{cards}")
    unchecked = "\n".join(
        f"- {u['id']} (seen only in the {u['video']} video): {u['sentence']}" for u in result["unchecked"]
    )
    out.append(f"""
## Areas that were not checked

{unchecked or "Both videos saw every place that holds a candidate change."}

## Coverage

The other video checked {pct(cov["checked"])} of all surface points. {pct(cov["only_before"])} were seen only
in the before video and {pct(cov["only_after"])} only in the after video; the tinted parts of the top view
show where.

Of all surface points that the other video looked through, {pct(led["confirmed"])} belong to confirmed
changes, {pct(led["unverified"])} to unverified changes, {pct(led["dismissed"])} to dismissed candidates and
{pct(led["residual"])} to no candidate.

## Method

Both videos were reconstructed in one pass into a shared metric frame with z up and the floor at z = 0. A
surface counts as removed or added only where the other video looked through the place it occupied, and each
change keeps the share of its points that were looked through, seen, hidden or never in view.
""")
    path.write_text("\n".join(out))
