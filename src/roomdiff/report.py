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
Pair = tuple[np.ndarray, np.ndarray]  # one array per video, before then after
# Pillow bundles this font, so images look the same everywhere; it lacks minus and arrow, hence tag().
font = ImageFont.load_default


def kind(change: dict) -> str:
    return "unverified" if change["status"] == "unverified" else change["type"]


def corners(box: dict) -> np.ndarray:
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    signs = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)])
    return np.asarray(box["center"]) + (signs * np.asarray(box["size"]) / 2) @ R.T  # (8, 3)


def tag(d: ImageDraw.ImageDraw, x: float, y: float, k: str, text: str, h: int) -> tuple:
    # A rounded box in the change colour with its glyph drawn as lines, then the id.
    f, cx, cy, s = font(round(h * 0.66)), x + h / 2, y + h / 2, h * 0.62
    w, lw, r = h + (d.textlength(text, font=f) + h * 0.3 if text else 0), max(2, round(s / 6)), s * 0.38
    d.rounded_rectangle((x, y, x + w, y + h), radius=h * 0.22, fill=COLORS[k])
    if k == "unverified":
        d.text((cx, cy), "?", font=font(round(s * 1.15)), fill="white", anchor="mm")
    else:
        d.line([(cx - r, cy), (cx + (s * 0.2 if k == "moved" else r), cy)], fill="white", width=lw)
    if k == "added":
        d.line([(cx, cy - r), (cx, cy + r)], fill="white", width=lw)
    if k == "moved":
        a, t = cx + s * 0.1, s * 0.3
        d.polygon([(cx + s * 0.46, cy), (a, cy - t), (a, cy + t)], fill="white")
    d.text((x + h * 0.95, cy), text, font=f, fill="white", anchor="lm")
    return (x, y, x + w, y + h)


def ring(d: ImageDraw.ImageDraw, pts: np.ndarray, fill, width: int, dash=0.0, closed=True) -> None:
    for p, q in zip(pts, np.roll(pts, -1, 0)[: len(pts) - (not closed)]):
        n = max(np.linalg.norm(q - p), 1e-6)
        for a in np.arange(0, n, 2 * dash) if dash else [0]:
            b = min(a + dash, n) if dash else n
            d.line([tuple(p + (q - p) * a / n), tuple(p + (q - p) * b / n)], fill=fill, width=width)


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


def card(image: np.ndarray, K: np.ndarray, c2w: np.ndarray, box: dict, c: dict, caption: str) -> Image.Image:
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
    d, k, f = ImageDraw.Draw(img, "RGBA"), kind(c), font(20)
    for width, color in ((8, (0, 0, 0, 110)), (4, COLORS[k])):
        for seg in segs:
            ring(d, seg, color, width, dash=12 if k == "unverified" else 0, closed=False)
    top = segs.reshape(-1, 2).min(0)
    tag(d, np.clip(top[0], 6, CARD[0] - 90), np.clip(top[1] - 44, 6, CARD[1] - 44), k, str(c["id"]), 36)
    w = d.textlength(caption, font=f)
    d.rounded_rectangle((6, CARD[1] - 36, 18 + w, CARD[1] - 6), 6, fill=(0, 0, 0, 150))
    d.text((12, CARD[1] - 21), caption, font=f, fill="white", anchor="lm")
    return img


def write_cards(
    out_dir: Path, changes: list[dict], objects: dict[int, tuple], views: tuple, hires: Pair
) -> None:
    # objects[id] holds the object's world points in each video, None where it is absent; views are
    # Views at reconstruction resolution and hires the keyframes at a whole multiple of it.
    (out_dir / "cards").mkdir(parents=True, exist_ok=True)
    sharp = []  # per video, (N,) variance of the Laplacian of each keyframe, higher is sharper
    for g in (h[:, ::2, ::2].astype(np.float32).mean(-1) for h in hires):
        lap = 4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
        sharp.append(lap.var((1, 2)))
    for c in changes:
        pts, c["cards"] = objects[c["id"]], {}
        for s, side in enumerate(SIDES):
            # Where the object is absent, the card shows its box from the other video over the empty place.
            present, box = pts[s] is not None, c[f"box_{side}"] or c[f"box_{SIDES[1 - s]}"]
            i = pick_frame(pts[s] if present else pts[1 - s], box, views[s], sharp[s], present)
            c["cards"][side] = None if i is None else f"cards/{c['id']}_{side}.jpg"
            if i is None:
                continue
            K = views[s].K[i].copy()
            K[:2] *= hires[s].shape[2] / views[s].depth.shape[2]
            empty = "" if present else ", same place" if c["status"] == "unverified" else ", place empty"
            caption = f"{side} video, frame {views[s].frame_ids[i]}{empty}"
            out = out_dir / c["cards"][side]
            card(hires[s][i], K, views[s].c2w[i], box, c, caption).save(out, quality=88)


def write_floorplan(
    path: Path, before: Cloud, after: Cloud, states: Pair, result: dict, cameras: Pair
) -> None:
    xyz, state, changes = np.concatenate([before.xyz, after.xyz]), np.concatenate(states), result["changes"]
    paths = [c[:, :2, 3] for c in cameras]  # (N, 2) camera centres on the floor, in keyframe order
    boxes = [corners(c[f"box_{s}"])[:, :2] for c in changes for s in SIDES if c[f"box_{s}"]]
    ext = np.concatenate([np.percentile(xyz[:, :2], [0.5, 99.5], 0), *paths, *boxes])
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
    d = ImageDraw.Draw(img, "RGBA")  # the RGBA mode blends translucent fills into this RGB image
    for side, p in zip(SIDES, paths):
        ring(d, px(p), VIDEO_RGB[side], 3, closed=False)
    taken = []
    for c in changes:
        k, col = kind(c), ImageColor.getrgb(COLORS[kind(c)])
        for side in (s for s in SIDES if c[f"box_{s}"]):
            fp = px(corners(c[f"box_{side}"])[[0, 1, 3, 2], :2])
            ghost = c["type"] == "moved" and side == "before"
            d.polygon([tuple(q) for q in fp], fill=(*col, 40 if ghost or k == "unverified" else 110))
            ring(d, fp, col, 2 if ghost else 4, dash=10 if k == "unverified" else 0)
            taken.append((*fp.min(0), *fp.max(0)))
        if c["type"] == "moved":
            a, b = px(c["box_before"]["center"][:2]), px(c["box_after"]["center"][:2])
            v = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            b = b - v * (min(c["box_after"]["size"][:2]) * s / 2 + 4)  # end the arrow at the outline, roughly
            n = np.array([-v[1], v[0]]) * 14
            d.line([tuple(a), tuple(b - v * 20)], fill=col, width=5)
            d.polygon([tuple(b), tuple(b - v * 30 + n), tuple(b - v * 30 - n)], fill=col)
        if c["check_from"]:
            camera_glyph(d, px(c["check_from"]["position"][:2]), px(c["check_from"]["look_at"][:2]))
    # Tags go on top, at the first corner of their footprint that is clear of footprints and other tags.
    for c in changes:
        fp = px(corners(c["box_after"] or c["box_before"])[:, :2])
        (x0, y0), (x1, y1) = fp.min(0), fp.max(0)
        w = 53 + d.textlength(str(c["id"]), font=font(26))
        spots = [(x1 + 6, y0 - 46), (x0 - w - 6, y0 - 46), (x1 + 6, y1 + 6), (x0 - w - 6, y1 + 6)]
        for x, y in spots:
            if not any(x < r[2] and r[0] < x + w and y < r[3] and r[1] < y + 40 for r in taken):
                break
        else:
            x, y = spots[0]
        taken.append(tag(d, x, y, kind(c), str(c["id"]), 40))
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
    y += 104

    def row(text: str, dx: float = 66, dy: int = 50) -> None:
        nonlocal y
        d.text((x + dx, y + (17 if dx else 24)), text, font=f, fill="black", anchor="lm" if dx else "la")
        y += dy

    texts = {"moved": "moved, arrow from old to new place", "unverified": "unverified, dashed outline"}
    for k in COLORS:
        tag(d, x, y, k, "", 34)
        row(texts.get(k, k))
    camera_glyph(d, np.array([x + 2.0, y + 17]), np.array([x + 54.0, y + 17]))
    row("where to film next (unverified)")
    row("Not checked", 0, 70)
    for side in SIDES:
        d.rectangle((x, y, x + 48, y + 34), fill=(*VIDEO_RGB[side], 60))
        row(f"seen only in the {side} video")
    row("Context", 0, 70)
    for i in range(48):
        d.line([(x + i, y), (x + i, y + 34)], fill=(int(236 - 116 * i / 47),) * 3)
    row(f"structure, darker is higher (0 to {HEIGHT_TOP} m)")
    for k, side in enumerate(SIDES):
        d.line([(x, y + 10 + 14 * k), (x + 48, y + 10 + 14 * k)], fill=VIDEO_RGB[side], width=3)
    row("camera paths of both videos")
    d.rectangle((x, y + 12, x + s, y + 22), fill="black")
    row("1 m", s + 14)


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def join_and(parts: list[str]) -> str:
    return " and ".join([", ".join(parts[:-1]), parts[-1]] if len(parts) > 1 else parts)


def write_report(path: Path, result: dict) -> None:
    n, cov, led = result["counts"], result["coverage"], result["ledger"]
    total = n["moved"] + n["added"] + n["removed"]
    found = ", ".join(f"{n[t]} {t}" for t in ("moved", "added", "removed") if n[t])
    found = f"{total} change{'s' * (total != 1)}: {found}" if total else "no changes"
    found += f"; {n['unverified']} of them unverified" if n["unverified"] else ""
    words = {"free": "looked through", "seen": "still seen", "occluded": "hidden", "unreached": "not in view"}
    rows, sections = [], []
    for c in result["changes"]:
        mo, w, e, g = c["motion"], c["check_from"], c["evidence"], GLYPHS[kind(c)]
        blocked = c["floor_blocked_m2"]
        rows.append(f"| {c['id']} | {g} | {c['type']} | {c['label']} | {c['status']} | {c['sentence']} |")
        turn = f", turned {mo['turn_deg']:.0f} degrees" if mo and mo["turn_deg"] is not None else ""
        shares = ", ".join(f"{pct(e[k])} {v}" for k, v in words.items() if round(100 * e[k]))
        facts = [
            f"Status: {c['status']} ({c['reason']}).",
            mo and f"Motion: {mo['distance_m']:.2f} m{turn}.",
            blocked and f"Floor now blocked: {blocked:.2f} m².",
            w and f"Film again from where the {w['video']} video stood at {w['time_s']:.1f} s.",
            f"Evidence: {e['points']:,} points: {shares}.",
        ]
        cards = [f'<img src="{p}" width="49%">' if p else f"(no {s} view)" for s, p in c["cards"].items()]
        sections.append(f"## {g} {c['id']}: {c['label']}, {c['type']}\n\n{c['sentence']}\n\n")
        sections[-1] += "\n".join(f"- {f}" for f in facts if f) + "\n\n" + " ".join(cards) + "\n"
    only = [f"{pct(cov['only_' + v])} only in the {v} video" for v in SIDES if round(100 * cov["only_" + v])]
    one = only and f" The rest was seen by one video only: {join_and(only)}; the top view tints those parts."
    one = one or ""
    checked = f"The other video checked {pct(cov['checked'])} of all surface points"
    checked += f"; {pct(cov['hidden'])} were hidden from it."
    names = ("confirmed changes", "unverified changes", "dismissed candidates", "no candidate")
    keys = ("confirmed", "unverified", "dismissed", "residual")
    ledger = join_and([f"{pct(led[k])} {v}" for k, v in zip(keys, names) if round(100 * led[k])])
    table, body = "\n".join(rows), "\n".join(sections)
    path.write_text(f"""# What changed in the room

Between {result["before"]} and {result["after"]} roomdiff found {found}.
Unverified means not enough evidence either way: the other video saw too little of it, or part of it is still
there. The camera was held {result["camera_height_m"]:.2f} m above the floor; positions are in metres with z
up and the floor at z = 0.

![Top view of the changes](floorplan.png)

| id | | type | label | status | sentence |
|---|---|---|---|---|---|
{table}

{body}
## Coverage

{checked}{one}

The points the other video looked through belong to: {ledger}.

## Method

Both videos were reconstructed in one pass into a shared metric frame with z up and the floor at z = 0. A
surface counts as removed or added only where the other video looked through the place it occupied, and each
change keeps the share of its points that were looked through, still seen, hidden or not in view.
""")
