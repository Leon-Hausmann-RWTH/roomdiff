import numpy as np

from .changes import Change, fit_box
from .config import Config, pick_device
from .evidence import SEEN
from .geometry import Cloud, Views
from .objects import Obj, best_frames, hires, visible
from .semantics import color_name, crop, label

# Below this label score the name is given as a guess: "possibly a ...".
HEDGE = 0.35


def mask_box(m: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def support(o: Obj, cloud: Cloud, state: np.ndarray, taken: np.ndarray, floor_found: bool) -> dict:
    """What the object stands on: the floor, or the static surface right under its footprint."""
    xyz = cloud.xyz[o.idx]
    bottom = float(np.percentile(xyz[:, 2], 3))
    if bottom < 0.08 and floor_found:
        # The floor is named from geometry; other supports are named by SigLIP later.
        return {"name": "floor", "height_m": 0.0, "points": None}
    lo, hi = xyz[:, :2].min(0) - 0.05, xyz[:, :2].max(0) + 0.05
    z = cloud.xyz[:, 2]
    # Seen points of no change, inside the footprint and just below the object's bottom.
    under = (
        np.all((cloud.xyz[:, :2] >= lo) & (cloud.xyz[:, :2] <= hi), 1)
        & (z > bottom - 0.12)
        & (z < bottom + 0.02)
        & (state == SEEN)
        & ~taken
    )
    if under.sum() < 10:
        return {"name": None, "height_m": max(0.0, round(bottom, 2)), "points": None}
    height = max(0.0, round(float(np.median(z[under])), 2))
    if floor_found and height < 0.08:  # thin legs leave no low points; the surface under them is the floor
        return {"name": "floor", "height_m": 0.0, "points": None}
    return {"name": None, "height_m": height, "points": np.nonzero(under)[0]}


def surface_crop(o: Obj, pts: np.ndarray, cloud: Cloud, views: Views, images) -> np.ndarray | None:
    """Crop around the support and the object in the keyframe that sees the support best."""
    frames = best_frames(cloud.xyz[pts], views, k=1)
    if not frames:
        return None
    i = frames[0]
    u, v, ok = visible(np.concatenate([cloud.xyz[pts], cloud.xyz[o.idx]]), views, i)
    px = hires(u[ok], v[ok])
    if len(px) < 10:
        return None
    lo, hi = px.min(0), px.max(0)
    return crop(images[o.side][i], None, (int(lo[0]), int(lo[1]), int(hi[0]) + 1, int(hi[1]) + 1), pad=0.1)


def article(word: str) -> str:
    return f"an {word}" if word[0] in "aeiou" else f"a {word}"


def on_phrase(s: dict) -> str:
    if s["name"] == "floor":
        return "on the floor"
    cm = round(s["height_m"] * 100)
    if s["name"]:
        return f"on the {s['name']} ({cm} cm up)" if cm >= 5 else f"on the {s['name']}"
    return f"on a surface {cm} cm up" if cm >= 5 else "on a surface"


def name_changes(
    changes: list[Change], clouds, states, taken, views, images, floor_found: bool, cfg: Config
) -> None:
    """Set the label, title, support, blocked floor area and sentence of each change; taken marks the
    points of all changes per video."""
    crops, surfaces, slots = [], [], []
    for c in changes:
        objs = [o for o in (c.before, c.after) if o is not None]
        cs = [crop(images[o.side][i], m, mask_box(m)) for o in objs for i, m in o.masks.items() if m.any()]
        crops.append(cs[:6] or [np.zeros((32, 32, 3), np.uint8)])
        c.facts["support"] = {}
        for o in objs:
            s = support(o, clouds[o.side], states[o.side], taken[o.side], floor_found)
            pts = s.pop("points")
            c.facts["support"]["before" if o.side == 0 else "after"] = s
            sc = None if pts is None else surface_crop(o, pts, clouds[o.side], views[o.side], images)
            if sc is not None:
                surfaces.append([sc])
                slots.append(s)
    labels, surface_names = label(crops, surfaces, pick_device(cfg.device))
    for s, (name, _) in zip(slots, surface_names):
        s["name"] = name
    for c, (word, score) in zip(changes, labels):
        objs = [o for o in (c.before, c.after) if o is not None]
        c.label, c.facts["label_score"] = word, round(float(score), 3)
        pixels = [images[o.side][i][m] for o in objs for i, m in o.masks.items()]
        colour = color_name(np.concatenate(pixels or [np.zeros((0, 3), np.uint8)]))
        name = "object" if score < HEDGE else f"{colour} {word}" if colour else word
        hedge = f", possibly {article(word)}," if score < HEDGE else ""
        c.facts["title"] = f"possibly {article(word)}" if score < HEDGE else word
        if c.kind != "removed" and c.facts["support"].get("after", {}).get("name") == "floor":
            # Floor area under the object's box at its new place.
            end = fit_box(clouds[objs[-1].side].xyz[objs[-1].idx])
            c.facts["floor_blocked_m2"] = round(float(end.size[0] * end.size[1]), 2)
        if c.kind == "moved":
            # A kept pair that moved less than min_move only turned.
            c.facts["in_place"] = c.motion.distance < cfg.min_move
        c.sentence = sentence(c, name, hedge, cfg.min_turn)


def sentence(c: Change, name: str, hedge: str, min_turn: float) -> str:
    sup, may = c.facts["support"], c.status == "unverified"
    if c.kind == "moved" and c.facts["in_place"]:
        verb = "may have turned" if may else "turned"
        turn = f"about {abs(round(c.motion.turn / 5) * 5)} degrees in place"
        text = f"The {name} {on_phrase(sup['before'])}{hedge} {verb} {turn}"
    elif c.kind == "moved":
        m, before, after = c.motion, sup["before"], sup["after"]
        text = f"The {name}{hedge} {'may have moved' if may else 'moved'} {m.distance:.1f} m"
        same = abs(before["height_m"] - after["height_m"]) < 0.05
        # Two supports off the floor at one height count as one surface, under either name.
        if same and "floor" not in (before["name"], after["name"]):
            before = after = before if before["name"] else after
        if before["name"] != after["name"] or not same:
            text += (
                f" from {on_phrase(before).removeprefix('on ')} onto {on_phrase(after).removeprefix('on ')}"
            )
        elif before["name"]:
            text += " across the floor" if before["name"] == "floor" else f" along the {before['name']}"
        if m.turn is not None and abs(m.turn) >= min_turn:
            text += f" and turned about {abs(round(m.turn / 5) * 5)} degrees"
    elif c.kind == "removed":
        text = f"The {name} {on_phrase(sup['before'])}{hedge} {'may have been' if may else 'was'} removed"
    else:
        verb = "may have appeared" if may else "appeared"
        text = f"{article(name).capitalize()}{hedge} {verb} {on_phrase(sup['after'])}"
    if c.facts.get("floor_blocked_m2", 0) >= 0.05:
        text += f", blocking {c.facts['floor_blocked_m2']:.2f} m² of floor"
    return text + "."
