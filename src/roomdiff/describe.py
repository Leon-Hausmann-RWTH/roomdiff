import numpy as np

from .changes import Change, fit_box
from .evidence import SEEN
from .geometry import Cloud, Views
from .objects import Obj, best_frames, hires, visible
from .semantics import color_name, crop, label


def mask_box(m: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def object_crops(o: Obj, images) -> list[np.ndarray]:
    return [crop(images[o.side][i], m, mask_box(m)) for i, m in o.masks.items() if m.any()]


def object_pixels(o: Obj, images) -> np.ndarray:
    return np.concatenate([images[o.side][i][m] for i, m in o.masks.items()] or [np.zeros((0, 3), np.uint8)])


def support(o: Obj, cloud: Cloud, state: np.ndarray, taken: np.ndarray, floor_found: bool) -> dict:
    """What the object stands on: the floor, or the static surface right under its footprint."""
    xyz = cloud.xyz[o.idx]
    bottom = float(np.percentile(xyz[:, 2], 3))
    if floor_found and bottom < 0.08:
        return {"name": "floor", "height_m": 0.0, "points": None}
    lo, hi = xyz[:, :2].min(0) - 0.05, xyz[:, :2].max(0) + 0.05
    z = cloud.xyz[:, 2]
    under = (
        np.all((cloud.xyz[:, :2] >= lo) & (cloud.xyz[:, :2] <= hi), 1)
        & (z > bottom - 0.12)
        & (z < bottom + 0.02)
    )
    under &= (state == SEEN) & ~taken
    if under.sum() < 10:
        return {"name": None, "height_m": round(bottom, 2), "points": None}
    return {"name": None, "height_m": round(float(np.median(z[under])), 2), "points": np.nonzero(under)[0]}


def surface_crop(o: Obj, pts: np.ndarray, cloud: Cloud, views: Views, images) -> np.ndarray | None:
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
    return f"on the {s['name']} ({cm} cm up)" if s["name"] else f"on a surface {cm} cm above the floor"


def name_changes(
    changes: list[Change], clouds, states, views, images, times, floor_found: bool, device
) -> None:
    """Labels and sentences. A VLM never decides what changed; it does not even run here."""
    taken = [np.zeros(len(c.xyz), bool) for c in clouds]
    for c in changes:
        for o in (c.before, c.after):
            if o is not None:
                taken[o.side][o.idx] = True
    crops, surfaces, slots = [], [], []
    for c in changes:
        objs = [o for o in (c.before, c.after) if o is not None]
        crops.append(
            [x for o in objs for x in object_crops(o, images)][:6] or [np.zeros((32, 32, 3), np.uint8)]
        )
        c.facts["support"] = {}
        for o in objs:
            s = support(o, clouds[o.side], states[o.side], taken[o.side], floor_found)
            pts = s.pop("points")
            c.facts["support"]["before" if o.side == 0 else "after"] = s
            sc = None if pts is None else surface_crop(o, pts, clouds[o.side], views[o.side], images)
            if sc is not None:
                surfaces.append([sc])
                slots.append(s)
    labels, surface_names = label(crops, surfaces, device)
    for s, (name, _) in zip(slots, surface_names):
        s["name"] = name
    for c, (word, score) in zip(changes, labels):
        objs = [o for o in (c.before, c.after) if o is not None]
        c.label, c.facts["label_score"] = word, round(float(score), 3)
        colour = color_name(np.concatenate([object_pixels(o, images) for o in objs]))
        # On 80 SceneDiff objects SigLIP was right for 95 % of names scored 0.5 or more and often wrong
        # under 0.35; say so instead of naming confidently.
        name = "object" if score < 0.35 else f"{colour} {word}" if colour else word
        hedge = f", possibly {article(word)}," if score < 0.35 else ""
        c.facts["height_m"] = round(float(fit_box(clouds[objs[0].side].xyz[objs[0].idx]).size[2]), 2)
        if c.kind != "removed" and c.facts["support"].get("after", {}).get("name") == "floor":
            # For a robot the question is what is now in the way.
            end = fit_box(clouds[objs[-1].side].xyz[objs[-1].idx])
            c.facts["floor_blocked_m2"] = round(float(end.size[0] * end.size[1]), 2)
        # A viewpoint helps only where the other video saw too little, not where part is still there.
        hidden = [o for o in objs if o.status == "unverified"]
        if c.status == "unverified" and hidden:
            c.facts["check_from"] = check_from(hidden[0], clouds, views, times)
        c.sentence = sentence(c, name, hedge)


def sentence(c: Change, name: str, hedge: str) -> str:
    sup, may = c.facts["support"], c.status == "unverified"
    if c.kind == "moved":
        m, before, after = c.motion, sup["before"], sup["after"]
        text = f"The {name}{hedge} {'may have moved' if may else 'moved'} {m.distance:.1f} m"
        # Two names for surfaces at one height are usually one piece of furniture named twice.
        if abs(before["height_m"] - after["height_m"]) < 0.05 and "floor" not in (
            before["name"],
            after["name"],
        ):
            after = before
        if before["name"] != after["name"]:
            text += (
                f" from {on_phrase(before).removeprefix('on ')} onto {on_phrase(after).removeprefix('on ')}"
            )
        elif before["name"]:
            text += " across the floor" if before["name"] == "floor" else f" along the {before['name']}"
        if m.turn is not None and abs(m.turn) >= 15:
            text += f" and turned about {abs(round(m.turn / 5) * 5)} degrees"
    elif c.kind == "removed":
        text = f"The {name} {on_phrase(sup['before'])}{hedge} {'may have been' if may else 'was'} removed"
    else:
        verb = "may have appeared" if may else "appeared"
        text = f"{article(name).capitalize()}{hedge} {verb} {on_phrase(sup['after'])}"
    if c.facts.get("floor_blocked_m2", 0) >= 0.05:
        text += f", blocking {c.facts['floor_blocked_m2']:.2f} m² of floor"
    return text + "."


def check_from(o: Obj, clouds, views, times) -> dict:
    """The camera pose that would settle an unverified change: the keyframe that saw the object best.

    Turning "could not check" into a place to stand is what a robot needs to resolve it.
    """
    i = best_frames(clouds[o.side].xyz[o.idx], views[o.side], k=1, min_points=1)[0]
    pose = views[o.side].c2w[i]
    return {
        "video": "before" if o.side == 0 else "after",
        "time_s": round(float(times[o.side][i]), 1),
        "position": np.round(pose[:3, 3], 3).tolist(),
        "look_at": np.round(clouds[o.side].xyz[o.idx].mean(0), 3).tolist(),
    }
