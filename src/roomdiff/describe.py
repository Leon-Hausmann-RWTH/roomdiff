import numpy as np

from .changes import Box, Change, components, fit_box
from .evidence import SEEN, UNREACHED
from .geometry import Cloud, Views
from .objects import Obj, best_frames, hires, visible
from .semantics import color_name, crop, label_objects, name_surfaces


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


def direction(d: np.ndarray, views: Views) -> str:
    # Words relative to the first frame of the before video, the view a person replaying it starts with.
    right = views.c2w[0, :3, 0]
    fwd = views.c2w[0, :3, 2]
    right, fwd = right[:2] / np.linalg.norm(right[:2]), fwd[:2] / np.linalg.norm(fwd[:2])
    x, y = float(d[:2] @ right), float(d[:2] @ fwd)
    if abs(x) >= abs(y):
        return "to the right" if x > 0 else "to the left"
    return "away from the camera" if y > 0 else "toward the camera"


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
        crops.append([x for o in objs for x in object_crops(o, images)][:6])
        c.facts["support"] = {}
        for o in objs:
            s = support(o, clouds[o.side], states[o.side], taken[o.side], floor_found)
            pts = s.pop("points")
            c.facts["support"]["before" if o.side == 0 else "after"] = s
            if pts is not None:
                sc = surface_crop(o, pts, clouds[o.side], views[o.side], images)
                if sc is not None:
                    surfaces.append([sc])
                    slots.append(s)
    labels = label_objects([x if x else [np.zeros((32, 32, 3), np.uint8)] for x in crops], device)
    for s, (name, _) in zip(slots, name_surfaces(surfaces, device) if surfaces else []):
        s["name"] = name
    for c, (label, score), objs in zip(
        changes, labels, ([o for o in (c.before, c.after) if o] for c in changes)
    ):
        c.label, c.facts["label_score"] = label, round(float(score), 3)
        colour = color_name(np.concatenate([object_pixels(o, images) for o in objs]))
        name = f"{colour} {label}" if colour else label
        if score < 0.25:
            # Objects SigLIP scored under 0.25 were often misnamed on the SceneDiff check; say so.
            name = f"object (possibly a {label})"
        box = fit_box(clouds[objs[0].side].xyz[objs[0].idx])
        c.facts["height_m"] = round(float(box.size[2]), 2)
        last = objs[-1]
        if c.kind != "removed" and c.facts["support"].get("after", {}).get("name") == "floor":
            # For a robot the question is what is now in the way.
            end = fit_box(clouds[last.side].xyz[last.idx])
            c.facts["floor_blocked_m2"] = round(float(end.size[0] * end.size[1]), 2)
        if c.status == "unverified":
            c.facts["check_from"] = check_from(objs[0], clouds, views, times)
        c.sentence = sentence(c, name, views[0])


def sentence(c: Change, name: str, first_view: Views) -> str:
    sup = c.facts["support"]
    if c.kind == "moved":
        m = c.motion
        text = f"The {name} moved {m.distance:.1f} m {direction(m.shift, first_view)}"
        before, after = sup.get("before"), sup.get("after")
        if before and after and before["name"] != after["name"]:
            text += (
                f", from {on_phrase(before).removeprefix('on ')} onto {on_phrase(after).removeprefix('on ')}"
            )
        if m.turn is not None and abs(m.turn) >= 15:
            text += f", and turned about {abs(round(m.turn / 5) * 5)} degrees"
        text += "."
    elif c.kind == "removed":
        text = f"The {name} {on_phrase(sup['before'])} was removed."
    else:
        text = f"A {name} appeared {on_phrase(sup['after'])}."
    if c.facts.get("floor_blocked_m2", 0) >= 0.05:
        text = text[:-1] + f", blocking {c.facts['floor_blocked_m2']:.2f} m² of floor."
    if c.status == "unverified":
        # The reason and the viewpoint that would settle it travel as their own fields.
        text = f"Possibly: {text[0].lower()}{text[1:]}"
    return text


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


def unchecked(cloud: Cloud, state: np.ndarray, side: int, taken: np.ndarray) -> list[dict]:
    """Object-sized regions one video saw and the other never looked at: no claim either way."""
    keep = np.nonzero((state == UNREACHED) & ~taken & (cloud.xyz[:, 2] > 0.05) & (cloud.xyz[:, 2] < 2.2))[0]
    if len(keep) < 200:
        return []
    labels = components(cloud.xyz[keep], 0.08)
    areas = []
    for lab in np.unique(labels):
        idx = keep[labels == lab]
        if len(idx) < 200:
            continue
        box = fit_box(cloud.xyz[idx])
        # Object-sized only: long strips of wall or floor the other video skipped are coverage, which
        # the floor plan shows as a tint, not a place where something may hide.
        if not 0.3 <= box.size[:2].max() <= 1.5 or box.size.min() < 0.04:
            continue
        areas.append((len(idx), box))
    areas.sort(key=lambda a: -a[0])
    other = "after" if side == 0 else "before"
    return [
        {"video": "before" if side == 0 else "after", "box": b, "sentence": unchecked_sentence(b, other)}
        for _, b in areas[:2]
    ]


def unchecked_sentence(b: Box, other: str) -> str:
    w, d = sorted(b.size[:2])[::-1]
    top = b.center[2] + b.size[2] / 2
    return f"The {other} video never looked at a region {w:.1f} m by {d:.1f} m, up to {top:.1f} m high."
