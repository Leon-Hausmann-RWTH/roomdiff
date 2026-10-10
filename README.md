# roomdiff

Compare two phone videos of the same room and find, in 3D, what was moved, added or removed. Every verdict comes with the evidence behind it, and what one video could not check is reported as such instead of being called a change.

![The detected edits applied to the before scene](docs/media/edit_slider.gif)

*The before scene with the detected edits applied, from 0 to 100 %: the cushion travels along the sofa, the stool and the coat disappear.*

## See the results without installing

- 3D viewer: [docs/index.html](docs/index.html). One self-contained file: download it and open it in a browser, it works offline. With GitHub Pages serving `/docs`, it opens at the Pages URL.
- Living room example: [report](examples/living_room/output/report.md), [changes.json](examples/living_room/output/changes.json), input videos [before](examples/living_room/before.mp4) and [after](examples/living_room/after.mp4).
- Bedroom example (with additions): [report](examples/bedroom/output/report.md), [viewer](examples/bedroom/output/viewer.html), [before](examples/bedroom/before.mp4), [after](examples/bedroom/after.mp4).

![Top view of the changes](docs/media/floorplan.png)

Evidence cards: the same 3D box drawn into the best before frame and the best after frame.

![Evidence cards, living room](docs/media/cards_living_room.jpg)
![Evidence cards, bedroom](docs/media/cards_bedroom.jpg)

Both examples are SceneDiff pairs (living_room_17/18 and bedroom_32/33, MIT). In the living room, the ground truth lists 7 changes. roomdiff reports 4, all real: the cushion, the stool, the coat (named only "an object, possibly a blanket") and a 5 cm tote bag that SigLIP calls a shoe. The bag is left unverified: 62 % of its points still test as seen against the seat under it, and its new place on the coffee table was not matched. It misses the bread (6 cm tall, too few free points), the tissue box (the before video never looked through its place) and the remote (1 cm). In the bedroom it reports 3 of 4 and misses a cable lying flat on the desk.

## Quickstart

```bash
uv sync
uv run roomdiff run examples/living_room/before.mp4 examples/living_room/after.mp4 --out runs/demo
open runs/demo/viewer.html   # xdg-open on Linux
```

Runs on Apple Silicon (MPS) or an NVIDIA GPU (CUDA); the device is chosen automatically, and the CPU works but is slow. The first run downloads about 10 GB of model weights. On an M2 Max with 32 GB a pair of 10 s videos takes about 2 minutes (80 s of it the reconstruction) and peaks at 12 GB of GPU memory; on a 16 GB Mac add `--config.frames-per-video 16`.

If `uv run` stops with `No module named 'roomdiff'` on a Mac, a folder sync has set the hidden flag on the venv's `.pth` files, which Python then skips: `chflags nohidden .venv/lib/python3*/site-packages/*.pth`.

The output folder holds `changes.json` (one record per change), `viewer.html` (3D viewer with an edit slider), `floorplan.png`, `report.md` with evidence cards, and `scene.ply` (both point clouds, with the video and the change id of every point).

## How it works

![Ray evidence](docs/media/evidence.svg)

1. **One shared 3D frame.** Keyframes of both videos go through one Pi3X pass, which returns depth and a camera pose for every frame in one metric frame. Gravity comes from the cameras (handheld phones keep roll near zero), and the floor is set to z = 0.
2. **Change needs evidence.** Every surface point the before video saw is projected into every after frame. If most of at least three after frames saw farther along that ray, they looked through the place where the surface used to be: the point is *free*, evidence of absence. If something nearer blocked the ray, the point is *occluded*; if no frame reached it, *unreached*. Neither says anything about change. The same test runs from after to before for additions. A group of free points counts only if cameras in two or more places looked through it.
3. **Objects and verdicts.** Each candidate becomes an object: SAM 2.1 segments it in the views that see it best, and the masks are lifted to 3D. Its verdict comes from its points. It is *confirmed* when the other video looked through at least half of the points it could check, *dismissed* below a fifth, and *unverified* in between or when the other video could check less than a fifth of it; each comes with the reason (hidden, out of view, or partly still there). For hidden or unseen objects it also gives the spot to film from.
4. **Moves.** A disappeared and an appeared object that look alike (DINOv2 features) are matched, and a rigid fit gives the distance and the turn. When only one end has evidence, the other video is searched for the object at a place the first never filmed.
5. **Words from geometry.** The support is found in 3D (the static points under the object) and named by SigLIP 2 from a crop of it; SigLIP 2 also names the object from its masked crops, and the colour comes from the masked pixels. Height, distance, turn and blocked floor area come from the 3D points, and the floor is only ever named by geometry. No language model decides what changed.

## Results

Report set: 18 indoor test pairs of the [SceneDiff benchmark](https://huggingface.co/datasets/yuqun/SceneDiff), with 50 changed objects (15 of them lower than 8 cm). In the 3D scores, a change matches a ground-truth object (lifted into 3D with the run's own depth) when a quarter of its points lie within 5 cm of it, one-to-one.

| | roomdiff | naive differencing |
|---|---|---|
| changes reported | 54 (42 confirmed) | 435 |
| recall, all objects | 0.62 | 0.72 |
| recall, objects 8 cm or taller (35) | 0.77 | 0.94 |
| precision | 0.57 (confirmed only: 0.62) | 0.08 |
| false changes where the other video never looked | 1 | 183 |
| correct type (moved, added, removed) among found | 0.71 | 0.47 |
| unchanged room: changes reported on 6 half-video controls | 0 | 37 |
| static background seen through (false-free rate), mean / median | 1.2 % / 0.2 % | |
| SceneDiff official: px/im IoU, obj/im AP, obj/sc AP | 42.4, 27.6, 15.7 | |

Naive differencing calls a point changed when the other cloud has nothing within 5 cm, after the same joint reconstruction (as in VGGT-CD), with the same clustering. The half-video controls cut one video in two: the same unchanged room, filmed along two stretches that overlap by 8 to 84 %. The official SceneDiff evaluator was run on masks exported for the same 18 pairs; SceneDiff reports 39.1, 43.8 and 22.8 on its full SD-V test set ([arXiv 2512.16908](https://arxiv.org/abs/2512.16908)), so the numbers are not directly comparable. Of the 9 type errors among the 31 found objects, 7 are moves where only one end left evidence and the search did not find the other, so they read as a removal or an addition; 2 are a removal or an addition paired with a similar object and reported as a move.

Detection thresholds were set on 4 validation pairs (living_room_1_2, bedroom_2_4, lounge_1_2, living_room_35_36) and the living room example, never on the report pairs. The labeller (model, prompts, crop and hedge thresholds) was chosen on ground-truth crops from all 24 pairs, report pairs included, with a vocabulary that holds every SceneDiff label, so its 87.5 % top-1 on those crops is optimistic. The worst false-free rate is study_1_study_2 (7 % and 19 %): its two videos share one corner of the room, and two walls of the before reconstruction do not line up with the after video.

**Reproducing the numbers.** Download `scenediff_benchmark.zip` (11.9 GB) from the dataset page and unpack it so that `data/scenediff_benchmark/data/<pair>/` and `data/scenediff_benchmark/splits/` exist. Then `uv run python scripts/evaluate.py --set report --official` writes the 3D scores and `object_masks.pkl` per pair (about 3 minutes per pair on an M2 Max), and `--control --pairs lounge_1_lounge_2 living_room_1_living_room_2 living_room_13_living_room_14 bedroom_6_bedroom_7 office_1_office_2 study_1_study_2` runs the six half-video controls (2 tuning pairs, 4 report pairs). The official metrics come from SceneDiff's `scripts/evaluate_multiview.py` (github.com/yuqunw/scene_diff, needs faiss-cpu and pycocotools) with `--pred_dir runs/eval/official`.

## Using your own videos

Record two videos of the same room, 15 to 30 s each, in the same light, and move, add or remove a few objects in between.

- 1x lens, 1080p, HDR Video off; lock exposure, focus and white balance (press and hold), and turn on Lock Camera.
- Walk slowly with the camera facing into the room. Do not pivot on the spot: the reconstruction needs the camera to move.
- Film the second video along a different path, but cover the same objects.
- `uv run roomdiff run before.MOV after.MOV --out runs/mine`. iPhone `.MOV` files (HEVC, rotated) work directly. The default of 28 keyframes per video was tuned on 8 to 19 s clips; for videos longer than about 20 s, `--config.frames-per-video 40` samples them more densely (the reconstruction of 80 frames takes 2 minutes and 14 GB of GPU memory on the M2 Max). To rebuild the committed example and the README media from your pair: `uv run python scripts/make_example.py before.MOV after.MOV --name my_room`.

## Limitations

- Objects lower than about 8 cm (remotes, phones, books, cables) sit inside the depth noise and are rarely found: 27 % recall on the report set, against 77 % for taller objects.
- Scale comes from Pi3X and is approximate (about 15 %); distances are given to 0.1 m.
- Deformable objects (curtains, blankets, clothes) are often split or reported as removals.
- The turn of a moved object is reported only when the rigid fit is unambiguous.
- Labels come from a closed vocabulary of 260 indoor objects; names scored below 0.35 are hedged ("an object, possibly a stool"), and a confident name can still be wrong (the tote bag above).
- Mirrors and windows can create spurious geometry; candidates seen from farther than 6 m are ignored.

## Credits and licences

The code in this repository is MIT. Model weights are downloaded at run time from Hugging Face, each pinned to a revision:

| Model | Use | Licence |
|---|---|---|
| Pi3X (yyfz233/Pi3X) | joint reconstruction | weights CC BY-NC 4.0 (non-commercial), code BSD-3 |
| SAM 2.1 hiera-base-plus | object masks | Apache-2.0 |
| DINOv2 base | matching moved objects | Apache-2.0 |
| SigLIP 2 large (patch16-384) | labels | Apache-2.0 |
| three.js r147 (bundled) | viewer | MIT |

Example videos: SceneDiff benchmark (Wu et al., arXiv 2512.16908), MIT. The ray test follows SceneDiff's `foreground_distance`, Khronos (Schmid et al., RSS 2024) and Fehr et al. (ICRA 2017). [DESIGN.md](DESIGN.md) explains the design choices.
