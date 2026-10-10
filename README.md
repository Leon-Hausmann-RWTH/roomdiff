# roomdiff

Compare two phone videos of the same room and find, in 3D, what was moved, added or removed, with a reason for every verdict and an honest list of what could not be checked.

![The detected edits applied to the before scene](docs/media/edit_slider.gif)

## See the results without installing

- [Interactive 3D viewer](docs/index.html) (one self-contained file that opens offline; GitHub Pages serves it from `/docs`)
- [Example report](examples/living_room/output/report.md) and [change file](examples/living_room/output/changes.json)
- Input videos: [before](examples/living_room/before.mp4), [after](examples/living_room/after.mp4) (SceneDiff pair living_room_17/18, MIT)

![Top view of the changes](docs/media/floorplan.png)

## Quickstart

```bash
git clone <this repository> roomdiff && cd roomdiff
uv sync
uv run roomdiff run examples/living_room/before.mp4 examples/living_room/after.mp4 --out runs/demo
```

Runs on Apple Silicon (MPS) or an NVIDIA GPU (CUDA); the device is chosen automatically, and the CPU works but is slow. The first run downloads about 10 GB of model weights. On an M2 Max with 32 GB a pair of 10 s videos takes about 2 minutes once the models are cached (80 s of it the reconstruction) and peaks at 12 GB of GPU memory. On a 16 GB Mac add `--config.frames-per-video 16`.

The output folder holds `changes.json` (one record per change), `viewer.html` (3D viewer with an edit slider), `floorplan.png`, `report.md` with evidence cards, and `scene.ply` (both point clouds with a change id per point).

## How it works

![Ray evidence](docs/media/evidence.svg)

1. **One shared 3D frame.** Keyframes of both videos go through one Pi3X pass, which returns depth and a camera pose for every frame in one metric frame. Gravity comes from the cameras (handheld phones keep roll near zero) and the floor plane is set to z = 0.
2. **Change needs evidence.** Every surface point the before video saw is projected into every after frame. If the after frame saw farther along that ray, it looked through the place the surface used to be: the point is *free*, evidence of absence. If it saw something nearer, the point is *occluded*; if no frame reached it, *unreached*. Neither of these says anything about change. The same test runs from after to before for additions. A cluster of free points counts only if cameras in two or more places looked through it.
3. **Objects and verdicts.** Each candidate is turned into an object with SAM 2.1 in the views that see it best and lifted to 3D. Its verdict comes from the evidence over its points: *confirmed* when the other video looked through most of it, *unverified* with a reason (hidden, out of view, partly still there) otherwise, and dismissed when the other video still saw it.
4. **Moves.** A disappeared and an appeared object that look alike (DINOv2 features) are matched, and a rigid fit over turn and shift gives the distance and the turn.
5. **Words from geometry.** SigLIP 2 names the object from its masked crops and the surface it stands on; colour, support, distance, direction and turn in every sentence come from the 3D measurements. No language model decides what changed.

## Results

Report set: 18 indoor test pairs of the SceneDiff benchmark with 50 changed objects (13 of them lower than 8 cm). 3D scores match a reported change to a ground-truth object, lifted into 3D with the run's own depth, when a quarter of its points lie within 5 cm of it; one-to-one.

| | roomdiff | naive differencing |
|---|---|---|
| changes reported | 53 (41 confirmed) | 436 |
| recall, all objects | 0.62 | 0.72 |
| recall, objects 8 cm or taller | 0.78 | 0.95 |
| precision | 0.59 (confirmed: 0.66) | 0.08 |
| false changes where the other video never looked | 1 | 184 |
| correct type (moved, added, removed) among found | 0.45 | 0.47 |
| static background seen through (false-free rate) | 1.2 % | |
| SceneDiff official, px/im IoU / obj/im AP / obj/sc AP | 39.7 / 26.8 / 14.3 | |

The naive baseline calls a point changed when the other cloud has nothing within 5 cm (nearest-neighbour differencing after the same joint reconstruction). SceneDiff reports 39.1 px/im IoU and 22.8 obj/sc AP on its full SD-V set (ECCV 2026); our numbers are on an 18-pair indoor subset, so they are not directly comparable. On the two halves of single videos (the same unchanged room filmed along two stretches), roomdiff reported no change on 2 of 2 pairs and the naive baseline 4 per pair (more pairs pending).

Tuning used 4 validation pairs (living_room_1_2, bedroom_2_4, lounge_1_2, living_room_35_36) and the showcase pair; the report pairs were never used to set a threshold. `uv run python scripts/evaluate.py --set report` reproduces the table.

## Using your own videos

Record two videos of the same room, 20 to 60 s each, in the same light. Before the second video, move, add or remove a few objects.

- 1x lens, landscape, 1080p; turn off HDR Video; lock exposure, focus and white balance (press and hold), and turn on Lock Camera.
- Walk slowly with the camera facing into the room. Do not pivot on the spot: the 3D reconstruction needs the camera to move.
- Film the second video along a different path, but cover the same objects.
- Then run `uv run roomdiff run before.MOV after.MOV --out runs/mine`. iPhone `.MOV` files (HEVC, rotated) work directly. To rebuild the committed example and the README media from your pair: `uv run python scripts/make_example.py before.MOV after.MOV --name my_room`.

## Limitations

- Objects lower than about 8 cm (remotes, phones, books, cables) sit inside the depth noise and are rarely found: 15 % recall on the report set, against 78 % for taller objects.
- Metric scale comes from Pi3X and is approximate (about 15 %); distances are reported to 0.1 m.
- Deformable objects (curtains, blankets, clothes) often come out as removals or unverified, not as moves.
- The turn of a moved object is reported only when the fit is unambiguous; symmetric objects get none.
- Labels come from a closed vocabulary of 260 indoor objects; names scored below 0.25 are hedged ("an object, possibly a stool").
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

Example data: SceneDiff benchmark (Wu et al., ECCV 2026), MIT. The ray test follows SceneDiff's `foreground_distance`, Khronos (Schmid et al., RSS 2024) and Fehr et al. (ICRA 2017). See [DESIGN.md](DESIGN.md) for the design choices.
