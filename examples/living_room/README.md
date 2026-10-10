# Living room example

SceneDiff pair living_room_17_living_room_18 (Wu et al., arXiv 2512.16908, huggingface.co/datasets/yuqun/SceneDiff), MIT licence, re-encoded to 1280 px H.264 by `scripts/make_example.py`.

- Outputs: [report](output/report.md), [changes.json](output/changes.json), [top view](output/floorplan.png), [scene.ply](output/scene.ply). The 3D viewer `output/viewer.html` (the same file as `docs/index.html`) opens after download.
- Rerun: `uv run roomdiff run examples/living_room/before.mp4 examples/living_room/after.mp4 --out runs/living_room`
- Ground truth: 7 changes. Found: the cushion (moved), the stool and the coat (removed), the tote bag (unverified). Missed: the bread (6 cm tall), the tissue box (the before video never looked through its place) and the remote (1 cm).
