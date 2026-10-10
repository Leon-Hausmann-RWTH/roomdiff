# Bedroom example

SceneDiff pair bedroom_32_bedroom_33 (Wu et al., arXiv 2512.16908, huggingface.co/datasets/yuqun/SceneDiff), MIT licence, re-encoded to 1280 px H.264 by `scripts/make_example.py`.

- Outputs: [report](output/report.md), [changes.json](output/changes.json), [top view](output/floorplan.png), [scene.ply](output/scene.ply). The 3D viewer `output/viewer.html` (the same file as `docs/bedroom.html`) opens after download.
- Rerun: `uv run roomdiff run examples/bedroom/before.mp4 examples/bedroom/after.mp4 --out runs/bedroom`
- Ground truth: 4 changes. Found: the plastic bag (moved 0.3 m), the stuffed animal and the water bottle (added). Missed: a white cable lying flat on the desk.
