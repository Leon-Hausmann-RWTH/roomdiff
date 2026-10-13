# Bedroom example

SceneDiff pair bedroom_32_bedroom_33 (MIT licence), video as recorded.

- Outputs: [report](output/report.md), [changes.json](output/changes.json), [top view](output/floorplan.png). The viewer `output/viewer.html` opens after download.
- Rerun: `uv run roomdiff run examples/bedroom/before.mp4 examples/bedroom/after.mp4 --out runs/bedroom`
- Ground truth: 4 changes. Found: the plastic bag (moved 0.4 m), the stuffed animal and the water bottle (added). Missed: a flat cable on the desk. The stuffed animal stands on a small table, which roomdiff calls a shelf, and the bottle stands on the desk, not the windowsill.
