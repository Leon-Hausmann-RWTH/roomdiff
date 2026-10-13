# Living room example

SceneDiff pair living_room_17_living_room_18 (MIT licence), video as recorded.

- Outputs: [report](output/report.md), [changes.json](output/changes.json), [top view](output/floorplan.png). The viewer `output/viewer.html` opens after download.
- Rerun: `uv run roomdiff run examples/living_room/before.mp4 examples/living_room/after.mp4 --out runs/living_room`
- Ground truth: 7 changes. Found: the cushion (moved), the stool and the coat (removed, named a blanket), and the tote bag, which moved but is reported as a possible removal of a slipper. Missed: the bread, the tissue box and the remote.
