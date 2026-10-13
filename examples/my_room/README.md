# My room example

Two 16 s videos of a living room, filmed with an iPhone (1080p) for this project. Video as recorded. HDR Video could not be switched off, so the report warns about colours.

- Outputs: [report](output/report.md), [changes.json](output/changes.json), [top view](output/floorplan.png). The viewer `output/viewer.html` opens after download.
- Rerun: `uv run roomdiff run examples/my_room/before.mp4 examples/my_room/after.mp4 --out runs/my_room`
- Changes made between the videos: the wooden chair moved from the middle of the rug to the window and turned, the white stool and a 40 cm candle holder were removed, a backpack was put on the floor. roomdiff finds all four with the right type. It also reports the chair's reflection in the glass door as an added object and a 20-point cluster near the recliner as a removal.
