# Design note

## Idea

A surface counts as gone only where the other video looked through the place it occupied: the other video's depth along those rays lies beyond it. This is evidence of absence, as in Fehr et al. (ICRA 2017) and Khronos (RSS 2024); the per-pixel test is SceneDiff's `foreground_distance`. A nearer surface means the spot was hidden, and too few views mean it was not checked. Neither can produce a change. The same test runs in both directions. The rest of the pipeline counts these votes per point and per object.

SceneDiff also reconstructs both videos jointly and uses SAM and DINO, but it scores unobserved pixels as unchanged and outputs 2D masks. OASIS-Map keeps an unknown state per object but needs LiDAR poses. roomdiff adds votes over several camera positions, a verdict with a reason for each object, sentences built from measured geometry, and outputs that mark what was not checked.

## Decisions

1. **One joint reconstruction.** Pi3X reconstructs both videos in one pass, so no alignment step is needed. On the test pairs 1.2 % of the static background comes out free (median 0.2 %). Splat-based change detection (3DGS-CD) needs a dense before capture and scores 0.5 obj/sc AP on SceneDiff.
2. **Ray evidence instead of nearest-neighbour differencing.** Differencing reports every region the other video missed: 435 changes at precision 0.08, against 54 at 0.57. The price is recall, 0.62 against 0.72.
3. **Votes from two camera positions.** Neighbouring keyframes see nearly the same ray, so one burst of bad depth looks like many votes. Requiring two positions 25 cm apart removed 33 of 45 false candidates on the validation pairs and kept 14 of 17 objects.
4. **Verdicts with reasons.** An object is confirmed, unverified (partly still there, too flat to tell from its support, or seen too little) or dismissed. A removal or addition without a partner gets one search of the other video by appearance. The match counts only where the first video saw that place empty, because elsewhere an unchanged twin looks the same. The search raised the share of correct types from 0.45 to 0.61.
5. **Models only name things.** Used as change detectors, VLMs reach 7.1 obj/im AP in SceneDiff's comparison, so none decides anything here. SigLIP 2 names masked crops in-process. Heights, distances and turns come from the points.

## More numbers

- Of 27 ground-truth moves, 8 are found as moves and 7 of those have both ends right, with a median distance error of 0.06 m. roomdiff reports 16 moves in all.
- 12 of the 31 found objects have the wrong type. 10 are moves read as a removal or an addition; in 8 of them the other end was not found.
- SceneDiff's evaluator needs a 2D mask IoU of 0.5 and counts 16 of 50 objects as found. On px/im IoU, obj/im AP and obj/sc AP roomdiff scores 42.7, 28.0 and 15.7 on these 18 pairs. SceneDiff's own method scores 39.1, 43.8 and 22.8 on its full test set, so the numbers are not directly comparable.
- The labeller was chosen on ground-truth crops from all 23 SceneDiff pairs used here, so its 87.5 % top-1 is optimistic.

## Failure modes

Objects under 8 cm stay inside the depth band of 4 cm plus 3 % of the depth (27 % recall). A camera that moves less than about 1 m loses changes to the two-position rule. Soft objects and slightly moved furniture split. Walls misalign when the videos share little of the room (19 % false free on study_1_study_2). Names can be wrong. Next steps: an appearance cue for flat objects, and filming again where the viewer marks areas not checked.

## Reproduce

Unpack `scenediff_benchmark.zip` from the dataset page into `data/scenediff_benchmark/`. The commands below score the 18 report pairs, run the controls (each pair's first video split in two) and run SceneDiff's evaluator:

```bash
uv run python scripts/evaluate.py --set report --official
uv run python scripts/evaluate.py --control --pairs lounge_1_lounge_2 living_room_1_living_room_2 living_room_13_living_room_14 bedroom_6_bedroom_7 office_1_office_2 study_1_study_2
git clone https://github.com/yuqunw/scene_diff ../scene_diff   # tested at commit 62dcbd2
uv run --with faiss-cpu --with pycocotools --with opencv-python-headless --with matplotlib python ../scene_diff/scripts/evaluate_multiview.py --pred_dir runs/eval/official --output_path runs/eval/official_result.txt
```

References: SceneDiff (arXiv 2512.16908); Pi3 (arXiv 2507.13347); Khronos (arXiv 2402.13817); Fehr et al. (ICRA 2017); OASIS-Map (arXiv 2607.14899); 3DGS-CD (arXiv 2411.03706); SAM 2 (arXiv 2408.00714); DINOv2 (arXiv 2304.07193); SigLIP 2 (arXiv 2502.14786).
