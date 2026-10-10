# Design note

## Problem

Two short handheld videos of one room, filmed along different paths, with a few objects moved, added or removed in between. The system has to say which objects changed, where they are in 3D, and what they are. It must not call something a change only because the second video did not look there, looked from another side, or had something in the way.

## Principle: change needs evidence

Both videos are reconstructed together, so every keyframe has depth and a pose in one metric frame. A surface the before video saw counts as gone only if a frame of the after video looked *through* the place where it was: the after depth along that ray lies beyond the point. That is evidence of absence (Fehr et al., ICRA 2017; Khronos, RSS 2024; the per-pixel test is SceneDiff's `foreground_distance`). A nearer after depth means something blocks the view, and a place no after frame reaches was never filmed. Neither carries information, so neither can produce a change. The same test runs from after to before for additions. Everything else follows from counting these votes per point and per object, and from reporting what stayed undecided instead of hiding it.

## Decisions

1. **One joint feed-forward reconstruction, not per-scene splats or separate passes.** Splat-based detectors assume a dense before capture; on SceneDiff's casual pairs 3DGS-CD reaches 0.5 obj/sc AP against 22.8. A joint Pi3X pass needs no alignment step. Across the report pairs, 1.2 % of the static background comes out "free" (the false-free rate), and the top views show no doubled walls. Pi3X was chosen over Pi3 for its metric scale, which the sentences and the absolute thresholds use: on 16 of the 18 pairs the camera comes out 1.2 to 1.8 m above the floor; the other two film a desk and a bed from close up, where the lowest large plane is not the floor.
2. **Ray evidence, not nearest-neighbour differencing.** Differencing calls every region one video did not cover a change. On the report pairs it reports 435 changes at precision 0.08, 183 of them in places the other video never saw; the ray test reports 53 at precision 0.58, with 1 there. On six half-video controls (an unchanged room) differencing reports 37 changes and roomdiff none.
3. **Votes from distinct camera positions.** Neighbouring keyframes see nearly the same ray, so one burst of bad depth looks like many votes. Requiring that cameras in two places (25 cm apart) looked through at least 30 % of a candidate removed 33 of 45 false candidates on the tuning pairs and kept 14 of 17 detectable objects. A thickness rule and a free-margin rule were tried and dropped: the first rejects objects seen from one side, the second did not separate true from false candidates.
4. **Verdicts with reasons, and a place to look.** Each object is confirmed, unverified (hidden, out of view, or partly still there) or dismissed, from the states of its points above the contact zone, which always tests as seen. An unverified object that was hidden or out of view comes with the camera pose that would settle it. A confirmed removal without a partner triggers a search of the other video for the object at a place the first never filmed; on the report pairs it raised the share of found changes with the right type from 0.45 to 0.68.
5. **Language from geometry; models only name.** VLMs are poor change detectors (7.1 against 43.8 obj/im AP in SceneDiff's comparison), so none decides anything here. SigLIP 2 names masked crops in-process (87.5 % top-1 on 80 SceneDiff objects, no server needed), and every spatial word (support surface, height, distance, turn, blocked floor) comes from the 3D measurements.

## Results

On 18 indoor SceneDiff test pairs (50 changed objects), roomdiff finds 62 % of all objects and 77 % of those 8 cm or taller, at precision 0.58; the official evaluator gives 39.2 px/im IoU, 27.8 obj/im AP and 17.5 obj/sc AP (SceneDiff: 39.1, 43.8 and 22.8 on its full set) on this subset. A pair takes about 2 minutes on an M2 Max. The full table is in the README.

## Failure modes

- Flat objects (under about 8 cm) stay within the depth band: 27 % recall.
- A move whose new place the other video never saw is a removal (or an addition) unless the search finds the other end.
- Deformable objects (curtains, blankets) split or come out as removals.
- The turn of a partly seen object is ambiguous and then not reported.
- Labels can be confidently wrong (a tote bag named a shoe), and support names can disagree with the furniture.

## Next steps

- An appearance cue (DINOv2 feature change on points that test as seen) for flat objects, which is where most of the recall is lost.
- Use the suggested viewpoints in a loop: film again from there and re-run, as a robot would.

## References

SceneDiff (Wu et al., arXiv 2512.16908); Pi3 and Pi3X (github.com/yyfz/Pi3); Khronos (Schmid et al., RSS 2024); Fehr et al., ICRA 2017; Herbst et al., ICRA 2011; OASIS-Map (Oh et al., arXiv 2607.14899); VGGT-CD (arXiv 2605.16859); SAM 2 (Ravi et al., 2024); DINOv2 (Oquab et al., 2024); SigLIP 2 (Tschannen et al., 2025).
