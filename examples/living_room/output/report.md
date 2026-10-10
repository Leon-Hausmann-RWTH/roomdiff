# What changed in the room

Between before.mp4 and after.mp4 roomdiff found 4 changes: 1 moved, 3 removed; 1 of them unverified.
Unverified means not enough evidence either way: the other video saw too little of it, or part of it is still
there. The camera was held 1.49 m above the floor; positions are in metres with z
up and the floor at z = 0.

![Top view of the changes](floorplan.png)

| id | | type | label | status | sentence |
|---|---|---|---|---|---|
| 1 | → | moved | cushion | confirmed | The grey cushion moved 1.6 m along the sofa. |
| 2 | − | removed | stool | confirmed | The white stool on the floor was removed. |
| 3 | − | removed | blanket | confirmed | The object on the sofa (83 cm up), possibly a blanket, was removed. |
| 4 | ? | removed | shoe | unverified | The beige shoe on the shelf (50 cm up) may have been removed. |

## → 1: cushion, moved

The grey cushion moved 1.6 m along the sofa.

- Status: confirmed (the after video looked through 70 % of the points it could check).
- Motion: 1.56 m.
- Evidence: 888 points: 70 % looked through, 30 % still seen.

<img src="cards/1_before.jpg" width="49%"> <img src="cards/1_after.jpg" width="49%">

## − 2: stool, removed

The white stool on the floor was removed.

- Status: confirmed (the after video looked through 93 % of the points it could check).
- Evidence: 462 points: 92 % looked through, 7 % still seen, 1 % not in view.

<img src="cards/2_before.jpg" width="49%"> <img src="cards/2_after.jpg" width="49%">

## − 3: blanket, removed

The object on the sofa (83 cm up), possibly a blanket, was removed.

- Status: confirmed (the after video looked through 87 % of the points it could check).
- Evidence: 221 points: 69 % looked through, 10 % still seen, 20 % not in view.

<img src="cards/3_before.jpg" width="49%"> <img src="cards/3_after.jpg" width="49%">

## ? 4: shoe, removed

The beige shoe on the shelf (50 cm up) may have been removed.

- Status: unverified (the after video looked through 38 % of it and still saw 62 %).
- Evidence: 122 points: 38 % looked through, 62 % still seen.

<img src="cards/4_before.jpg" width="49%"> <img src="cards/4_after.jpg" width="49%">

## Coverage

The other video checked 70 % of all surface points; 7 % were hidden from it. The rest was seen by one video only: 5 % only in the before video and 17 % only in the after video; the top view tints those parts.

The points the other video looked through belong to: 88 % confirmed changes, 2 % unverified changes, 8 % dismissed candidates and 3 % no candidate.

## Method

Both videos were reconstructed in one pass into a shared metric frame with z up and the floor at z = 0. A
surface counts as removed or added only where the other video looked through the place it occupied, and each
change keeps the share of its points that were looked through, still seen, hidden or not in view.
