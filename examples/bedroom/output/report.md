# What changed in the room

Between before.mp4 and after.mp4 roomdiff found 3 changes: 1 moved, 2 added.
Unverified means not enough evidence either way: the other video saw too little of it, or part of it is still there. The camera was held 1.53 m above the
floor; positions are in metres with z up and the floor at z = 0.

![Top view of the changes](floorplan.png)

| id | | type | label | status | sentence |
|---|---|---|---|---|---|
| 1 | → | moved | plastic bag | confirmed | The plastic bag moved 0.3 m along the desk. |
| 2 | + | added | stuffed animal | confirmed | A yellow stuffed animal appeared on the shelf (84 cm up). |
| 3 | + | added | water bottle | confirmed | A black water bottle appeared on the desk (85 cm up). |

## → 1: plastic bag, moved

The plastic bag moved 0.3 m along the desk.

- Status: confirmed (the after video looked through 96 % of the points it could check).
- Motion: 0.33 m.
- Evidence: 575 points: 61 % looked through, 3 % still seen, 16 % hidden, 19 % not in view.

<img src="cards/1_before.jpg" width="49%"> <img src="cards/1_after.jpg" width="49%">

## + 2: stuffed animal, added

A yellow stuffed animal appeared on the shelf (84 cm up).

- Status: confirmed (the before video looked through 68 % of the points it could check).
- Evidence: 218 points: 64 % looked through, 31 % still seen, 5 % not in view.

<img src="cards/2_before.jpg" width="49%"> <img src="cards/2_after.jpg" width="49%">

## + 3: water bottle, added

A black water bottle appeared on the desk (85 cm up).

- Status: confirmed (the before video looked through 100 % of the points it could check).
- Evidence: 146 points: 100 % looked through.

<img src="cards/3_before.jpg" width="49%"> <img src="cards/3_after.jpg" width="49%">

## Coverage

The other video checked 82 % of all surface points. The rest was seen by one video only: 10 % only in the before video and 8 % only in the after video; the top view tints those parts.

The points the other video looked through belong to: 96 % confirmed changes and 4 % no candidate.

## Method

Both videos were reconstructed in one pass into a shared metric frame with z up and the floor at z = 0. A
surface counts as removed or added only where the other video looked through the place it occupied, and each
change keeps the share of its points that were looked through, still seen, hidden or not in view.
