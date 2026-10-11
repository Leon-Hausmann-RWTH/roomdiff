# What changed in the room

Between before.mp4 and after.mp4 roomdiff found 3 changes: 1 moved, 2 added.
The camera was held 1.53 m above the floor. Positions and boxes are in changes.json.

![Top view of the changes](floorplan.png)

| id | | type | name | status | sentence |
|---|---|---|---|---|---|
| 1 | → | moved | plastic bag | confirmed | The plastic bag moved 0.3 m along the desk. |
| 2 | + | added | stuffed animal | confirmed | A yellow stuffed animal appeared on the shelf (84 cm up). |
| 3 | + | added | water bottle | confirmed | A black water bottle appeared on the desk (85 cm up). |

## → 1: plastic bag, moved

The plastic bag moved 0.3 m along the desk.

- Status: confirmed (the after video looked through 97 % of the points it could check).
- Motion: 0.33 m.
- Evidence: 517 points: 64 % looked through, 2 % surface seen, 14 % hidden, 21 % too few views.

<img src="cards/1_before.jpg" width="49%"> <img src="cards/1_after.jpg" width="49%">

## + 2: stuffed animal, added

A yellow stuffed animal appeared on the shelf (84 cm up).

- Status: confirmed (the before video looked through 64 % of the points it could check).
- Evidence: 232 points: 61 % looked through, 34 % surface seen, 5 % too few views.

<img src="cards/2_before.jpg" width="49%"> <img src="cards/2_after.jpg" width="49%">

## + 3: water bottle, added

A black water bottle appeared on the desk (85 cm up).

- Status: confirmed (the before video looked through 100 % of the points it could check).
- Evidence: 135 points: 100 % looked through.

<img src="cards/3_before.jpg" width="49%"> <img src="cards/3_after.jpg" width="49%">

## Coverage

The other video checked 72 % of the surface points. It had 10 % hidden and saw 18 % in fewer than three frames; the viewer can tint both.

## Method

Both videos are reconstructed together in one metric frame. A surface counts as removed or added only where
the other video looked through the place it occupied. Each change lists how many of its points the other video
looked through, saw as a surface, had hidden, or saw too rarely to decide.
