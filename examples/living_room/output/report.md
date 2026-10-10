# What changed in the room

Between before.mp4 and after.mp4 roomdiff found 4 changes (1 moved and 3 removed), 1 of them unverified. An unverified change is
one whose place the other video did not see well enough to decide. The camera was held
1.49 m above the floor; positions are in metres with z up and the floor at z = 0.

![Top view of the changes](floorplan.png)

| id | | type | label | status | sentence |
|---|---|---|---|---|---|
| 1 | → | moved | cushion | confirmed | The grey cushion moved 1.6 m to the left. |
| 2 | − | removed | stool | confirmed | The white stool on the floor was removed. |
| 3 | − | removed | blanket | confirmed | The grey blanket on the sofa (83 cm up) was removed. |
| 4 | ? | removed | shoe | unverified | Possibly: the shoe on the floor was removed. |

## → 1: cushion, moved

The grey cushion moved 1.6 m to the left.

- Status: confirmed, because the after video looked through 73% of its surface.
- Motion: 1.56 m.
- Evidence: of its 1241 points, the other video looked through 72 %, saw 28 %, had 0 % hidden, never had 0 % in view.

<img src="cards/1_before.jpg" width="49%"> <img src="cards/1_after.jpg" width="49%">

## − 2: stool, removed

The white stool on the floor was removed.

- Status: confirmed, because the after video looked through 96% of its surface.
- Evidence: of its 525 points, the other video looked through 92 %, saw 6 %, had 0 % hidden, never had 1 % in view.

<img src="cards/2_before.jpg" width="49%"> <img src="cards/2_after.jpg" width="49%">

## − 3: blanket, removed

The grey blanket on the sofa (83 cm up) was removed.

- Status: confirmed, because the after video looked through 70% of its surface.
- Evidence: of its 282 points, the other video looked through 61 %, saw 17 %, had 0 % hidden, never had 22 % in view.

<img src="cards/3_before.jpg" width="49%"> <img src="cards/3_after.jpg" width="49%">

## ? 4: shoe, removed

Possibly: the shoe on the floor was removed.

- Status: unverified, because the after video looked through 39% of it; the rest is still there.
- Where to film next: where the before video stood at 0.4 s, at (2.04, 0.23, 1.53), looking at (-0.32, -0.72, 0.52).
- Evidence: of its 122 points, the other video looked through 38 %, saw 62 %, had 0 % hidden, never had 0 % in view.

<img src="cards/4_before.jpg" width="49%"> <img src="cards/4_after.jpg" width="49%">

## Areas that were not checked

- U1 (seen only in the before video): The after video never looked at a region 1.0 m by 1.0 m, up to 1.0 m high.
- U2 (seen only in the after video): The before video never looked at a region 1.4 m by 1.4 m, up to 0.9 m high.
- U3 (seen only in the after video): The before video never looked at a region 1.1 m by 0.4 m, up to 1.1 m high.

## Coverage

The other video checked 78 % of all surface points. 5 % were seen only
in the before video and 17 % only in the after video; the tinted parts of the top view
show where.

Of all surface points that the other video looked through, 88 % belong to confirmed
changes, 2 % to unverified changes, 8 % to dismissed candidates and
3 % to no candidate.

## Method

Both videos were reconstructed in one pass into a shared metric frame with z up and the floor at z = 0. A
surface counts as removed or added only where the other video looked through the place it occupied, and each
change keeps the share of its points that were looked through, seen, hidden or never in view.
