import numpy as np
from scipy.spatial.transform import Rotation
from test_evidence import STATIC, STRUCTURE, look_at, make_views

from roomdiff.world import transform, world_frame


def test_gravity_from_pitched_cameras_that_never_turn():
    # Cameras 1.4 m up, pitched 35 degrees down, sliding sideways: their x axes are all parallel, so
    # only the surface normals can give the up direction.
    poses = [
        look_at((x, -2.0, 1.4), (x, 0.0, 1.4 - 2.0 * np.tan(np.radians(35)))) for x in np.linspace(-1, 1, 6)
    ]
    views, _ = make_views({**STRUCTURE, **STATIC}, poses)
    tilt = np.eye(4)
    tilt[:3, :3] = Rotation.from_euler("xyz", [20, -15, 40], degrees=True).as_matrix()
    views = transform(views, tilt.astype(np.float32))
    frame = world_frame(views, views)
    up = np.linalg.inv(frame.T[:3, :3] @ tilt[:3, :3])[:, 2]
    assert np.degrees(np.arccos(abs(up[2]))) < 3
    assert frame.floor_found and abs(frame.camera_height - 1.4) < 0.05
