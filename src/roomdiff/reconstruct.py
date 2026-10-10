import contextlib
import functools
import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from .config import Config, free_device, pick_device
from .geometry import Views, depth_edges
from .video import Keyframes, probe, recon_size, resize, select

PI3X_REPO, PI3X_REVISION = "yyfz233/Pi3X", "bb1deea4d7423de5b30691739cb451a3f57dc1d5"
ATTENTION_BLOCK = 8192  # query rows per attention call; keeps each GPU dispatch under the macOS watchdog


@dataclass
class Recon:
    keys: tuple[Keyframes, Keyframes]
    depth: np.ndarray  # (N, H, W) float32 metres, frames of video A then video B
    conf: np.ndarray  # (N, H, W) float32 in [0, 1]
    c2w: np.ndarray  # (N, 4, 4) float32 world from camera, OpenCV axes
    K: np.ndarray  # (N, 3, 3) float32, one shared focal length per video
    fit_px: tuple[float, float]  # median pinhole reprojection residual per video, pixels


def chunked_attention(sdpa, q, k, v, **kw):
    if q.shape[-2] <= ATTENTION_BLOCK:
        return sdpa(q, k, v, **kw)
    # Exact: softmax is computed per query row.
    return torch.cat(
        [
            sdpa(q[..., i : i + ATTENTION_BLOCK, :], k, v, **kw)
            for i in range(0, q.shape[-2], ATTENTION_BLOCK)
        ],
        -2,
    )


def chunked_encoder(forward, x, is_training=True):
    # The DINOv2 encoder works per frame; chunks bound the memory of its explicit attention.
    tokens = [forward(x[i : i + 8], is_training=True)["x_norm_patchtokens"] for i in range(0, len(x), 8)]
    return {"x_norm_patchtokens": torch.cat(tokens)}


def load_pi3x(device: torch.device):
    # Pi3 prints a notice about its optional CUDA RoPE kernel on import; the PyTorch fallback is fine.
    with contextlib.redirect_stdout(io.StringIO()):
        from pi3.models.pi3x import Pi3X

    path = hf_hub_download(PI3X_REPO, "model.safetensors", revision=PI3X_REVISION)
    with torch.device(device):
        model = Pi3X(use_multimodal=False)
    names = set(model.state_dict())
    with safe_open(path, "pt", device=str(device)) as f:
        state = {k: f.get_tensor(k) for k in names}
    model.load_state_dict(state, assign=True)
    model.encoder.forward = functools.partial(chunked_encoder, model.encoder.forward)
    model._chunked_conv_head = functools.partial(type(model)._chunked_conv_head, model, chunk_size=8)
    return model.eval()


def run_pi3x(images: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    model = load_pi3x(device)
    import pi3.models.layers.attention as pi3_attention

    x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float().div(255)
    # Pi3 imports scaled_dot_product_attention by name, so the patch must target its module.
    sdpa = functools.partial(chunked_attention, pi3_attention.scaled_dot_product_attention)
    with torch.no_grad(), mock.patch.object(pi3_attention, "scaled_dot_product_attention", sdpa):
        res = model(x[None])
    out = {k: res[k][0].float().cpu().numpy() for k in ("local_points", "camera_poses", "conf")}
    del model, res, x
    free_device(device)
    return out


def fit_focal(local_points: np.ndarray, conf: np.ndarray) -> tuple[float, float]:
    """One pinhole focal length (principal point at the centre) for a stack of point maps."""
    n, h, w, _ = local_points.shape
    v, u = np.mgrid[0:h, 0:w]
    z = local_points[..., 2]
    ok = (conf > 0.5) & (z > 0)
    a = np.concatenate([(local_points[..., 0] / z)[ok], (local_points[..., 1] / z)[ok]])
    b = np.concatenate(
        [np.broadcast_to(u - (w - 1) / 2, ok.shape)[ok], np.broadcast_to(v - (h - 1) / 2, ok.shape)[ok]]
    )
    f = float(a @ b / (a @ a))
    return f, float(np.median(np.abs(f * a - b)))


def cache_key(paths, cfg: Config, required) -> str:
    h = hashlib.sha1()
    for p in paths:
        size = p.stat().st_size
        with open(p, "rb") as fh:
            h.update(fh.read(1 << 20))
            fh.seek(max(size - (1 << 20), 0))
            h.update(fh.read())
        h.update(str(size).encode())
    h.update(repr((cfg.frames_per_video, cfg.pixels_per_frame, required, PI3X_REVISION)).encode())
    return h.hexdigest()[:16]


def reconstruct(path_a: Path, path_b: Path, cfg: Config, required=((), ())) -> Recon:
    t = time.time()
    # One image size for both videos; if one is portrait and the other landscape, meet in the middle.
    w, h = recon_size(float(np.sqrt(probe(path_a)[0] * probe(path_b)[0])), cfg.pixels_per_frame)
    keys = tuple(select(p, cfg.frames_per_video, (w, h), req) for p, req in zip((path_a, path_b), required))
    print(f"keyframes: {len(keys[0].times)} + {len(keys[1].times)} at {w}x{h} ({time.time() - t:.0f} s)")
    # Only geometry is cached; keyframes are re-decoded, which is deterministic and keeps the cache small.
    cache = cfg.cache_dir / f"recon_{cache_key((path_a, path_b), cfg, required)}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"reconstruction: loaded from {cache}")
        return Recon(
            keys, z["depth"], z["conf"].astype(np.float32) / 255, z["c2w"], z["K"], tuple(z["fit_px"])
        )
    t = time.time()
    small = np.concatenate([[resize(im, (w, h)) for im in k.images] for k in keys])
    device = pick_device(cfg.device)
    out = run_pi3x(small, device)
    conf = 1 / (1 + np.exp(-out["conf"][..., 0]))
    na = len(keys[0].times)
    fits = [fit_focal(out["local_points"][s], conf[s]) for s in (slice(0, na), slice(na, None))]
    K = np.zeros((len(conf), 3, 3), np.float32)
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = (w - 1) / 2, (h - 1) / 2, 1
    K[:na, 0, 0] = K[:na, 1, 1] = fits[0][0]
    K[na:, 0, 0] = K[na:, 1, 1] = fits[1][0]
    conf8 = np.round(conf * 255).astype(np.uint8)
    depth = out["local_points"][..., 2].astype(np.float32)
    c2w = out["camera_poses"].astype(np.float32)
    fit_px = (fits[0][1], fits[1][1])
    fit = f"{fit_px[0]:.1f} / {fit_px[1]:.1f} px"
    print(f"reconstruction: {len(conf)} frames on {device.type}, pinhole fit {fit} ({time.time() - t:.0f} s)")
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, depth=depth, conf=conf8, c2w=c2w, K=K, fit_px=np.array(fit_px))
    return Recon(keys, depth, conf8.astype(np.float32) / 255, c2w, K, fit_px)


def to_views(r: Recon, cfg: Config) -> tuple[Views, Views]:
    na = len(r.keys[0].times)
    h, w = r.depth.shape[1:]
    out = []
    for s, k in zip((slice(0, na), slice(na, None)), r.keys):
        depth = r.depth[s].copy()
        bad = (r.conf[s] < cfg.conf_threshold) | np.stack([depth_edges(d, cfg.edge_rtol) for d in depth])
        depth[bad] = 0
        images = np.stack([resize(im, (w, h)) for im in k.images])
        out.append(Views(images, depth, r.K[s], r.c2w[s], k.frame_ids))
    return out[0], out[1]
