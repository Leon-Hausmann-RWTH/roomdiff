from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import av
import torch
import tyro

from .config import Config


@dataclass
class Run:
    """Compare two videos of the same room and report what was moved, added or removed."""

    before: Annotated[Path, tyro.conf.Positional, tyro.conf.arg(metavar="BEFORE")]
    """Video recorded before the changes (.mp4, .mov)."""
    after: Annotated[Path, tyro.conf.Positional, tyro.conf.arg(metavar="AFTER")]
    """Video recorded after the changes."""
    out: Path = Path("runs/latest")
    """Output folder for changes.json, viewer.html, floorplan.png, report.md and scene.ply."""
    config: Config = field(default_factory=Config)
    """Settings; frames_per_video, device and cache_dir are the ones to change."""


def check_video(path: Path, cfg: Config) -> None:
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise SystemExit(f"no video stream in {path}")
        stream = container.streams.video[0]
        if 0 < stream.frames < cfg.frames_per_video:
            raise SystemExit(
                f"{path} has {stream.frames} frames; {cfg.frames_per_video} keyframes are needed"
            )
        # PQ or HLG transfer: iPhones film HDR by default, and without tone mapping its colours wash out.
        if stream.codec_context.color_trc in (16, 18):
            print(f"warning: {path} is HDR video; record with HDR Video off for true colours")


def main() -> None:
    args = tyro.extras.subcommand_cli_from_dict({"run": Run})
    cfg = args.config
    for p in (args.before, args.after):
        check_video(p, cfg)
    if args.before.resolve() == args.after.resolve():
        print("warning: before and after are the same file; expect no changes")
    if args.out.exists() and not args.out.is_dir():
        raise SystemExit(f"not a folder: {args.out}")
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if not available.get(cfg.device, True):
        raise SystemExit(f"--config.device {cfg.device} is not available on this machine")
    from .pipeline import analyse, write_outputs

    result = write_outputs(analyse(args.before, args.after, cfg), args.out, cfg)
    for c in result["changes"]:
        print(f"{c['id']} {c['type']}, {c['status']}: {c['sentence']}")
    print(f"wrote {args.out}/; open {args.out / 'viewer.html'} to see the changes in 3D")


if __name__ == "__main__":
    main()
