from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

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


def main() -> None:
    args = tyro.extras.subcommand_cli_from_dict({"run": Run})
    for p in (args.before, args.after):
        if not p.is_file():
            raise SystemExit(f"not a file: {p}")
    from .pipeline import analyse, write_outputs

    write_outputs(analyse(args.before, args.after, args.config), args.out, args.config)
    print(f"wrote {args.out}/")
