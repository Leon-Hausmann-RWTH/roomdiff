from dataclasses import dataclass, field
from pathlib import Path

import tyro

from .config import Config


@dataclass
class Run:
    """Compare two videos of the same room and report what was moved, added or removed."""

    before: tyro.conf.Positional[Path]
    """Video recorded before the changes (.mp4, .mov)."""
    after: tyro.conf.Positional[Path]
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

    write_outputs(analyse(args.before, args.after, args.config), args.out)
    print(f"wrote {args.out}/")
