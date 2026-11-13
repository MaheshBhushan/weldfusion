"""Command line entry point: `weldfusion <stage>` runs one pipeline stage (or `all`)."""

from __future__ import annotations

import argparse

from . import correlate, download, export, features, fuse, train
from .paths import Paths, load_params

STAGES = {
    "fuse": fuse.run,
    "features": features.run,
    "correlate": correlate.run,
    "train": train.run,
    "export": export.run,
}


def run_all(paths: Paths, params: dict, fetch: bool = True) -> None:
    if fetch:
        download.run(paths)
    for stage in STAGES.values():
        stage(paths, params)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="weldfusion", description=__doc__)
    ap.add_argument("stage", choices=["download", *STAGES, "all"])
    ap.add_argument("--root", default=".", help="project root holding data/, models/, reports/")
    ap.add_argument("--raw", default=None, help="raw data directory (default: <root>/data/raw)")
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args(argv)

    paths = Paths.at(args.root, args.raw)
    params = load_params(args.params)
    if args.stage == "download":
        download.run(paths)
    elif args.stage == "all":
        run_all(paths, params, fetch=args.raw is None)
    else:
        STAGES[args.stage](paths, params)


if __name__ == "__main__":
    main()
