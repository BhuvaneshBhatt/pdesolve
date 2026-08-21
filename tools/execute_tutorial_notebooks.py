#!/usr/bin/env python3
"""Execute the maintained PDESolve PDESolve documentation tutorial notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="execute notebooks without rewriting them"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="per-cell execution timeout in seconds"
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="optional notebook filenames under notebooks/tutorials",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    tutorial_dir = root / "notebooks" / "tutorials"
    paths = (
        [tutorial_dir / name for name in args.notebooks]
        if args.notebooks
        else sorted(tutorial_dir.glob("*.ipynb"))
    )
    if not paths:
        raise SystemExit("No tutorial notebooks found")

    failures: list[tuple[Path, Exception]] = []
    for path in paths:
        print(f"[tutorial] {path.name}", flush=True)
        nb = nbformat.read(path, as_version=4)
        try:
            client = NotebookClient(
                nb,
                timeout=args.timeout,
                startup_timeout=60,
                kernel_name="python3",
                resources={"metadata": {"path": str(root)}},
            )
            client.execute()
            if not args.check:
                nbformat.write(nb, path)
            print("  ok", flush=True)
        except Exception as exc:  # execution failures should be reported together
            failures.append((path, exc))
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"{len(failures)} tutorial notebook(s) failed", file=sys.stderr)
        return 1
    print(f"Executed {len(paths)} tutorial notebook(s) successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
