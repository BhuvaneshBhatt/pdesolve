"""Generate or verify the package-level public API snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pdesolve

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = ROOT / "test_reports" / "public-api-audit.json"


def current_inventory() -> dict[str, object]:
    exports = list(pdesolve.__all__)
    return {
        "version": pdesolve.__version__,
        "export_count": len(exports),
        "exports": exports,
        "policy": "Package-level exports are reviewed intentionally; additions and removals require an API snapshot update.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory = current_inventory()

    if args.check:
        try:
            recorded = json.loads(args.output.read_text())
        except FileNotFoundError:
            print(f"Missing public API snapshot: {args.output}")
            return 1
        if recorded != inventory:
            print(
                "Public API snapshot differs from pdesolve.__all__. Regenerate it intentionally after reviewing the API change."
            )
            return 1
        print(f"Public API snapshot OK: {inventory['export_count']} exports.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Wrote {args.output} ({inventory['export_count']} exports).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
