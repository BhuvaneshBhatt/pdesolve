"""Generate a machine-readable documentation inventory for PDESolve.

This script intentionally inspects the installed source tree rather than keeping a
second handwritten list of exports and executable method keys.
"""

from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import re
from collections import Counter
from typing import Any

import pdesolve
from pdesolve import solver_execution

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _signature(obj: Any) -> str:
    if not callable(obj):
        return ""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def _kind(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj):
        return "function"
    if callable(obj):
        return "callable"
    return type(obj).__name__


def build_inventory(root: pathlib.Path) -> dict[str, Any]:
    exports = []
    for name in pdesolve.__all__:
        obj = getattr(pdesolve, name)
        exports.append(
            {
                "name": name,
                "module": getattr(obj, "__module__", ""),
                "kind": _kind(obj),
                "signature": _signature(obj),
            }
        )

    test_files = sorted((root / "tests").glob("test_*.py"))
    test_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in test_files
    )
    method_keys = sorted(solver_execution._METHOD_REGISTRY)
    method_test_mentions = {
        method: len(re.findall(re.escape(method), test_text)) for method in method_keys
    }

    return {
        "package_version": getattr(pdesolve, "__version__", None),
        "top_level_export_count": len(exports),
        "top_level_exports": exports,
        "exports_by_module": dict(
            sorted(Counter(item["module"] for item in exports).items())
        ),
        "canonical_execution_method_count": len(method_keys),
        "canonical_execution_methods": method_keys,
        "method_test_mentions": method_test_mentions,
        "test_file_count": len(test_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of tools/).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Write JSON to this path instead of stdout.",
    )
    args = parser.parse_args()

    inventory = build_inventory(args.root.resolve())
    text = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
