"""Refresh API, method, and notebook reference documentation from the source tree."""

from __future__ import annotations

import inspect
import re
from collections import defaultdict
from pathlib import Path

import pdesolve
from pdesolve import solver_execution

ROOT = Path(__file__).resolve().parents[1]


def _signature(obj):
    if not callable(obj):
        return ""
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""


def _kind(obj):
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj):
        return "function"
    return type(obj).__name__


def write_api_inventory():
    groups = defaultdict(list)
    for name in pdesolve.__all__:
        obj = getattr(pdesolve, name)
        module = getattr(obj, "__module__", "pdesolve") or "pdesolve"
        groups[module].append((name, _kind(obj), _signature(obj)))

    out = [
        "# Public API inventory",
        "",
        f"`pdesolve.__all__` contains **{len(pdesolve.__all__)} package-level exports**. "
        "The main solving entry point is `pdesolve(...)`; the remaining exports provide focused solvers, structured models, planning, verification, and inspection APIs.",
        "",
    ]
    for module in sorted(groups):
        out.extend([f"## `{module}`", "", "| Export | Kind | Signature |", "|---|---|---|"])
        for name, kind, sig in sorted(groups[module]):
            sig = sig.replace("|", "\\|")
            out.append(f"| `{name}` | {kind} | `{sig}` |")
        out.append("")
    (ROOT / "docs/public-api-inventory.md").write_text("\n".join(out) + "\n")


def _method_families():
    text = (ROOT / "tests/test_canonical_method_hardening.py").read_text()
    pattern = re.compile(r'"([^"]+)": MethodCoverageSpec\("([^"]+)"')
    return dict(pattern.findall(text))


def write_method_inventory():
    families = _method_families()
    methods = sorted(solver_execution._METHOD_REGISTRY)
    out = [
        "# Methods and solver inventory",
        "",
        "PDESolve distinguishes public solving functions, planner method keys, and internal helper functions. "
        "The canonical method keys below are the values accepted by the execution registry.",
        "",
        f"## Canonical execution methods ({len(methods)})",
        "",
        "| Method key | Solver family |",
        "|---|---|",
    ]
    for method in methods:
        out.append(f"| `{method}` | `{families.get(method, 'unclassified')}` |")
    out.extend(
        [
            "",
            "## Recognition and planning",
            "",
            "Recognition is layered: canonicalization records equation structure; family recognizers identify applicable mathematical forms; condition and domain analysis add geometry/data constraints; the planner ranks executable method keys; the solver coordinator executes the selected method.",
            "",
            "Important recognition and planning entry points include `build_canonical_representation(...)`, `recognize_pde_structure(...)`, `recognize_canonical_problem(...)`, `plan_canonical_problem(...)`, `build_separable_geometry_plan(...)`, `build_transform_method_plan(...)`, and `build_kernel_method_plan(...)`.",
            "",
            "## Focused solver APIs",
            "",
            "Focused functions remain useful when the formulation itself matters, including first-order linear/nonlinear solvers, complete-integral and Cauchy solvers, conservation-law solvers, unified transforms, hyperbolic systems, Sturm–Liouville analysis, and Green/fundamental-solution APIs.",
            "",
            "## Formal representations",
            "",
            "Some transform and reduction functions intentionally return unevaluated integral, series, or implicit representations. These are valid symbolic endpoints and are verified according to their representation rather than being forced into a closed form.",
            "",
        ]
    )
    (ROOT / "docs/method-inventory.md").write_text("\n".join(out))


def _notebook_title(path):
    import json

    data = json.loads(path.read_text())
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    return path.stem.replace("_", " ").title()


def write_notebook_index():
    tutorials = sorted((ROOT / "notebooks/tutorials").glob("*.ipynb"))
    examples = sorted((ROOT / "notebooks").glob("*.ipynb"))
    out = [
        "# Notebook index",
        "",
        "## Tutorial curriculum",
        "",
        "| Notebook | Topic |",
        "|---|---|",
    ]
    for path in tutorials:
        out.append(f"| `{path.name}` | {_notebook_title(path)} |")
    out.extend(["", "## Focused example notebooks", "", "| Notebook | Topic |", "|---|---|"])
    for path in examples:
        out.append(f"| `{path.name}` | {_notebook_title(path)} |")
    out.extend(
        [
            "",
            "Tutorial notebooks combine derivation, planner or recognizer inspection, solver execution, independent checks, and visualizations where they clarify the mathematics.",
            "",
        ]
    )
    (ROOT / "docs/notebook-index.md").write_text("\n".join(out))


def main():
    write_api_inventory()
    write_method_inventory()
    write_notebook_index()


if __name__ == "__main__":
    main()
