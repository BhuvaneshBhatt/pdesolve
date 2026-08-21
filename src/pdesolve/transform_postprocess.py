from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class TransformPostprocessReport:
    changed: bool
    stages: tuple[str, ...]
    skipped_reason: str | None = None


def _small(expr, max_ops):
    try:
        return int(sp.count_ops(expr)) <= max_ops
    except Exception:
        return False


def evaluate_inner_transforms(expr, *, max_ops: int = 60):
    """Evaluate safe one-dimensional inner transform integrals, never outer inversions."""
    if not isinstance(expr, sp.Basic):
        return expr, TransformPostprocessReport(False, (), "non_symbolic")
    replacements = {}
    for integ in sorted(expr.atoms(sp.Integral), key=lambda i: len(i.atoms(sp.Integral))):
        if any(isinstance(a, sp.Integral) for a in integ.function.atoms(sp.Integral)):
            continue
        if len(integ.limits) != 1 or not _small(integ.function, max_ops):
            continue
        _, lo, hi = integ.limits[0]
        # Restrict automatic evaluation to canonical Fourier sine/cosine/Laplace profile integrals.
        if (lo, hi) not in ((0, sp.oo), (-sp.oo, sp.oo)):
            continue
        f = integ.function
        if not (
            f.has(sp.sin, sp.cos) or any(node.func == sp.exp for node in sp.preorder_traversal(f))
        ):
            continue
        try:
            val = integ.doit(risch=False)
        except TypeError:
            try:
                val = integ.doit()
            except Exception:
                continue
        except Exception:
            continue
        if val != integ and not val.has(sp.Integral) and _small(val, max_ops * 3):
            replacements[integ] = val
    out = expr.xreplace(replacements)
    return out, TransformPostprocessReport(
        bool(replacements),
        ("evaluate_inner_transforms",) if replacements else (),
        None if replacements else "no_safe_inner_transform",
    )


def postprocess_transform_result(result: Any, *, max_ops: int = 60):
    solution = getattr(result, "solution", result)
    processed, report = evaluate_inner_transforms(solution, max_ops=max_ops)
    if hasattr(result, "__dataclass_fields__") and hasattr(result, "solution"):
        import dataclasses

        try:
            result = dataclasses.replace(result, solution=processed)
        except Exception:
            pass
    return result if hasattr(result, "solution") else processed, report


__all__ = [
    "TransformPostprocessReport",
    "evaluate_inner_transforms",
    "postprocess_transform_result",
]
