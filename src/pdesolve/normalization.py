from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class NormalizationPolicy:
    enabled: bool = True
    max_ops: int = 80
    simplify_max_ops: int = 36
    keep_only_if_not_larger: bool = True


@dataclass(frozen=True)
class NormalizationReport:
    attempted: bool
    changed: bool
    skipped_reason: str | None = None
    original_ops: int | None = None
    normalized_ops: int | None = None
    stages: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "attempted": self.attempted,
            "changed": self.changed,
            "skipped_reason": self.skipped_reason,
            "original_ops": self.original_ops,
            "normalized_ops": self.normalized_ops,
            "stages": self.stages,
        }


_SKIP_METHOD_TOKENS = (
    "implicit",
    "shock",
    "rarefaction",
    "conservation",
    "series",
    "transform",
    "fourier",
    "laplace",
    "kernel",
    "green",
)
_SKIP_ATOMS = (sp.Integral, sp.Sum, sp.Product, sp.Piecewise, sp.Derivative)


def _ops(expr):
    try:
        return int(sp.count_ops(expr, visual=False))
    except Exception:
        return None


def _eligible(expr, method: str, policy: NormalizationPolicy):
    if not policy.enabled:
        return False, "disabled"
    low_method = (method or "").lower()
    if any(tok in low_method for tok in _SKIP_METHOD_TOKENS):
        return False, "method_specific_representation"
    if isinstance(expr, (dict, list, tuple, set)):
        return False, "structured_solution"
    target = expr.rhs if isinstance(expr, sp.Equality) else expr
    if not isinstance(target, sp.Basic):
        return False, "non_symbolic_solution"
    if target.has(*_SKIP_ATOMS):
        return False, "formal_or_piecewise_expression"
    n = _ops(target)
    if n is not None and n > policy.max_ops:
        return False, "complexity_budget"
    return True, None


def _normalize_expr(expr: sp.Expr, policy: NormalizationPolicy):
    current = expr
    stages = []
    original_ops = _ops(expr)
    candidates = []
    for name, func in (
        ("cancel", sp.cancel),
        ("factor_terms", sp.factor_terms),
        ("powsimp", lambda e: sp.powsimp(e, force=False)),
        ("trigsimp", sp.trigsimp),
    ):
        try:
            candidate = func(current)
            candidates.append((name, candidate))
            c_ops, cur_ops = _ops(candidate), _ops(current)
            if (
                not policy.keep_only_if_not_larger
                or c_ops is None
                or cur_ops is None
                or c_ops <= cur_ops
            ):
                if candidate != current:
                    current = candidate
                    stages.append(name)
        except Exception:
            continue
    if (_ops(current) or 0) <= policy.simplify_max_ops:
        try:
            candidate = sp.simplify(current)
            c_ops, cur_ops = _ops(candidate), _ops(current)
            if (
                not policy.keep_only_if_not_larger
                or c_ops is None
                or cur_ops is None
                or c_ops <= cur_ops
            ):
                if candidate != current:
                    current = candidate
                    stages.append("simplify")
        except Exception:
            pass
    return current, original_ops, _ops(current), tuple(stages)


def normalize_solution(
    solution: Any, *, method: str = "", policy: NormalizationPolicy | None = None
):
    """Conservatively normalize a closed-form symbolic solution.

    Formal transforms, series, weak/implicit solutions, piecewise expressions,
    and expressions above the operation budget are intentionally left alone.
    """
    policy = policy or NormalizationPolicy()
    ok, reason = _eligible(solution, method, policy)
    target = solution.rhs if isinstance(solution, sp.Equality) else solution
    if not ok:
        return solution, NormalizationReport(False, False, reason, _ops(target), _ops(target), ())
    normalized, before, after, stages = _normalize_expr(target, policy)
    if isinstance(solution, sp.Equality):
        out = sp.Eq(solution.lhs, normalized, evaluate=False)
    else:
        out = normalized
    return out, NormalizationReport(True, out != solution, None, before, after, stages)


__all__ = ["NormalizationPolicy", "NormalizationReport", "normalize_solution"]
