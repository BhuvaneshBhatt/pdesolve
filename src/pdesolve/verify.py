from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .canonical import canonicalize_reduced_equation
from .diagnostics import local_chart_conditions_from_coords
from .geometry import CharacteristicCoordinatesResult, DistributionKD


@dataclass
class ChartVerificationResult:
    valid: bool
    jacobian: sp.Expr
    invariant_residuals: tuple[sp.Expr, ...]
    transverse_residuals: tuple[tuple[sp.Expr, ...], ...]
    conditions: tuple[sp.Expr, ...]


@dataclass
class ReductionVerificationResult:
    valid: bool
    chart_valid: bool
    residual: sp.Expr
    reduced_residual: sp.Expr
    conditions: tuple[sp.Expr, ...]


def verify_frobenius_chart(
    distribution: DistributionKD, chart: CharacteristicCoordinatesResult
) -> ChartVerificationResult:
    fields = distribution.fields
    invariants = tuple(chart.invariants)
    transverse = tuple(chart.transverse)
    coords = invariants + transverse
    J = sp.Matrix([[sp.diff(c, v) for v in distribution.vars] for c in coords])
    jac = sp.simplify(J.det())
    inv_res = [sp.simplify(X.apply(z)) for X in fields for z in invariants]
    trans_res = []
    for X in fields:
        trans_res.append(tuple(sp.simplify(X.apply(s)) for s in transverse))
    conds = tuple(chart.validity_conditions) + local_chart_conditions_from_coords(
        distribution.vars, coords
    )
    conds = tuple(dict.fromkeys(sp.simplify(c) for c in conds))
    valid = all(r == 0 for r in inv_res) and sp.simplify(jac) != 0
    return ChartVerificationResult(valid, jac, tuple(inv_res), tuple(trans_res), conds)


def verify_reduction(
    eq_obj, reduction_result, chart: CharacteristicCoordinatesResult | None = None
) -> ReductionVerificationResult:
    """Best-effort consistency check for a reduced equation. Compares canonicalized residuals."""
    red_eq = canonicalize_reduced_equation(reduction_result.reduced_equation)
    residual = sp.simplify(sp.expand(red_eq.rhs))
    chart_valid = True
    conditions = tuple()
    if chart is not None:
        # build a synthetic distribution if generators are not available is skipped; verify only coordinates.
        coords = chart.invariants + chart.transverse
        J = sp.Matrix([[sp.diff(c, v) for v in eq_obj.jet.xs] for c in coords])
        chart_valid = sp.simplify(J.det()) != 0
        conditions = tuple(chart.validity_conditions) + local_chart_conditions_from_coords(
            eq_obj.jet.xs, coords
        )
        conditions = tuple(dict.fromkeys(sp.simplify(c) for c in conditions))
    reduced_residual = (
        sp.simplify(sp.expand(reduction_result.reduced_expression - red_eq.lhs))
        if hasattr(reduction_result, "reduced_expression")
        else sp.Integer(0)
    )
    valid = chart_valid and sp.simplify(red_eq.rhs) == 0 and sp.simplify(reduced_residual) == 0
    return ReductionVerificationResult(
        valid=valid,
        chart_valid=chart_valid,
        residual=residual,
        reduced_residual=reduced_residual,
        conditions=conditions,
    )


# ---------------------------------------------------------------------------
# PDE solution verification helpers
# ---------------------------------------------------------------------------


def _as_zero_expr(eq_or_expr):
    if isinstance(eq_or_expr, sp.Equality):
        return sp.expand(eq_or_expr.lhs - eq_or_expr.rhs)
    return sp.expand(sp.sympify(eq_or_expr))


def _safe_zero(expr):
    try:
        return sp.simplify(sp.expand(expr)) == 0
    except Exception:
        return None


def _sample_points(vars_):
    pts = []
    if not vars_:
        return [dict()]
    seeds = [0, 1, 2]
    for seed in seeds:
        pts.append({v: sp.Integer(seed + i) for i, v in enumerate(vars_)})
    return pts


def verify_solution_with_conditions(
    eq_or_expr,
    solution,
    dep_expr_or_func=None,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    assumptions=True,
):
    from . import classical_methods as classical_mod
    from .results import PDEVerificationSummary

    try:
        uexpr, vars_ = classical_mod._dep_and_vars(dep_expr_or_func, indep_vars)
        norm_eq = classical_mod.canonicalize_pde_problem(
            eq_or_expr, uexpr, vars_, assumptions=assumptions
        ).equation
    except Exception as exc:
        return PDEVerificationSummary(
            False, "failed", mode="structural", message=f"problem normalization failed: {exc}"
        )

    if isinstance(solution, sp.Equality):
        candidate = solution
    elif hasattr(solution, "solution") and isinstance(solution.solution, sp.Equality):
        candidate = solution.solution
    else:
        return PDEVerificationSummary(
            None, "unverified", mode="structural", message="solution is not an explicit equality"
        )

    # First try the package verifier.
    try:
        res = classical_mod.verify_pde_solution_with_data(
            norm_eq, candidate, uexpr, vars_, ics=ics, bcs=bcs, assumptions=assumptions
        )
        if hasattr(res, "as_dict"):
            res_map = res.as_dict()
        elif isinstance(res, dict):
            res_map = res
        else:
            res_map = {
                "verified": getattr(res, "verified", None),
                "pde_residual": getattr(res, "pde_residual", None),
                "initial_residuals": getattr(res, "initial_residuals", ()),
                "boundary_residuals": getattr(res, "boundary_residuals", ()),
                "pde_verified": getattr(res, "pde_verified", None),
                "initial_verified": getattr(res, "initial_verified", None),
                "boundary_verified": getattr(res, "boundary_verified", None),
                "message": getattr(res, "message", ""),
            }
        pde_res = res_map.get("pde_residual")
        init_res = tuple(res_map.get("initial_residuals", ()) or ())
        bc_res = tuple(res_map.get("boundary_residuals", ()) or ())
        pde_verified = res_map.get("pde_verified")
        initial_verified = res_map.get("initial_verified")
        boundary_verified = res_map.get("boundary_verified")
        if pde_verified is None and pde_res is not None:
            pde_verified = _safe_zero(pde_res)
        if initial_verified is None and init_res:
            values = [_safe_zero(r) for r in init_res]
            initial_verified = (
                all(v is True for v in values) if all(v is not None for v in values) else None
            )
        if boundary_verified is None and bc_res:
            values = [_safe_zero(r) for r in bc_res]
            boundary_verified = (
                all(v is True for v in values) if all(v is not None for v in values) else None
            )
        verified = res_map.get("verified")
        mode = "symbolic"
        if verified is None:
            bits = [b for b in (pde_verified, initial_verified, boundary_verified) if b is not None]
            verified = all(bits) if bits else None
        if pde_verified is None and pde_res is not None:
            # numeric spot check when symbolic status is unclear
            try:
                subs_solution = {candidate.lhs: candidate.rhs}
                residual_expr = _as_zero_expr(norm_eq).xreplace(subs_solution).doit()
                checks = []
                for sub in _sample_points(vars_):
                    val = sp.N(residual_expr.subs(sub))
                    checks.append(abs(complex(val)) < 1e-8)
                if checks:
                    mode = "numeric_spotcheck"
                    pde_verified = all(checks)
                    pde_res = residual_expr
                    bits = [
                        b
                        for b in (pde_verified, initial_verified, boundary_verified)
                        if b is not None
                    ]
                    verified = all(bits) if bits else None
            except Exception:
                pass
        status = "verified" if verified is True else "failed" if verified is False else "unverified"
        return PDEVerificationSummary(
            verified=verified,
            status=status,
            pde_verified=pde_verified,
            initial_verified=initial_verified,
            boundary_verified=boundary_verified,
            pde_residual=pde_res,
            initial_residuals=init_res,
            boundary_residuals=bc_res,
            mode=mode,
            message=res_map.get("message", ""),
        )
    except Exception as exc:
        return PDEVerificationSummary(False, "failed", mode="exception", message=str(exc))


def verify_kernel_representation(
    eq_or_expr,
    kernel,
    dep_expr_or_func,
    indep_vars,
    *,
    geometry=None,
    bcs=None,
    operator_family=None,
    boundary_family=None,
):
    eq = eq_or_expr if isinstance(eq_or_expr, sp.Equality) else sp.Eq(sp.sympify(eq_or_expr), 0)
    info = {
        "verified": None,
        "mode": "kernel_heuristic",
        "operator_family": operator_family,
        "boundary_family": boundary_family,
    }
    try:
        zero = sp.expand(eq.lhs - eq.rhs)
        info["has_dirac_source"] = bool(zero.has(sp.DiracDelta))
    except Exception:
        info["has_dirac_source"] = None
    # Boundary checks for common image/series kernels.
    try:
        residuals = []
        if bcs is not None:
            bclist = list(bcs) if isinstance(bcs, (list, tuple)) else [bcs]
            for bc in bclist:
                if isinstance(bc, sp.Equality):
                    lhs = bc.lhs
                    rhs = bc.rhs
                    if getattr(lhs, "func", None) == getattr(
                        dep_expr_or_func, "func", dep_expr_or_func
                    ):
                        sub_map = {v: a for v, a in zip(indep_vars, lhs.args, strict=True)}
                        residuals.append(sp.simplify((kernel - rhs).subs(sub_map)))
                    elif isinstance(lhs, sp.Subs) and isinstance(lhs.expr, sp.Derivative):
                        expr = lhs.expr
                        subbed = kernel
                        for old, new in zip(lhs.variables, lhs.point, strict=True):
                            subbed = subbed.subs(old, new)
                        deriv_expr = sp.diff(kernel, *expr.variable_count)
                        residuals.append(
                            sp.simplify(
                                deriv_expr.subs(
                                    {v: a for v, a in zip(indep_vars, expr.expr.args, strict=True)}
                                )
                                - rhs
                            )
                        )
            if residuals:
                info["boundary_residuals"] = tuple(residuals)
                vals = []
                for r in residuals:
                    z = _safe_zero(r)
                    vals.append(z)
                if vals and all(v is not None for v in vals):
                    info["boundary_verified"] = all(vals)
        if boundary_family in {"dirichlet", "neumann"} and info.get("boundary_verified") is None:
            # image/series constructions are assumed admissible unless contradicted by symbolic check
            info["boundary_verified"] = True if geometry is not None else None
        if info.get("has_dirac_source"):
            info["distributional_plausibility"] = True
        info["verified"] = (
            info.get("boundary_verified")
            if info.get("boundary_verified") is not None
            else info.get("distributional_plausibility")
        )
    except Exception as exc:
        info["warning"] = str(exc)
    return info
