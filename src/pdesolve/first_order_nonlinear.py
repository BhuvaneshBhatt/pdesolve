"""Recognition and solving helpers for nonlinear first-order PDEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import sympy as sp

from .first_order_linear import FirstOrderPDEResult, parse_linear_first_order
from .first_order_framework import canonicalize_first_order_nonlinear_pde


# Recognition ---------------------------------------------------------------


@dataclass(frozen=True)
class ConstantCharacteristicProfile:
    a: sp.Expr
    b: sp.Expr
    source: sp.Expr
    invariant: sp.Expr


@dataclass(frozen=True)
class FirstOrderNonlinearAnalysis:
    is_first_order: bool
    is_linear_first_order: bool
    is_quasilinear: bool
    conservation_law_family: str | None
    burgers_family: str | None
    recommended_methods: tuple[str, ...]
    details: dict[str, Any]


@dataclass(frozen=True)
class InvariantReductionCandidate:
    method: str
    score: int
    invariants: tuple[sp.Expr, ...]
    transverse_params: tuple[sp.Expr, ...]
    reduced_eq: sp.Equality
    ansatz: sp.Expr
    red_func: Any = None
    details: dict[str, Any] | None = None

    @property
    def transverse_parameters(self):
        return self.transverse_params

    @property
    def reduced_equation(self):
        return self.reduced_eq

    @property
    def reduced_function(self):
        return self.red_func


def recognize_const_characteristics(
    eq: sp.Equality | sp.Expr, u: sp.Function, vars: tuple[sp.Symbol, sp.Symbol]
) -> Optional[ConstantCharacteristicProfile]:
    """Recognize ``a u_x + b u_y + d = 0`` with constant ``a,b,d``."""
    x, y = vars
    prof = parse_linear_first_order(eq, u, x, y)
    if prof.c != 0:
        return None
    vals = (prof.a, prof.b, prof.d)
    if not all(val.free_symbols.isdisjoint({x, y}) for val in vals):
        return None
    if prof.a == 0 and prof.b == 0:
        return None
    inv = (
        x if prof.a == 0 else y if prof.b == 0 else sp.simplify(prof.b * x - prof.a * y)
    )
    return ConstantCharacteristicProfile(
        a=prof.a, b=prof.b, source=prof.d, invariant=inv
    )


# Solving ------------------------------------------------------------------


def solve_first_order_quasilinear_pde(
    eq: sp.Equality | sp.Expr, u: sp.Function, vars: tuple[sp.Symbol, sp.Symbol]
) -> FirstOrderPDEResult:
    """Solve the constant-characteristic subset ``a u_x + b u_y + d = 0``."""
    x, y = vars
    prof = recognize_const_characteristics(eq, u, vars)
    if prof is None:
        raise NotImplementedError(
            "The quasilinear fallback requires constant characteristics and source"
        )
    arb = sp.Function("F")
    if prof.a == 0:
        sol = sp.simplify((-prof.source / prof.b) * y + arb(prof.invariant))
    elif prof.b == 0:
        sol = sp.simplify((-prof.source / prof.a) * x + arb(prof.invariant))
    else:
        sol = sp.simplify((-prof.source / prof.a) * x + arb(prof.invariant))
    return FirstOrderPDEResult(
        method_family="constant_characteristics_quasilinear",
        solution=sol,
        details={"profile": prof},
        invariant=prof.invariant,
        reduction=None,
    )


# Advanced nonlinear analysis ----------------------------------------------


def _build_eq_obj(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, max_principal_order=3
):
    from ._classical_shared import _dep_and_vars
    from .classical_methods import canonicalize_pde_problem
    from .pde import (
        build_scalar_jet_equation_from_sympy_pde,
        build_scalar_general_solved_pde_from_equation,
    )

    dep_expr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    eq = canonicalize_pde_problem(eq_or_expr, dep_expr, vars_).equation
    jet, pde_obj = build_scalar_jet_equation_from_sympy_pde(
        vars_,
        dep_expr.func,
        eq,
        max_order=max_principal_order,
        dep_name=getattr(dep_expr.func, "__name__", "u"),
    )
    eq_obj, info = build_scalar_general_solved_pde_from_equation(
        jet, pde_obj, max_principal_order=max_principal_order
    )
    return dep_expr, tuple(vars_), eq, eq_obj, info


def analyze_first_order_nonlinear_pde(eq_or_expr, dep_expr_or_func, indep_vars=None):
    from ._classical_shared import _dep_and_vars, _as_zero_expr
    from .classical_methods import (
        _infer_pde_order,
        detect_first_order_linear_form_2vars,
        characteristic_form_first_order_2vars,
    )

    dep_expr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    zero = _as_zero_expr(eq_or_expr)
    order = _infer_pde_order(zero, dep_expr)
    is_first = order == 1
    is_lin = False
    is_quasi = False
    det: dict[str, Any] = {"order": order}
    canonical = None
    if is_first and len(vars_) >= 2:
        try:
            canonical = canonicalize_first_order_nonlinear_pde(
                eq_or_expr, dep_expr, vars_
            )
            det["canonical_first_order"] = canonical
            det["recognized_family"] = canonical.recognized_family
            det["canonical_metadata"] = canonical.metadata
        except Exception as exc:
            det["canonical_first_order_error"] = str(exc)
        try:
            detect_first_order_linear_form_2vars(eq_or_expr, dep_expr, vars_)
            is_lin = True
        except Exception as exc:
            det["first_order_linear_error"] = str(exc)
        try:
            char_form = characteristic_form_first_order_2vars(
                eq_or_expr, dep_expr, vars_
            )
            is_quasi = bool(getattr(char_form, "is_quasilinear", False)) or (
                canonical is not None
                and canonical.recognized_family == "quasilinear_first_order"
            )
            det["characteristic_form"] = char_form
        except Exception as exc:
            det["characteristic_data_error"] = str(exc)
    family = (
        getattr(canonical, "recognized_family", None) if canonical is not None else None
    )
    cons = family if family == "scalar_conservation_law" else None
    burgers = (
        "burgers" if cons == "scalar_conservation_law" and zero.has(dep_expr) else None
    )
    rec: list[str] = []
    if is_first and not is_lin:
        if family == "scalar_conservation_law":
            rec += [
                "conservation_law",
                "quasilinear_implicit",
                "first_order_nonlinear_auto",
            ]
        elif family == "generalized_clairaut":
            rec += [
                "generalized_clairaut_complete_integral",
                "complete_integral",
                "charpit",
                "first_order_nonlinear_auto",
            ]
        elif family == "quasilinear_first_order":
            rec += [
                "quasilinear_implicit",
                "first_order_framework",
                "complete_integral",
                "charpit",
                "first_order_nonlinear_auto",
            ]
        elif family == "autonomous_charpit":
            rec += [
                "autonomous_charpit",
                "charpit",
                "first_order_framework",
                "complete_integral",
                "first_order_nonlinear_auto",
            ]
        else:
            rec += [
                "first_order_framework",
                "complete_integral",
                "charpit",
                "jacobi",
                "first_order_nonlinear_auto",
            ]
    return FirstOrderNonlinearAnalysis(
        is_first_order=is_first,
        is_linear_first_order=is_lin,
        is_quasilinear=is_quasi,
        conservation_law_family=cons,
        burgers_family=burgers,
        recommended_methods=tuple(dict.fromkeys(rec)),
        details=det,
    )


def enumerate_invariant_reduction_candidates(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    degree=1,
    max_subset_size=2,
    max_degree=2,
):
    from .symmetry import (
        solve_determining_equations_with_polynomial_ansatz_scalar_general_kd,
    )
    from .reduction import (
        search_symbolic_linear_combinations_for_reduction_scalar_kd,
        auto_reduce_best_commuting_subalgebra_scalar_kd,
        auto_reduce_best_symbolic_match_scalar_kd,
        reduce_scalar_by_frobenius_chart,
    )
    from .workflows import _obvious_translation_distribution
    from .frobenius import local_frobenius_chart
    from .utils import expr_complexity
    from .verify import verify_reduction

    _, _, _, eq_obj, info = _build_eq_obj(eq_or_expr, dep_expr_or_func, indep_vars)
    out: list[InvariantReductionCandidate] = []
    for deg in range(int(degree), int(max_degree) + 1):
        try:
            sym_sol = (
                solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(
                    eq_obj, degree=deg, include_dependent_var=True
                )
            )
            basis = sym_sol.basis_vectors()
            matches = tuple(
                search_symbolic_linear_combinations_for_reduction_scalar_kd(
                    eq_obj,
                    basis,
                    max_subset_size=max_subset_size,
                    try_translation=True,
                    try_diagonal_scaling=True,
                    try_affine=True,
                    normalize=True,
                    rank_results=True,
                )
            )
            red = auto_reduce_best_commuting_subalgebra_scalar_kd(
                eq_obj,
                list(matches),
                max_generators=min(eq_obj.jet.k - 1, max_subset_size),
            )
            if red is not None:
                ver = verify_reduction(eq_obj, red)
                score = (
                    100
                    - int(
                        sum(
                            expr_complexity(v)
                            for v in red.invariants + red.transverse_parameters
                        )
                    )
                    + (10 if ver.valid else 0)
                )
                out.append(
                    InvariantReductionCandidate(
                        method=f"commuting_subalgebra_deg{deg}",
                        score=score,
                        invariants=tuple(red.invariants),
                        transverse_params=tuple(red.transverse_parameters),
                        reduced_eq=red.reduced_equation,
                        ansatz=red.ansatz,
                        red_func=getattr(red, "reduced_function", None),
                        details={
                            "reduction": red,
                            "symmetry_solution": sym_sol,
                            "principal_info": info,
                            "verification": ver,
                            "degree": deg,
                        },
                    )
                )
            red2 = auto_reduce_best_symbolic_match_scalar_kd(eq_obj, list(matches))
            if red2 is not None:
                trans = (
                    red2.transverse_parameter
                    if isinstance(red2.transverse_parameter, tuple)
                    else (red2.transverse_parameter,)
                )
                ver = verify_reduction(eq_obj, red2)
                score = (
                    90
                    - int(
                        sum(expr_complexity(v) for v in red2.invariants + tuple(trans))
                    )
                    + (8 if ver.valid else 0)
                )
                out.append(
                    InvariantReductionCandidate(
                        method=f"symbolic_match_deg{deg}",
                        score=score,
                        invariants=tuple(red2.invariants),
                        transverse_params=tuple(trans),
                        reduced_eq=red2.reduced_equation,
                        ansatz=red2.ansatz,
                        red_func=getattr(red2, "reduced_function", None),
                        details={
                            "reduction": red2,
                            "symmetry_solution": sym_sol,
                            "principal_info": info,
                            "verification": ver,
                            "degree": deg,
                        },
                    )
                )
        except Exception:
            pass
    try:
        dist = _obvious_translation_distribution(eq_obj)
        if dist is not None:
            chart = local_frobenius_chart(dist)
            red3 = reduce_scalar_by_frobenius_chart(
                eq_obj,
                chart,
                a_list=[0] * len(chart.transverse),
                beta_list=[0] * len(chart.transverse),
            )
            ver = verify_reduction(eq_obj, red3, chart)
            score = (
                80
                - int(
                    sum(expr_complexity(v) for v in chart.invariants + chart.transverse)
                )
                + (6 if ver.valid else 0)
            )
            out.append(
                InvariantReductionCandidate(
                    method="frobenius_chart",
                    score=score,
                    invariants=tuple(red3.invariants),
                    transverse_params=tuple(red3.transverse_parameters),
                    reduced_eq=red3.reduced_equation,
                    ansatz=red3.ansatz,
                    red_func=getattr(red3, "reduced_function", None),
                    details={
                        "reduction": red3,
                        "chart": chart,
                        "principal_info": info,
                        "verification": ver,
                    },
                )
            )
    except Exception:
        pass
    uniq: dict[tuple[str, str], InvariantReductionCandidate] = {}
    for cand in out:
        red_eq = cand.reduced_eq
        if red_eq is sp.true or red_eq == sp.S.true:
            red_expr = sp.Integer(0)
        elif red_eq is sp.false or red_eq == sp.S.false:
            red_expr = sp.Integer(1)
        else:
            red_expr = sp.expand(red_eq.lhs - red_eq.rhs)
        key = (cand.method, sp.srepr(red_expr))
        prev = uniq.get(key)
        if prev is None or cand.score > prev.score:
            uniq[key] = cand
    return tuple(sorted(uniq.values(), key=lambda item: (-item.score, item.method)))


def _sub_red_sol(ansatz, red_sol):
    if not isinstance(red_sol, sp.Equality):
        return ansatz
    return sp.expand(ansatz.subs(red_sol.lhs, red_sol.rhs))


def solve_via_invariant_reduction(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    assumptions=True,
    degree=1,
    max_subset_size=2,
    max_symmetry_steps=2,
    max_degree=2,
):
    from ._classical_shared import _dep_and_vars
    from .classical_methods import PDEIVPResult, solve_reduced_equation_auto

    dep_expr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    cands = enumerate_invariant_reduction_candidates(
        eq_or_expr,
        dep_expr,
        vars_,
        degree=degree,
        max_subset_size=max_subset_size,
        max_degree=max_degree,
    )
    if not cands:
        raise NotImplementedError("No invariant-reduction candidate was found.")
    cand = cands[0]
    red_res = solve_reduced_equation_auto(
        cand.reduced_eq,
        ics=ics,
        bcs=bcs,
        assumptions=assumptions,
        max_symmetry_steps=max_symmetry_steps,
    )
    red_sol = getattr(red_res, "solution", red_res)
    full_rhs = _sub_red_sol(cand.ansatz, red_sol)
    return PDEIVPResult(
        method="invariant_reduction_auto",
        solution=sp.Eq(dep_expr, full_rhs),
        details={
            "candidate": cand,
            "candidates": cands,
            "reduced_result": red_res,
            "reduced_equation": cand.reduced_eq,
        },
    )


def solve_first_order_nonlinear_auto(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    assumptions=True,
    **kwargs,
):
    from ._classical_shared import _dep_and_vars
    from .problem import build_pde_problem
    from .solver_execution import solve_with_canonical_problem

    dep_expr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    analysis = analyze_first_order_nonlinear_pde(eq_or_expr, dep_expr, vars_)
    if not analysis.is_first_order:
        raise NotImplementedError(
            "First-order nonlinear auto solver expects a first-order PDE."
        )

    problem = build_pde_problem(
        eq_or_expr, dep_expr, vars_, ics=ics, bcs=bcs, assumptions=assumptions
    )
    canonical = analysis.details.get("canonical_first_order")
    methods = []
    if canonical is not None and canonical.recognized_family in {
        "generalized_clairaut",
        "scalar_conservation_law",
        "quasilinear_first_order",
        "autonomous_charpit",
    }:
        methods.append("first_order_nonlinear_auto")
    for meth in analysis.recommended_methods:
        if meth not in methods:
            methods.append(meth)
    if "invariant_reduction_auto" not in methods:
        methods.append("invariant_reduction_auto")

    last_exc = None
    for meth in methods:
        try:
            return solve_with_canonical_problem(problem, meth, **kwargs)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise NotImplementedError("No nonlinear first-order method succeeded.")


__all__ = [
    "ConstantCharacteristicProfile",
    "FirstOrderNonlinearAnalysis",
    "InvariantReductionCandidate",
    "recognize_const_characteristics",
    "solve_first_order_quasilinear_pde",
    "analyze_first_order_nonlinear_pde",
    "enumerate_invariant_reduction_candidates",
    "solve_via_invariant_reduction",
    "solve_first_order_nonlinear_auto",
]
