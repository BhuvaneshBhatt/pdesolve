from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from .complete_integral_helpers import (
    recognize_generalized_clairaut_pde,
    solve_charpit_complete_integral_2vars,
    solve_complete_integral_pde,
    solve_generalized_clairaut_complete_integral,
)
from .conservation_laws import (
    canonicalize_scalar_conservation_law_1d,
    parse_scalar_conservation_law_initial_data,
    solve_scalar_conservation_law_ivp,
)
from .results import ImplicitPDEResult


@dataclass(frozen=True)
class CanonicalFirstOrderPDE:
    variables: tuple[sp.Symbol, ...]
    dependent_variable: sp.Expr
    equation: sp.Equality
    F: sp.Expr
    p_symbol: sp.Symbol
    q_symbol: sp.Symbol | None
    recognized_family: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _safe_condition_model(problem):
    return (
        (
            getattr(problem, "canonical_representation", None)
            and getattr(problem.canonical_representation, "details", {})
        )
        or {}
    ).get("condition_model") or getattr(problem, "details", {}).get("condition_model")


def _extract_initial_profile(problem, uexpr, vars_):
    """Extract a scalar initial profile from either dictionary ICs or the structured ConditionModel."""
    ics = getattr(problem, "ics", None)
    if isinstance(ics, dict) and ics.get("initial_profile") is not None:
        return ics.get("initial_profile"), ics.get("curve_value", 0)
    cm = _safe_condition_model(problem)
    if cm is None:
        return None, 0
    x = vars_[0] if vars_ else None
    t = vars_[1] if len(vars_) > 1 else None
    for cond in getattr(cm, "conditions", ()):
        role = getattr(cond, "role", "")
        eq = getattr(cond, "equation", None)
        if role not in {"initial", "initial_derivative"} or not isinstance(eq, sp.Equality):
            continue
        lhs, rhs = eq.lhs, eq.rhs
        if lhs == uexpr and x is not None and t is not None:
            if lhs.args == (x, lhs.args[1]) and not lhs.args[1].has(x):
                return rhs, lhs.args[1]
            if lhs.args == (lhs.args[0], t) and lhs.args[0] == x and not lhs.args[1].has(x):
                return rhs, lhs.args[1]
        # direct matching on time-slice u(x, t0)
        if (
            lhs.func == uexpr.func
            and len(lhs.args) >= 2
            and lhs.args[0] == x
            and not lhs.args[1].has(x)
        ):
            return rhs, lhs.args[1]
    return None, 0


def canonicalize_first_order_nonlinear_pde(eq, uexpr, vars_):
    if len(vars_) < 2:
        raise NotImplementedError(
            "Canonical first-order normalization requires at least two variables."
        )
    x, y = vars_[0], vars_[1]
    p = sp.Symbol("p", real=True)
    q = sp.Symbol("q", real=True)
    zero = (eq.lhs - eq.rhs) if isinstance(eq, sp.Equality) else sp.sympify(eq)
    F = sp.expand(zero.xreplace({sp.diff(uexpr, x): p, sp.diff(uexpr, y): q}))
    fam = "generic_first_order"
    meta: dict[str, Any] = {"is_autonomous_in_xy": not F.has(x, y)}
    try:
        clair = recognize_generalized_clairaut_pde(eq, uexpr, vars_)
        if getattr(clair, "recognized", False):
            fam = "generalized_clairaut"
            meta["phi"] = getattr(clair, "phi", None)
            meta["recognizer"] = clair
    except Exception:
        pass
    try:
        cf = canonicalize_scalar_conservation_law_1d(eq, uexpr, vars_)
        if cf is not None:
            fam = "scalar_conservation_law"
            meta["canonical_conservation_law"] = cf
    except Exception:
        pass
    try:
        poly = sp.Poly(F, p, q)
    except Exception:
        poly = None
    if poly is not None:
        meta["derivative_total_degree"] = poly.total_degree()
        if poly.total_degree() <= 1 and fam == "generic_first_order":
            fam = "quasilinear_first_order"
        if poly.total_degree() == 1:
            try:
                meta["affine_in_derivatives"] = True
                meta["coeff_p"] = sp.simplify(poly.coeff_monomial(p))
                meta["coeff_q"] = sp.simplify(poly.coeff_monomial(q))
                meta["free_term"] = sp.simplify(poly.coeff_monomial(1))
            except Exception:
                pass
    if uexpr in sp.expand(zero).free_symbols:
        meta["depends_on_u"] = True
    if fam == "generic_first_order" and meta.get("is_autonomous_in_xy"):
        fam = "autonomous_charpit"
    return CanonicalFirstOrderPDE(
        tuple(vars_), uexpr, sp.Eq(sp.expand(zero), 0), F, p, q, fam, meta
    )


def execute_first_order_plan(
    problem, classical_mod=None, canonical: CanonicalFirstOrderPDE | None = None, **kwargs
):
    classical_mod = classical_mod or __import__("pdesolve.classical_methods", fromlist=["dummy"])
    canonical = (
        canonical
        or (
            (getattr(problem.canonical_representation, "details", {}) or {}).get(
                "first_order_canonical"
            )
            if getattr(problem, "canonical_representation", None) is not None
            else None
        )
        or problem.details.get("first_order_canonical")
    )
    if canonical is None:
        canonical = canonicalize_first_order_nonlinear_pde(
            problem.equation, problem.dep_function, problem.indep_vars
        )
    eq = problem.equation
    uexpr = problem.dep_function
    vars_ = problem.indep_vars
    fam = canonical.recognized_family
    initial_profile, curve_value = _extract_initial_profile(problem, uexpr, vars_)
    if fam == "generalized_clairaut":
        return solve_generalized_clairaut_complete_integral(eq, uexpr, vars_)
    if fam == "scalar_conservation_law":
        cf = canonical.metadata.get("canonical_conservation_law")
        init = parse_scalar_conservation_law_initial_data(problem.ics, uexpr, vars_)
        if init is None and initial_profile is not None:
            x = vars_[0]
            init = parse_scalar_conservation_law_initial_data(
                sp.Eq(uexpr.func(x, curve_value), initial_profile), uexpr, vars_
            )
        result = solve_scalar_conservation_law_ivp(
            cf.normalized_equation if cf is not None and cf.normalized_equation is not None else eq,
            uexpr,
            vars_,
            initial_conditions=init,
        )
        return result
    if fam == "quasilinear_first_order":
        return classical_mod.solve_quasilinear_pde_characteristics_implicit(
            eq, uexpr, vars_, initial_profile=initial_profile, initial_curve_value=curve_value
        )
    if fam == "autonomous_charpit":
        try:
            return solve_charpit_complete_integral_2vars(eq, uexpr, vars_)
        except Exception:
            pass
    try:
        return solve_complete_integral_pde(
            eq, uexpr, vars_, assumptions=getattr(problem, "assumptions", True), **kwargs
        )
    except Exception:
        pass
    return ImplicitPDEResult(
        method="first_order_framework",
        solution=sp.Eq(uexpr, sp.Function("F")(*vars_)),
        metadata={
            "canonical_first_order": canonical,
            "native_structured_execution": True,
            "initial_profile": initial_profile,
            "initial_curve_value": curve_value,
        },
        reduced_problem=canonical.equation,
    )


__all__ = [
    "CanonicalFirstOrderPDE",
    "canonicalize_first_order_nonlinear_pde",
    "execute_first_order_plan",
]
