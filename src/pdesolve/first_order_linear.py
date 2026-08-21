"""Recognition and solving for linear first-order PDEs."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .first_order_geometry import AdaptedCoordinateReduction, adapted_coordinate_reduction
from .results import SolverMethodResult

# Recognition ---------------------------------------------------------------


@dataclass(frozen=True)
class LinearFirstOrderProfile:
    a: sp.Expr
    b: sp.Expr
    c: sp.Expr
    d: sp.Expr
    reduction: AdaptedCoordinateReduction | None


def parse_linear_first_order(
    eq: sp.Equality | sp.Expr, u: sp.Function, x: sp.Symbol, y: sp.Symbol
) -> LinearFirstOrderProfile:
    """Parse ``a u_x + b u_y + c u + d = 0`` into coefficients."""
    expr = sp.simplify(eq.lhs - eq.rhs) if isinstance(eq, sp.Equality) else sp.simplify(eq)
    dep = u(x, y)
    dep_x = sp.diff(dep, x)
    dep_y = sp.diff(dep, y)
    a = sp.simplify(expr.coeff(dep_x))
    expr = sp.simplify(expr - a * dep_x)
    b = sp.simplify(expr.coeff(dep_y))
    expr = sp.simplify(expr - b * dep_y)
    c = sp.simplify(expr.coeff(dep))
    expr = sp.simplify(expr - c * dep)
    d = sp.simplify(expr)
    red = adapted_coordinate_reduction(a, b, x, y)
    return LinearFirstOrderProfile(a=a, b=b, c=c, d=d, reduction=red)


def recognize_first_order_linear_pde(
    eq: sp.Equality | sp.Expr, u: sp.Function, vars: tuple[sp.Symbol, sp.Symbol]
) -> LinearFirstOrderProfile | None:
    x, y = vars
    prof = parse_linear_first_order(eq, u, x, y)
    if prof.a == 0 and prof.b == 0:
        return None
    return prof


# Solving ------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class FirstOrderPDEResult(SolverMethodResult):
    invariant: sp.Expr | None = None
    reduction: AdaptedCoordinateReduction | None = None


def solve_first_order_linear_pde(
    eq: sp.Equality | sp.Expr, u: sp.Function, vars: tuple[sp.Symbol, sp.Symbol]
) -> FirstOrderPDEResult:
    """Solve a linear first-order PDE using a simple adapted-coordinate reduction."""
    prof = recognize_first_order_linear_pde(eq, u, vars)
    if prof is None or prof.reduction is None:
        raise NotImplementedError("Could not build an adapted-coordinate reduction for this PDE")

    red = prof.reduction
    xi = red.transverse_var
    eta = red.param
    coeff = sp.simplify(red.coeff)

    c_sub = sp.simplify(prof.c.subs(red.subst_map))
    d_sub = sp.simplify(prof.d.subs(red.subst_map))
    p_term = sp.simplify(c_sub / coeff)
    q_term = sp.simplify(d_sub / coeff)
    arb = sp.Function("F")

    try:
        mu = sp.simplify(sp.exp(sp.integrate(p_term, xi)))
        red_sol = sp.simplify((arb(eta) - sp.integrate(q_term * mu, xi)) / mu)
    except Exception as exc:
        raise NotImplementedError("Failed to solve the reduced first-order ODE") from exc

    sol = sp.simplify(red_sol.subs(eta, red.invariant))
    return FirstOrderPDEResult(
        method_family="first_integral_adapted_coordinates",
        solution=sol,
        details={"profile": prof},
        invariant=red.invariant,
        reduction=red,
    )


__all__ = [
    "LinearFirstOrderProfile",
    "FirstOrderPDEResult",
    "parse_linear_first_order",
    "recognize_first_order_linear_pde",
    "solve_first_order_linear_pde",
]
