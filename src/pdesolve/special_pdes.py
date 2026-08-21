"""Recognizers and simple symbolic families for special PDE classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp


@dataclass
class SpecialPDEResult:
    solution_family: sp.Expr
    family_name: str


def recognize_heat_or_advection_diffusion(
    eq: sp.Equality | sp.Expr, u: sp.Function, vars: tuple[sp.Symbol, sp.Symbol]
) -> Optional[SpecialPDEResult]:
    """Recognize ``u_t = k u_xx`` and ``u_t = k u_xx + gamma u_x``.

    The returned solution family is a one-parameter exponential family,

        exp(c0 + r x + (k r**2 + gamma r) t),

    which is a genuine family of exact solutions rather than the full general
    solution.
    """
    x, t = vars
    expr = sp.simplify((eq.lhs - eq.rhs) if isinstance(eq, sp.Equality) else eq)
    uxt = u(x, t)
    ut = sp.diff(uxt, t)
    ux = sp.diff(uxt, x)
    uxx = sp.diff(uxt, x, 2)
    a_t = sp.simplify(expr.coeff(ut))
    expr = sp.simplify(expr - a_t * ut)
    a_xx = sp.simplify(expr.coeff(uxx))
    expr = sp.simplify(expr - a_xx * uxx)
    a_x = sp.simplify(expr.coeff(ux))
    expr = sp.simplify(expr - a_x * ux)
    if expr != 0 or a_t == 0:
        return None

    k = sp.simplify(-a_xx / a_t)
    gamma = sp.simplify(-a_x / a_t)
    if not k.free_symbols.isdisjoint({x, t}) or not gamma.free_symbols.isdisjoint(
        {x, t}
    ):
        return None

    c0, r = sp.symbols("c0 r")
    family = sp.exp(c0 + r * x + (k * r**2 + gamma * r) * t)
    name = "heat" if gamma == 0 else "advection_diffusion"
    return SpecialPDEResult(solution_family=family, family_name=name)


def recognize_laplace_or_helmholtz(
    eq: sp.Equality | sp.Expr, u: sp.Function, vars: tuple[sp.Symbol, sp.Symbol]
) -> Optional[SpecialPDEResult]:
    """Recognize 2D Laplace/Helmholtz equations and return a separated family."""
    x, y = vars
    expr = sp.simplify((eq.lhs - eq.rhs) if isinstance(eq, sp.Equality) else eq)
    uxy = u(x, y)
    uxx = sp.diff(uxy, x, 2)
    uyy = sp.diff(uxy, y, 2)
    a_xx = sp.simplify(expr.coeff(uxx))
    expr = sp.simplify(expr - a_xx * uxx)
    a_yy = sp.simplify(expr.coeff(uyy))
    expr = sp.simplify(expr - a_yy * uyy)
    a_u = sp.simplify(expr.coeff(uxy))
    expr = sp.simplify(expr - a_u * uxy)
    if expr != 0 or a_xx == 0 or a_yy == 0:
        return None

    if sp.simplify(a_xx - a_yy) != 0:
        return None
    k = sp.simplify(a_u / a_xx)
    if not k.free_symbols.isdisjoint({x, y}):
        return None

    mu = sp.symbols("mu")
    A, B = sp.symbols("A B")
    if k == 0:
        family = sp.exp(mu * x) * (A * sp.cos(mu * y) + B * sp.sin(mu * y))
        return SpecialPDEResult(solution_family=family, family_name="laplace_2d")

    lam = sp.sqrt(k + mu**2)
    family = sp.exp(mu * x) * (A * sp.cos(lam * y) + B * sp.sin(lam * y))
    return SpecialPDEResult(solution_family=family, family_name="helmholtz_2d")


def solve_special_pde(
    eq: sp.Equality | sp.Expr, u: sp.Function, vars: tuple[sp.Symbol, sp.Symbol]
) -> Optional[SpecialPDEResult]:
    """Try the lightweight special-family recognizers."""
    return recognize_heat_or_advection_diffusion(
        eq, u, vars
    ) or recognize_laplace_or_helmholtz(eq, u, vars)
