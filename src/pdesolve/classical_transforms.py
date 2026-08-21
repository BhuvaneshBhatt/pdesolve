from __future__ import annotations

from dataclasses import dataclass
import sympy as sp

from ._classical_shared import _dep_and_vars


@dataclass(frozen=True)
class TransformMethodResult:
    method: str
    transformed_equation: sp.Equality | sp.Expr
    solution: sp.Expr | sp.Equality
    details: dict


def solve_heat_equation_1d_fourier_transform(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    diffusivity=1,
    initial_profile=None,
    fourier_variable=None,
):
    """
    Fourier-transform solution of u_t = k u_xx on the whole line.

    Returns both the transformed solution
        Uhat(w,t) = exp(-k w^2 t) Uhat0(w)
    and the inverse-transform representation.
    """
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    if len(vars_) != 2:
        raise ValueError("Expected two variables (x,t).")
    x, t = vars_
    kappa = sp.sympify(diffusivity)
    if initial_profile is None:
        raise ValueError("initial_profile is required.")
    w = (
        sp.Symbol("w", real=True)
        if fourier_variable is None
        else sp.sympify(fourier_variable)
    )
    xi = sp.Symbol("xi", real=True)

    if callable(initial_profile):
        phi = initial_profile(xi)
    else:
        expr = sp.sympify(initial_profile)
        free = list(expr.free_symbols)
        phi = expr.subs(free[0], xi) if len(free) == 1 else expr

    U0 = sp.Integral(phi * sp.exp(-sp.I * w * xi), (xi, -sp.oo, sp.oo))
    Uhat = sp.exp(-kappa * w**2 * t) * U0
    inv = sp.Integral(Uhat * sp.exp(sp.I * w * x), (w, -sp.oo, sp.oo)) / (2 * sp.pi)

    return TransformMethodResult(
        method="fourier_heat_whole_line",
        transformed_equation=sp.Eq(sp.Function("Uhat")(w, t), Uhat),
        solution=sp.Eq(uexpr, inv),
        details={"diffusivity": kappa, "fourier_variable": w},
    )


def solve_advection_equation_1d_fourier_transform(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    speed=1,
    reaction=0,
    initial_profile=None,
    fourier_variable=None,
):
    """
    Fourier-transform solution of
        u_t + c u_x + r u = 0
    on the whole line.
    """
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    if len(vars_) != 2:
        raise ValueError("Expected two variables (x,t).")
    x, t = vars_
    c = sp.sympify(speed)
    r = sp.sympify(reaction)
    if initial_profile is None:
        raise ValueError("initial_profile is required.")
    w = (
        sp.Symbol("w", real=True)
        if fourier_variable is None
        else sp.sympify(fourier_variable)
    )
    xi = sp.Symbol("xi", real=True)

    if callable(initial_profile):
        phi = initial_profile(xi)
    else:
        expr = sp.sympify(initial_profile)
        free = list(expr.free_symbols)
        phi = expr.subs(free[0], xi) if len(free) == 1 else expr

    U0 = sp.Integral(phi * sp.exp(-sp.I * w * xi), (xi, -sp.oo, sp.oo))
    Uhat = sp.exp(-(sp.I * c * w + r) * t) * U0
    inv = sp.Integral(Uhat * sp.exp(sp.I * w * x), (w, -sp.oo, sp.oo)) / (2 * sp.pi)

    return TransformMethodResult(
        method="fourier_advection_whole_line",
        transformed_equation=sp.Eq(sp.Function("Uhat")(w, t), Uhat),
        solution=sp.Eq(uexpr, inv),
        details={"speed": c, "reaction": r, "fourier_variable": w},
    )


__all__ = [
    "TransformMethodResult",
    "solve_heat_equation_1d_fourier_transform",
    "solve_advection_equation_1d_fourier_transform",
]
