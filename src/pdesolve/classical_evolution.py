from __future__ import annotations

import sympy as sp

from ._classical_shared import _dep_and_vars
from .classical_first_order import PDEIVPResult


def solve_wave_equation_1d_ivp(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    wave_speed=1,
    initial_displacement=None,
    initial_velocity=None,
):
    """
    d'Alembert solution of
        u_tt = c^2 u_xx
    with
        u(x,0) = f(x),
        u_t(x,0) = g(x).

    The returned solution may contain an unevaluated integral for the velocity term.
    """
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    if len(vars_) != 2:
        raise ValueError("Wave IVP helper expects two variables (x,t).")
    x, t = vars_
    c = sp.sympify(wave_speed)

    if initial_displacement is None:
        raise ValueError("initial_displacement is required.")
    if initial_velocity is None:

        def initial_velocity(_):
            return 0

    s = sp.Symbol("s", real=True)

    def _apply_profile(profile, arg):
        if callable(profile):
            return profile(arg)
        expr = sp.sympify(profile)
        free = list(expr.free_symbols)
        if len(free) == 1:
            return expr.subs(free[0], arg)
        return expr

    fterm = (
        _apply_profile(initial_displacement, x - c * t)
        + _apply_profile(initial_displacement, x + c * t)
    ) / 2
    gterm = sp.Integral(
        _apply_profile(initial_velocity, s), (s, x - c * t, x + c * t)
    ) / (2 * c)
    sol = sp.expand(fterm + gterm)

    return PDEIVPResult(
        method="dAlembert_wave_ivp",
        solution=sp.Eq(uexpr, sol),
        details={"wave_speed": c},
    )


def solve_heat_equation_1d_whole_line_ivp(
    dep_expr_or_func, *, x=None, t=None, diffusivity=1, initial_profile=None
):
    """
    Heat kernel solution of
        u_t = k u_xx
    on the whole line with initial profile phi(x).

    Returns the formal convolution integral.
    """
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    if len(vars_) != 2:
        raise ValueError("Heat IVP helper expects two variables (x,t).")
    x, t = vars_
    kappa = sp.sympify(diffusivity)
    if initial_profile is None:
        raise ValueError("initial_profile is required.")

    xi = sp.Symbol("xi", real=True)
    if callable(initial_profile):
        phi = initial_profile(xi)
    else:
        expr = sp.sympify(initial_profile)
        free = list(expr.free_symbols)
        phi = expr.subs(free[0], xi) if len(free) == 1 else expr

    kernel = sp.exp(-((x - xi) ** 2) / (4 * kappa * t)) / sp.sqrt(4 * sp.pi * kappa * t)
    sol = sp.Integral(kernel * phi, (xi, -sp.oo, sp.oo))

    return PDEIVPResult(
        method="heat_kernel_whole_line_ivp",
        solution=sp.Eq(uexpr, sol),
        details={"diffusivity": kappa},
    )


def solve_heat_equation_1d_dirichlet_series(
    dep_expr_or_func,
    *,
    x=None,
    t=None,
    diffusivity=1,
    length=sp.pi,
    initial_profile=None,
    terms=6,
):
    """
    Fourier-sine series solution of
        u_t = k u_xx,  0 < x < L,
        u(0,t)=u(L,t)=0,
        u(x,0)=phi(x).

    This returns a truncated symbolic series with coefficients represented by
    integrals when the initial profile is symbolic.
    """
    uexpr, vars_ = _dep_and_vars(
        dep_expr_or_func, (x, t) if x is not None and t is not None else None
    )
    if len(vars_) != 2:
        raise ValueError("Heat Dirichlet-series helper expects two variables (x,t).")
    x, t = vars_
    kappa = sp.sympify(diffusivity)
    L = sp.sympify(length)
    if initial_profile is None:
        raise ValueError("initial_profile is required.")

    xi = sp.Symbol("xi", real=True)
    if callable(initial_profile):
        phi = initial_profile(xi)
    else:
        expr = sp.sympify(initial_profile)
        free = list(expr.free_symbols)
        phi = expr.subs(free[0], xi) if len(free) == 1 else expr

    series = 0
    for n in range(1, int(terms) + 1):
        bn = 2 / L * sp.Integral(phi * sp.sin(n * sp.pi * xi / L), (xi, 0, L))
        series += (
            bn * sp.sin(n * sp.pi * x / L) * sp.exp(-kappa * (n * sp.pi / L) ** 2 * t)
        )

    return PDEIVPResult(
        method="heat_dirichlet_sine_series",
        solution=sp.Eq(uexpr, sp.simplify(series)),
        details={"diffusivity": kappa, "length": L, "terms": int(terms)},
    )


__all__ = [
    "solve_wave_equation_1d_ivp",
    "solve_heat_equation_1d_whole_line_ivp",
    "solve_heat_equation_1d_dirichlet_series",
]
