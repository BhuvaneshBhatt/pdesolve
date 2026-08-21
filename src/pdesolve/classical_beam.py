from __future__ import annotations

from dataclasses import dataclass
import sympy as sp

from ._classical_shared import _dep_and_vars, _safe_sub_profile_general


@dataclass(frozen=True)
class SpectralPDEResult:
    method: str
    solution: sp.Equality
    details: dict


def solve_simply_supported_beam_ibvp(
    dep_expr_or_func,
    *,
    x,
    t,
    length,
    stiffness=1,
    damping=0,
    initial_displacement=None,
    initial_velocity=None,
    n_terms=8,
):
    uexpr, _ = _dep_and_vars(dep_expr_or_func, (x, t))
    L = sp.sympify(length)
    c = sp.sympify(stiffness)
    gamma = sp.sympify(damping)
    if initial_displacement is None or initial_velocity is None:
        raise ValueError("Both initial_displacement and initial_velocity are required.")
    f = _safe_sub_profile_general(initial_displacement, x)
    g = _safe_sub_profile_general(initial_velocity, x)
    n = sp.Symbol("n", integer=True, positive=True)
    lam = (n * sp.pi / L) ** 2
    omega = sp.sqrt(4 * c * lam**2 - gamma**2) / 2
    an = sp.sqrt(sp.Integer(2) / L) * sp.integrate(
        f * sp.sin(n * sp.pi * x / L), (x, 0, L)
    )
    bn = (
        sp.sqrt(sp.Integer(2) / L)
        * sp.integrate((gamma * f / 2 + g) * sp.sin(n * sp.pi * x / L), (x, 0, L))
        / omega
    )
    basis = sp.sqrt(sp.Integer(2) / L) * sp.sin(n * sp.pi * x / L)
    term = (
        sp.exp(-gamma * t / 2)
        * basis
        * (an * sp.cos(omega * t) + bn * sp.sin(omega * t))
    )
    series = sp.Sum(term, (n, 1, n_terms))
    return SpectralPDEResult(
        "simply_supported_beam_spectral",
        sp.Eq(uexpr, series),
        {"length": L, "stiffness": c, "damping": gamma, "basis": "sine"},
    )


__all__ = ["SpectralPDEResult", "solve_simply_supported_beam_ibvp"]
