"""Geometric helpers for first-order PDE reductions.

This module contains the characteristic first-integral search and the adapted
coordinate reduction object used by the linear first-order solver family.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

# First-integral search -----------------------------------------------------


@dataclass(frozen=True)
class AdaptedCoordinateReduction:
    """Description of a simple adapted-coordinate reduction."""

    invariant: sp.Expr
    transverse_var: sp.Symbol
    param: sp.Symbol
    subst_map: dict[sp.Symbol, sp.Expr]
    coeff: sp.Expr


def characteristic_first_integral(
    a: sp.Expr, b: sp.Expr, x: sp.Symbol, y: sp.Symbol
) -> sp.Expr | None:
    """Return a simple first integral for the field ``(a, b)``.

    The implementation intentionally focuses on a few robust patterns:
    constant fields, exact one-forms, and characteristic ratios depending on a
    single variable.
    """
    a = sp.simplify(a)
    b = sp.simplify(b)

    if a == 0 and b == 0:
        return None

    if a.free_symbols.isdisjoint({x, y}) and b.free_symbols.isdisjoint({x, y}):
        if a == 0:
            return x
        if b == 0:
            return y
        return sp.simplify(b * x - a * y)

    one_form_x = sp.simplify(b)
    one_form_y = sp.simplify(-a)
    if sp.simplify(sp.diff(one_form_x, y) - sp.diff(one_form_y, x)) == 0:
        try:
            pot = sp.integrate(one_form_x, x)
            corr = sp.integrate(sp.simplify(one_form_y - sp.diff(pot, y)), y)
            return sp.simplify(pot + corr)
        except Exception:
            pass

    if a != 0:
        ratio = sp.simplify(b / a)
        if ratio.free_symbols.isdisjoint({y}):
            try:
                return sp.simplify(y - sp.integrate(ratio, x))
            except Exception:
                pass
        if ratio.free_symbols.isdisjoint({x}):
            try:
                inv_ratio = sp.simplify(1 / ratio)
                return sp.simplify(sp.integrate(inv_ratio, y) - x)
            except Exception:
                pass
    return None


def adapted_coordinate_reduction(
    a: sp.Expr, b: sp.Expr, x: sp.Symbol, y: sp.Symbol
) -> AdaptedCoordinateReduction | None:
    """Construct a simple adapted-coordinate reduction from a first integral."""
    inv = characteristic_first_integral(a, b, x, y)
    if inv is None:
        return None

    eta = sp.Symbol("eta", real=True)

    # Prefer x as the characteristic parameter when possible.
    if sp.simplify(a) != 0:
        try:
            y_sol = sp.solve(sp.Eq(inv, eta), y, dict=True)
            if y_sol:
                return AdaptedCoordinateReduction(
                    invariant=inv,
                    transverse_var=x,
                    param=eta,
                    subst_map=y_sol[0],
                    coeff=sp.simplify(a),
                )
        except Exception:
            pass

    if sp.simplify(b) != 0:
        try:
            x_sol = sp.solve(sp.Eq(inv, eta), x, dict=True)
            if x_sol:
                return AdaptedCoordinateReduction(
                    invariant=inv,
                    transverse_var=y,
                    param=eta,
                    subst_map=x_sol[0],
                    coeff=sp.simplify(b),
                )
        except Exception:
            pass
    return None


__all__ = [
    "AdaptedCoordinateReduction",
    "characteristic_first_integral",
    "adapted_coordinate_reduction",
]
