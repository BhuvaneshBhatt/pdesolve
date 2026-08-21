from __future__ import annotations

from dataclasses import dataclass
import sympy as sp

from .family_recognizers import _canonical_linear_pde_1d_xt


@dataclass(frozen=True)
class SeparationOfVariablesResult:
    method: str
    normalized_equation: sp.Equality
    ansatz: sp.Equality
    separated_expression: sp.Expr
    separation_constant: sp.Symbol
    x_equation: sp.Equality
    t_equation: sp.Equality
    details: dict


def separate_variables(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    separation_constant=None,
    assumptions=True,
):
    """
    Attempt separation of variables for common linear homogeneous 1+1 PDE classes.

    Supported starter classes:
      - heat/reaction-diffusion type: u_t = k u_xx + q u
      - wave/Klein-Gordon type:      u_tt = c^2 u_xx + q u
      - Laplace/Helmholtz type in two vars: u_xx + a u_yy + q u = 0

    Returns ODEs for X(x) and T(t) under u(x,t)=X(x)T(t).
    """
    can = _canonical_linear_pde_1d_xt(
        eq_or_expr, dep_expr_or_func, indep_vars, assumptions=assumptions
    )
    if can is None:
        raise NotImplementedError(
            "No implemented separation-of-variables pattern matched this PDE."
        )

    x, t = can["vars"]
    uexpr = can["u"]
    X = sp.Function("X")
    T = sp.Function("T")
    lam = (
        separation_constant
        if separation_constant is not None
        else sp.Symbol("lambda_sep")
    )
    ansatz_expr = X(x) * T(t)

    if can["kind"] == "heat_like":
        if sp.simplify(can["forcing"]) != 0:
            raise NotImplementedError(
                "Current separation helper only handles homogeneous heat-like PDEs."
            )
        expr = sp.expand(
            sp.diff(ansatz_expr, t)
            - can["kappa"] * sp.diff(ansatz_expr, x, 2)
            - can["q"] * ansatz_expr
        )
        sep = sp.simplify(sp.factor(expr / ansatz_expr))
        xeq = sp.Eq(can["kappa"] * sp.diff(X(x), x, 2) + (can["q"] - lam) * X(x), 0)
        teq = sp.Eq(sp.diff(T(t), t) - lam * T(t), 0)
        return SeparationOfVariablesResult(
            "separation_heat_like",
            can["normalized"],
            sp.Eq(uexpr, ansatz_expr),
            sep,
            lam,
            xeq,
            teq,
            can,
        )

    if can["kind"] == "wave_like":
        if sp.simplify(can["forcing"]) != 0:
            raise NotImplementedError(
                "Current separation helper only handles homogeneous wave-like PDEs."
            )
        expr = sp.expand(
            sp.diff(ansatz_expr, t, 2)
            - can["c2"] * sp.diff(ansatz_expr, x, 2)
            - can["q"] * ansatz_expr
        )
        sep = sp.simplify(sp.factor(expr / ansatz_expr))
        xeq = sp.Eq(can["c2"] * sp.diff(X(x), x, 2) + (can["q"] - lam) * X(x), 0)
        teq = sp.Eq(sp.diff(T(t), t, 2) - lam * T(t), 0)
        return SeparationOfVariablesResult(
            "separation_wave_like",
            can["normalized"],
            sp.Eq(uexpr, ansatz_expr),
            sep,
            lam,
            xeq,
            teq,
            can,
        )

    if can["kind"] == "laplace_helmholtz_like":
        if sp.simplify(can["forcing"]) != 0:
            raise NotImplementedError(
                "Current separation helper only handles homogeneous Laplace/Helmholtz-like PDEs."
            )
        y = t
        expr = sp.expand(
            sp.diff(ansatz_expr, x, 2)
            + can["ay"] * sp.diff(ansatz_expr, y, 2)
            + can["lambda0"] * ansatz_expr
        )
        sep = sp.simplify(sp.factor(expr / ansatz_expr))
        xeq = sp.Eq(sp.diff(X(x), x, 2) + (can["lambda0"] + lam) * X(x), 0)
        teq = sp.Eq(can["ay"] * sp.diff(T(y), y, 2) - lam * T(y), 0)
        return SeparationOfVariablesResult(
            "separation_laplace_helmholtz_like",
            can["normalized"],
            sp.Eq(uexpr, ansatz_expr),
            sep,
            lam,
            xeq,
            teq,
            can,
        )

    raise NotImplementedError(
        "Matched PDE type is not yet handled by the separation engine."
    )


__all__ = ["SeparationOfVariablesResult", "separate_variables"]
