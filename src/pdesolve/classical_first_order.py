from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.solvers.pde import classify_pde, pdsolve

from ._classical_shared import _as_zero_expr, _dep_and_vars


@dataclass(frozen=True)
class FirstOrderCharacteristicForm:
    indep_vars: tuple[sp.Symbol, sp.Symbol]
    dep_function: sp.Expr
    A: sp.Expr
    B: sp.Expr
    C: sp.Expr
    normalized_equation: sp.Equality
    is_quasilinear: bool
    is_constant_coefficient: bool


@dataclass(frozen=True)
class PDEIVPResult:
    method: str
    solution: sp.Expr | sp.Equality
    details: dict

    @property
    def solved(self) -> bool:
        return bool(self.details.get("solved", True))


def characteristic_form_first_order_2vars(
    eq_or_expr, dep_expr_or_func, indep_vars=None
) -> FirstOrderCharacteristicForm:
    """
    Detect a broad first-order PDE class in two variables of the form
        A(x,y,u) u_x + B(x,y,u) u_y = C(x,y,u)
    where the equation is linear in first derivatives.
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    if len(vars_) != 2:
        raise ValueError(
            "This characteristic detector is for two independent variables."
        )
    x, y = vars_
    zero = _as_zero_expr(eq_or_expr)
    ux = sp.diff(uexpr, x)
    uy = sp.diff(uexpr, y)

    # Require that no higher-order derivatives of u appear.
    for node in sp.preorder_traversal(zero):
        if isinstance(node, sp.Derivative) and node.expr == uexpr:
            total_order = sum(count for _, count in node.variable_count)
            if total_order > 1:
                raise ValueError(
                    "PDE contains higher-order derivatives and is not a broad first-order characteristic problem."
                )

    try:
        poly = sp.Poly(zero, ux, uy, domain="EX")
    except Exception as exc:
        raise ValueError(
            "Could not treat PDE as polynomial in first derivatives."
        ) from exc

    # Require degree at most 1 in ux, uy.
    if poly.total_degree() > 1:
        raise ValueError("PDE is not linear in the first derivatives.")

    A = sp.expand(sp.diff(zero, ux))
    B = sp.expand(sp.diff(zero, uy))
    C = sp.expand(-(zero.subs({ux: 0, uy: 0})))

    # Ensure coefficients themselves do not still contain ux or uy.
    for coeff in (A, B, C):
        if ux in coeff.free_symbols or uy in coeff.free_symbols:
            raise ValueError("Could not isolate a quasilinear first-order form.")

    normalized = sp.Eq(sp.expand(A * ux + B * uy), sp.expand(C))
    is_const = all(v.free_symbols.isdisjoint({x, y, uexpr}) for v in (A, B, C))
    return FirstOrderCharacteristicForm(
        vars_, uexpr, A, B, C, normalized, True, is_const
    )


def solve_first_order_pde_characteristic(eq_or_expr, dep_expr_or_func, indep_vars=None):
    """
    Broad first-order PDE solve hook.

    Strategy:
      1. detect quasilinear first-order form in two variables,
      2. try SymPy's pdsolve,
      3. return a structured result.
    """
    form = characteristic_form_first_order_2vars(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    try:
        hints = classify_pde(form.normalized_equation)
    except Exception:
        hints = ()

    try:
        sol = pdsolve(form.normalized_equation)
        return PDEIVPResult(
            method="pdsolve_first_order",
            solution=sol,
            details={"classification_hints": tuple(hints), "characteristic_form": form},
        )
    except Exception as exc:
        raise NotImplementedError(
            "Could not solve the first-order PDE with current characteristic methods."
        ) from exc


def solve_transport_ivp(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    initial_curve_value=0,
    initial_profile=None,
):
    """
    Solve a constant-coefficient transport/advection-reaction PDE with initial data.

    Supported classes:
      A u_x + B u_t = C                  with constant A,B,C
      A u_x + B u_t + D u = E            with constant A,B,D,E

    Initial data are supplied on the line corresponding to the second independent
    variable equal to `initial_curve_value`, typically t = 0.

    `initial_profile` should be a callable or SymPy expression in the first variable.
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    if len(vars_) != 2:
        raise ValueError("Transport IVP helper is for two independent variables.")
    x, t = vars_
    zero = _as_zero_expr(eq_or_expr)
    ux = sp.diff(uexpr, x)
    ut = sp.diff(uexpr, t)

    for node in sp.preorder_traversal(zero):
        if isinstance(node, sp.Derivative) and node.expr == uexpr:
            total_order = sum(count for _, count in node.variable_count)
            if total_order > 1:
                raise NotImplementedError(
                    "Current transport IVP helper only supports first-order PDEs."
                )

    A = sp.expand(sp.diff(zero, ux))
    B = sp.expand(sp.diff(zero, ut))
    residual = sp.expand(zero - A * ux - B * ut)

    # residual = D*u - E, with constants D,E
    D = sp.expand(sp.diff(residual, uexpr))
    E = sp.expand(-(residual.subs({uexpr: 0})))

    if any(not coeff.free_symbols.isdisjoint({x, t, uexpr}) for coeff in (A, B, D, E)):
        raise NotImplementedError(
            "Current IVP transport helper supports only constant coefficients."
        )

    if sp.simplify(B) == 0:
        raise NotImplementedError(
            "Need a nonzero coefficient for the evolution variable in the transport IVP helper."
        )

    if initial_profile is None:
        raise ValueError("An initial_profile is required for the transport IVP helper.")

    xi = sp.simplify(x - (A / B) * (t - initial_curve_value))

    if callable(initial_profile):
        base = initial_profile(xi)
    else:
        g = sp.sympify(initial_profile)
        if isinstance(g, sp.Expr):
            if g.free_symbols == {x}:
                base = g.subs(x, xi)
            else:
                base = g
        else:
            base = g

    if sp.simplify(D) == 0:
        # u_t + (A/B) u_x = E/B
        sol = sp.expand(base + (E / B) * (t - initial_curve_value))
    else:
        phase = sp.exp(-(D / B) * (t - initial_curve_value))
        steady = sp.simplify(E / D)
        sol = sp.expand(steady + (base - steady) * phase)

    return PDEIVPResult(
        method="constant_coefficient_characteristics_ivp",
        solution=sp.Eq(uexpr, sol),
        details={"A": A, "B": B, "D": D, "E": E, "characteristic_coordinate": xi},
    )


__all__ = [
    "FirstOrderCharacteristicForm",
    "PDEIVPResult",
    "characteristic_form_first_order_2vars",
    "solve_first_order_pde_characteristic",
    "solve_transport_ivp",
]
