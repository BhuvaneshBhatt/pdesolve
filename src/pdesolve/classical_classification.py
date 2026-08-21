from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.assumptions import assuming

from .classical_symbolic_helpers import _as_zero_expr, _dep_and_vars


@dataclass(frozen=True)
class SecondOrderLinearType2D:
    indep_vars: tuple[sp.Symbol, sp.Symbol]
    dep_function: sp.Expr
    A: sp.Expr
    B: sp.Expr
    C: sp.Expr
    discriminant: sp.Expr
    classification: str
    normalized_equation: sp.Equality


@dataclass(frozen=True)
class LinearSecondOrderPDEClassification:
    indep_vars: tuple[sp.Symbol, ...]
    dep_function: sp.Expr
    coefficient_matrix: sp.Matrix
    eigenvalues: tuple[sp.Expr, ...]
    sign_pattern: tuple[sp.Expr | int, ...]
    classification: str
    normalized_equation: sp.Equality


def _extract_linear_second_order_coefficient_matrix(eq_or_expr, dep_expr_or_func, indep_vars=None):
    """
    Extract the symmetric principal coefficient matrix for a scalar linear
    second-order PDE in n independent variables.

    For
        sum_{i,j} a_{ij} u_{x_i x_j} + lower-order terms = 0,
    returns the symmetric matrix A = (a_{ij}) where mixed-derivative
    coefficients are halved in the standard way.
    """
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    xs = tuple(vars_)
    n = len(xs)
    zero = _as_zero_expr(eq_or_expr)

    # Create placeholders for derivatives up to second order.
    deriv_syms = {}
    replacements = {uexpr: sp.Symbol("U0")}
    linear_vars = [replacements[uexpr]]
    for i in range(n):
        d1 = sp.Symbol(f"U1_{i}")
        replacements[sp.diff(uexpr, xs[i])] = d1
        linear_vars.append(d1)
    for i in range(n):
        for j in range(i, n):
            d2 = sp.Symbol(f"U2_{i}_{j}")
            replacements[sp.diff(uexpr, xs[i], xs[j])] = d2
            deriv_syms[(i, j)] = d2
            linear_vars.append(d2)

    expr = sp.expand(zero.xreplace(replacements))

    # If any derivative of u remains, we failed to model the PDE as scalar second-order linear.
    if any(
        isinstance(node, sp.Derivative) and node.expr == uexpr
        for node in sp.preorder_traversal(expr)
    ):
        raise ValueError(
            "Could not eliminate derivative objects while extracting the principal matrix."
        )

    try:
        poly = sp.Poly(expr, *linear_vars, domain="EX")
    except Exception as exc:
        raise ValueError("Could not treat PDE as polynomial in derivative placeholders.") from exc

    if poly.total_degree() > 1:
        raise ValueError("PDE is not linear in u and its derivatives up to order 2.")

    # Build the symmetric coefficient matrix.
    A = sp.zeros(n, n)
    for i in range(n):
        A[i, i] = sp.expand(sp.diff(expr, deriv_syms[(i, i)]))
    for i in range(n):
        for j in range(i + 1, n):
            coeff = sp.expand(sp.diff(expr, deriv_syms[(i, j)]))
            A[i, j] = sp.expand(coeff / 2)
            A[j, i] = sp.expand(coeff / 2)

    return uexpr, xs, A, sp.Eq(sp.expand(zero), 0)


def _sign_with_assumptions(expr, assumptions=True):
    expr = sp.simplify(expr)

    def _sign_of(e):
        s = sp.simplify(sp.sign(e))
        if s in (sp.Integer(-1), sp.Integer(0), sp.Integer(1)):
            return int(s)
        return s

    if expr.is_real is True:
        s = _sign_of(expr)
        if s in (-1, 0, 1):
            return s

    if assumptions not in (True, None):
        try:
            with assuming(assumptions):
                expr_refined = sp.refine(expr)
                s = _sign_of(expr_refined)
                if s in (-1, 0, 1):
                    return s
        except Exception:
            pass

        try:
            expr_refined = sp.refine(expr, assumptions)
            s = _sign_of(expr_refined)
            if s in (-1, 0, 1):
                return s
        except Exception:
            pass

    s = _sign_of(expr)
    return s


def _classify_signature(sign_pattern):
    signs = list(sign_pattern)
    if any(s not in (-1, 0, 1) for s in signs):
        return "indeterminate"

    pos = sum(1 for s in signs if s == 1)
    neg = sum(1 for s in signs if s == -1)
    zero = sum(1 for s in signs if s == 0)
    n = len(signs)

    if zero > 0:
        return "parabolic"
    if pos == n or neg == n:
        return "elliptic"
    if pos == 1 or neg == 1 or pos == n - 1 or neg == n - 1:
        return "hyperbolic"
    if 1 < pos < n - 1:
        return "ultrahyperbolic"
    return "indeterminate"


def classify_linear_second_order_pde(
    eq_or_expr, dep_expr_or_func, indep_vars=None, *, assumptions=True
) -> LinearSecondOrderPDEClassification:
    """
    Classify a scalar linear second-order PDE in any number of variables using
    the signature of the symmetric principal coefficient matrix.

    Returns one of:
      - elliptic
      - hyperbolic
      - parabolic
      - ultrahyperbolic
      - indeterminate
    """
    uexpr, xs, A, normalized = _extract_linear_second_order_coefficient_matrix(
        eq_or_expr, dep_expr_or_func, indep_vars
    )

    # Eigenvalues determine the type. For exact symbolic matrices, SymPy may
    # return algebraic expressions; classify by their signs when possible.
    try:
        eig_dict = A.eigenvals()
    except Exception:
        eig_dict = A.eigenvals(simplify=True)
    eigs = []
    for ev, mult in eig_dict.items():
        eigs.extend([ev] * int(mult))

    sign_pattern = tuple(_sign_with_assumptions(ev, assumptions=assumptions) for ev in eigs)
    classification = _classify_signature(sign_pattern)

    return LinearSecondOrderPDEClassification(
        indep_vars=tuple(xs),
        dep_function=uexpr,
        coefficient_matrix=A,
        eigenvalues=tuple(sp.simplify(ev) for ev in eigs),
        sign_pattern=sign_pattern,
        classification=classification,
        normalized_equation=normalized,
    )


def classify_second_order_linear_pde_2vars(
    eq_or_expr, dep_expr_or_func, indep_vars=None
) -> SecondOrderLinearType2D:
    """
    Two-dimensional wrapper using the general linear second-order PDE
    classifier and also exposing the traditional A, B, C discriminant data.
    """
    uexpr, vars_, A_mat, normalized = _extract_linear_second_order_coefficient_matrix(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    if len(vars_) != 2:
        raise ValueError("Second-order type classification is for two independent variables.")
    A = sp.expand(A_mat[0, 0])
    B = sp.expand(2 * A_mat[0, 1])
    C = sp.expand(A_mat[1, 1])

    discriminant = sp.simplify(B**2 - 4 * A * C)
    general = classify_linear_second_order_pde(eq_or_expr, dep_expr_or_func, vars_)
    cls = general.classification
    if cls == "indeterminate":
        if discriminant.is_zero:
            cls = "parabolic"
        elif discriminant.is_positive:
            cls = "hyperbolic"
        elif discriminant.is_negative:
            cls = "elliptic"
        else:
            sdisc = sp.simplify(discriminant)
            if sdisc == 0:
                cls = "parabolic"
            else:
                cls = "symbolic"

    return SecondOrderLinearType2D(tuple(vars_), uexpr, A, B, C, discriminant, cls, normalized)


__all__ = [
    "SecondOrderLinearType2D",
    "LinearSecondOrderPDEClassification",
    "classify_linear_second_order_pde",
    "classify_second_order_linear_pde_2vars",
]
