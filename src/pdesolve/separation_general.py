from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class ProductSeparationResult:
    ansatz: sp.Equality
    separated_expression: sp.Expr
    separation_constant: sp.Symbol
    factor_equations: tuple[sp.Equality, ...]
    factors: tuple[sp.Expr, ...]
    variables: tuple[sp.Symbol, ...]
    verified_separable: bool


def _depends_only(expr, allowed):
    allowed = set(allowed)
    return expr.free_symbols <= allowed


def separate_product_pde(
    equation, dep_function, indep_vars, *, separation_constant=None
) -> ProductSeparationResult:
    """Attempt multiplicative separation for a scalar homogeneous PDE.

    This intentionally targets the robust two-variable case.  It derives the
    separated ODEs from the supplied PDE rather than recognizing a named PDE.
    """
    vars_ = tuple(indep_vars)
    if len(vars_) != 2:
        raise NotImplementedError(
            "general product separation currently supports exactly two independent variables"
        )
    x, y = vars_
    u = (
        dep_function
        if getattr(dep_function, "is_Function", False) and dep_function.args
        else dep_function(*vars_)
    )
    X = sp.Function(f"{u.func.__name__}_{x}")(x)
    Y = sp.Function(f"{u.func.__name__}_{y}")(y)
    prod = X * Y
    zero = (
        equation.lhs - equation.rhs if isinstance(equation, sp.Equality) else sp.sympify(equation)
    )
    replaced = zero.xreplace({u: prod}).doit()
    reduced = sp.factor_terms(sp.cancel(replaced / prod))
    parts = sp.Add.make_args(sp.expand(reduced))
    xparts, yparts, constants = [], [], []
    for term in parts:
        # Applied functions/derivatives don't appear in free_symbols; use has instead.
        has_x = term.has(x, X) or any(
            isinstance(n, sp.Derivative) and n.expr.has(X) for n in sp.preorder_traversal(term)
        )
        has_y = term.has(y, Y) or any(
            isinstance(n, sp.Derivative) and n.expr.has(Y) for n in sp.preorder_traversal(term)
        )
        if has_x and not has_y:
            xparts.append(term)
        elif has_y and not has_x:
            yparts.append(term)
        elif not has_x and not has_y:
            constants.append(term)
        else:
            raise ValueError(
                f"PDE is not multiplicatively separable by this engine; coupled term: {term}"
            )
    if constants:
        xparts.extend(constants)
    if not xparts or not yparts:
        raise ValueError("could not split substituted PDE into independent-variable factors")
    A = sp.simplify(sum(xparts))
    B = sp.simplify(sum(yparts))
    lam = separation_constant or sp.Symbol("lambda", real=True)
    eqx = sp.Eq(A, lam, evaluate=False)
    eqy = sp.Eq(B, -lam, evaluate=False)
    return ProductSeparationResult(
        sp.Eq(u, prod, evaluate=False), reduced, lam, (eqx, eqy), (X, Y), vars_, True
    )


__all__ = ["ProductSeparationResult", "separate_product_pde"]
