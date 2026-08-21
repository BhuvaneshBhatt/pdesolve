from __future__ import annotations

import sympy as sp


def _expr_complexity(expr):
    try:
        return int(sp.count_ops(sp.expand(expr), visual=False))
    except Exception:
        return 10**6


def _as_zero_expr(eq_or_expr):
    if isinstance(eq_or_expr, sp.Equality):
        return sp.expand(eq_or_expr.lhs - eq_or_expr.rhs)
    return sp.expand(eq_or_expr)


def _dep_and_vars(dep_expr_or_func, indep_vars=None):
    if isinstance(dep_expr_or_func, sp.Expr) and getattr(dep_expr_or_func, "is_Function", False):
        dep_expr = dep_expr_or_func
        vars_ = tuple(dep_expr.args) if indep_vars is None else tuple(indep_vars)
    else:
        if indep_vars is None:
            raise ValueError("indep_vars must be provided when dep function is not applied.")
        vars_ = tuple(indep_vars)
        dep_expr = dep_expr_or_func(*vars_)
    return dep_expr, vars_


def _safe_sub_profile(expr_or_callable, sym):
    if callable(expr_or_callable):
        return expr_or_callable(sym)
    expr = sp.sympify(expr_or_callable)
    free = list(expr.free_symbols)
    if len(free) == 1:
        return expr.subs(free[0], sym)
    return expr


def _safe_sub_profile_general(profile, var):
    if profile is None:
        return sp.Integer(0)
    if callable(profile):
        return sp.sympify(profile(var))
    expr = sp.sympify(profile)
    free = list(expr.free_symbols)
    if len(free) == 1:
        return expr.subs(free[0], var)
    return expr
