from __future__ import annotations

from .first_order_linear import solve_first_order_linear_pde
from .first_order_nonlinear import solve_first_order_quasilinear_pde


def solve_first_order_pde(eq, u, vars):
    try:
        return solve_first_order_linear_pde(eq, u, vars)
    except Exception:
        return solve_first_order_quasilinear_pde(eq, u, vars)
