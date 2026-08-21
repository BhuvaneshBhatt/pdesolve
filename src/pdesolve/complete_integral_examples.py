from __future__ import annotations

import sympy as sp

from .complete_integral_helpers import (
    solve_charpit_complete_integral_2vars,
    solve_jacobi_complete_integral,
)


def run_demo():
    x, y = sp.symbols("x y", positive=True, real=True)
    u = sp.Function("u")

    examples = []
    # autonomous eikonal-like
    eq1 = sp.Eq(sp.diff(u(x, y), x) ** 2 + sp.diff(u(x, y), y) ** 2, 1)
    examples.append(
        ("Charpit autonomous", solve_charpit_complete_integral_2vars(eq1, u(x, y), (x, y)))
    )

    # tractable dp/dx case
    eq2 = sp.Eq(x * sp.diff(u(x, y), x) + sp.diff(u(x, y), y), 1)
    examples.append(("Charpit dp/dx", solve_charpit_complete_integral_2vars(eq2, u(x, y), (x, y))))

    x, y, z = sp.symbols("x y z", real=True)
    w = sp.Function("w")
    eq3 = sp.Eq(sp.diff(w(x, y, z), x) ** 2, sp.diff(w(x, y, z), y) + sp.diff(w(x, y, z), z))
    examples.append(
        ("Jacobi autonomous", solve_jacobi_complete_integral(eq3, w(x, y, z), (x, y, z)))
    )

    for title, res in examples:
        print("=" * 80)
        print(title)
        print("method:", res.method)
        print("solutions:")
        for sol in res.solutions:
            print("  ", sol)
        print("details:", res.details)
        print("verification count:", len(res.verification))


if __name__ == "__main__":
    run_demo()
