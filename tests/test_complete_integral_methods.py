import sympy as sp

import pdesolve as rle


def test_charpit_additive_separation_returns_verification():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) ** 2 + sp.diff(u(x, y), y) ** 2, 1)
    res = rle.solve_charpit_complete_integral_2vars(eq, u(x, y), (x, y))
    assert res.solutions
    assert res.verification
    assert all(v["verified"] for v in res.verification)


def test_charpit_separated_form():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) ** 2 + sp.diff(u(x, y), y), 0)
    res = rle.solve_complete_integral_pde(eq, u(x, y), (x, y))
    assert res.solutions
    assert all(sol.lhs == u(x, y) for sol in res.solutions)


def test_jacobi_additive_or_autonomous_result_has_verification():
    x, y, z = sp.symbols("x y z", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y, z), x) ** 2, sp.diff(u(x, y, z), y) + sp.diff(u(x, y, z), z))
    res = rle.solve_jacobi_complete_integral(eq, u(x, y, z), (x, y, z))
    assert res.solutions
    assert res.verification
    assert any(v["verified"] for v in res.verification)
