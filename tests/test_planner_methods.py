import sympy as sp

from pdesolve.classification import rank_pde_solution_methods


def _methods(rank_out):
    return [cand.method for cand in rank_out[1]]


def test_planner_exposes_charpit_and_complete_integral():
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) * sp.diff(u(x, y), y), 1)
    methods = _methods(rank_pde_solution_methods(eq, u, (x, y)))
    assert "charpit" in methods
    assert "complete_integral" in methods
    assert "invariant_reduction_auto" in methods


def test_explicit_charpit_method_runs():
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) * sp.diff(u(x, y), y), 1)
    res = __import__("pdesolve").pdesolve(eq, u, (x, y), method="charpit")
    assert getattr(res, "method", "").startswith("charpit")
