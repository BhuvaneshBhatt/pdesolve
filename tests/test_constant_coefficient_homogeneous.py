import sympy as sp

from pdesolve.classical import build_constant_coefficient_homogeneous_solution, pdesolve


def test_repeated_linear_factor_homogeneous_family_residual_vanishes():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t, 2) - 2 * sp.diff(u(x, t), x, t) + sp.diff(u(x, t), x, 2), 0)
    res = build_constant_coefficient_homogeneous_solution(eq, u(x, t), (x, t))
    expr = res.solution.rhs
    residual = sp.expand(sp.diff(expr, t, 2) - 2 * sp.diff(expr, x, t) + sp.diff(expr, x, 2))
    assert residual == 0
    assert len(res.details["families"]) == 1
    fam = res.details["families"][0]
    assert fam.factor.multiplicity == 2
    assert fam.invariant is not None
    assert fam.transverse is not None


def test_auto_dispatch_uses_constant_coefficient_homogeneous_family_without_data():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x) + 2 * u(x, t), 0)
    res = pdesolve(eq, u(x, t), (x, t))
    expr = res.solution.rhs
    residual = sp.expand(sp.diff(expr, t) - sp.diff(expr, x) + 2 * expr)
    assert residual == 0
    assert res.method in {
        "constant_coefficient_homogeneous_family",
        "constant_coefficient_homogeneous_plus_particular",
    }
