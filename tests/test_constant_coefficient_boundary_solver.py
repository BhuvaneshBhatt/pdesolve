import sympy as sp

from pdesolve.classical import pdesolve_constant_coefficient


def test_point_condition_fit_for_1d_constant_coefficient_family():
    x = sp.symbols("x", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x), x, 2) - u(x), 0)
    res = pdesolve_constant_coefficient(
        eq,
        u(x),
        (x,),
        bcs={"point_value": 0, "point_conditions": {0: 1, 1: 0}},
    )
    expr = sp.expand(res.solution.rhs)
    assert sp.expand(sp.diff(expr, x, 2) - expr) == 0
    assert sp.expand(expr.subs(x, 0) - 1) == 0
    assert sp.expand(sp.diff(expr, x).subs(x, 0)) == 0


def test_generic_line_conditions_interface_fits_transport_profile():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x), 0)
    res = pdesolve_constant_coefficient(
        eq,
        u(x, t),
        (x, t),
        ics={"line_var": t, "line_value": 0, "line_conditions": {0: x**2 + 3 * x}},
    )
    expr = sp.expand(res.solution.rhs)
    assert sp.expand(sp.diff(expr, t) - sp.diff(expr, x)) == 0
    assert sp.expand(expr.subs(t, 0) - (x**2 + 3 * x)) == 0
