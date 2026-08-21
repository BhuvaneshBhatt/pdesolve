import sympy as sp

from pdesolve.classical import pdesolve, pdesolve_constant_coefficient


def test_fits_wave_initial_data_and_verifies():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t, 2) - sp.diff(u(x, t), x, 2), 0)
    res = pdesolve_constant_coefficient(
        eq,
        u(x, t),
        (x, t),
        ics={"initial_profile": x**2, "initial_time_derivative": 0, "curve_value": 0},
    )
    expr = sp.expand(res.solution.rhs)
    assert sp.expand(sp.diff(expr, t, 2) - sp.diff(expr, x, 2)) == 0
    assert sp.expand(expr.subs(t, 0) - x**2) == 0
    assert sp.expand(sp.diff(expr, t).subs(t, 0)) == 0
    assert res.details["verification_summary"].verified is True


def test_single_family_transport_fit_and_verifies():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x), 0)
    res = pdesolve_constant_coefficient(
        eq,
        u(x, t),
        (x, t),
        ics={"initial_profile": x**3 + 2 * x, "curve_value": 0},
    )
    expr = sp.expand(res.solution.rhs)
    assert sp.expand(sp.diff(expr, t) - sp.diff(expr, x)) == 0
    assert sp.expand(expr.subs(t, 0) - (x**3 + 2 * x)) == 0
    assert res.details["verification_summary"].verified is True


def test_resonant_exponential_marks_consistency():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x), sp.exp(x + t))
    res = pdesolve(eq, u(x, t), (x, t), method="constant_coefficient_inverse_operator")
    expr = sp.expand(res.solution.rhs)
    assert sp.simplify(sp.diff(expr, t) - sp.diff(expr, x) - sp.exp(x + t)) == 0
    assert res.details["resonance_consistent"] is True
