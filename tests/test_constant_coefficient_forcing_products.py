import sympy as sp

from pdesolve.classical import pdesolve, pdesolve_constant_coefficient


def test_polynomial_times_trig_forcing():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t), x * sp.cos(t))
    res = pdesolve(eq, u(x, t), (x, t), method="constant_coefficient_inverse_operator")
    expr = sp.expand(res.solution.rhs)
    residual = sp.simplify(sp.diff(expr, t) + expr - x * sp.cos(t))
    assert residual == 0
    assert res.details["particular_result"].method in {
        "constant_coefficient_trig_hyperbolic",
        "constant_coefficient_polynomial_exponential",
        "constant_coefficient_sum_split",
    }
    assert res.details["verification_summary"].verified is True


def test_polynomial_times_resonant_exponential_forcing():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x), x * sp.exp(x + t))
    res = pdesolve_constant_coefficient(eq, u(x, t), (x, t))
    expr = sp.expand(res.solution.rhs)
    residual = sp.simplify(sp.diff(expr, t) - sp.diff(expr, x) - x * sp.exp(x + t))
    assert residual == 0
    assert res.details["resonance_consistent"] is True
    part = res.details["particular_result"]
    assert part.details.get("resonant") is True
