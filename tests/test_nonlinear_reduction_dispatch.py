import sympy as sp

from pdesolve.classical import pdesolve


def test_auto_method_available_for_first_order_nonlinear():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)
    res = pdesolve(
        eq, u(x, t), (x, t), method="first_order_nonlinear_auto", ics={"initial_profile": x**2}
    )
    assert res.method in {
        "scalar_conservation_implicit_characteristics",
        "first_order_nonlinear_auto",
        "invariant_reduction_auto",
    }
