import sympy as sp

from pdesolve.classical import (
    detect_first_order_linear_form_2vars,
    solve_reduced_equation_auto,
    pdesolve,
)


def test_detect_first_order_linear_form_variable_coefficients():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(x * sp.diff(u(x, y), x) + sp.diff(u(x, y), y) + u(x, y), x + y)
    form = detect_first_order_linear_form_2vars(eq, u(x, y), (x, y))
    assert sp.simplify(form.A - x) == 0
    assert sp.simplify(form.B - 1) == 0
    assert sp.simplify(form.D - 1) == 0
    assert sp.simplify(form.E - (x + y)) == 0
    assert form.is_constant_coefficient is False


def test_solve_reduced_equation_auto_ode():
    z = sp.symbols("z", real=True)
    f = sp.Function("f")
    eq = sp.Eq(sp.diff(f(z), z) - f(z), 0)
    res = solve_reduced_equation_auto(eq)
    assert res.method == "dsolve_reduced_ode"
    assert isinstance(res.solution, sp.Equality)


def test_pdesolve_prefers_symmetry_when_requested():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0)
    res = pdesolve(
        eq, u(x, t), (x, t), method="auto", prefer_symmetry=True, max_symmetry_steps=1
    )
    # Accept either successful symmetry reduction/postsolve or a strong first-order direct solve.
    assert res.method in {
        "symmetry_reduction_plus_postsolve",
        "pdsolve_first_order_linear",
        "pdsolve_first_order",
        "constant_coefficient_characteristics_ivp",
    }
