import sympy as sp
from pdesolve.classical import (
    process_initial_curve_2d,
    fit_complete_integral_to_initial_curve,
    solve_first_order_cauchy_problem_2d,
)


def test_process_initial_curve_parabolic_graph():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    ic = sp.Eq(u(x, x**2), x**3)
    curve = process_initial_curve_2d(ic, u(x, y), (x, y))
    assert curve.parameter == x
    assert sp.simplify(curve.x_curve - x) == 0
    assert sp.simplify(curve.y_curve - x**2) == 0
    assert sp.simplify(curve.u_data - x**3) == 0


def test_fit_general_solution_to_initial_curve_single_free_function():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    F = sp.Function("F")
    fam = sp.Eq(u(x, y), F(x + y))
    ic = sp.Eq(u(x, 0), x**2)
    fitted = fit_complete_integral_to_initial_curve(fam, ic, u(x, y), (x, y))
    assert fitted is not None
    assert sp.simplify(fitted.rhs - (x + y) ** 2) == 0


def test_solve_first_order_cauchy_problem_transport():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) + sp.diff(u(x, y), y), 0)
    ic = sp.Eq(u(x, 0), x**2)
    res = solve_first_order_cauchy_problem_2d(eq, ic, u(x, y), (x, y))
    assert sp.simplify(res.solution.rhs - (x - y) ** 2) == 0
    assert res.verification["pde_verified"]
    assert res.verification["initial_verified"]
