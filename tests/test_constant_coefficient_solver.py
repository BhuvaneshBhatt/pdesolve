import sympy as sp

from pdesolve.classical import (
    detect_linear_constant_coefficient_pde,
    particular_solution_polynomial_rhs_inverse_operator,
    solve_linear_constant_coefficient_pde,
    solve_simply_supported_beam_ibvp,
    pdesolve,
)


def test_detect_linear_constant_coefficient_pde_and_operator_poly():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    ccpde = detect_linear_constant_coefficient_pde(
        sp.Eq(
            sp.diff(u(x, t), t, 2)
            - 2 * sp.diff(u(x, t), x, t)
            - 3 * sp.diff(u(x, t), x, 2),
            0,
        ),
        u(x, t),
        (x, t),
    )
    m0, m1 = sp.symbols("m0:2")
    assert sp.expand(ccpde.operator_polynomial - (m1**2 - 2 * m0 * m1 - 3 * m0**2)) == 0


def test_polynomial_rhs_inverse_operator_particular_solution():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    ccpde = detect_linear_constant_coefficient_pde(
        sp.Eq(u(x, t) + sp.diff(u(x, t), x), x + t),
        u(x, t),
        (x, t),
    )
    part = particular_solution_polynomial_rhs_inverse_operator(ccpde)
    resid = sp.expand(part + sp.diff(part, x) - (x + t))
    assert resid == 0


def test_factorized_constant_coefficient_solution_and_data_fit():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    res = solve_linear_constant_coefficient_pde(
        sp.Eq(sp.diff(u(x, t), t, 2) - sp.diff(u(x, t), x, 2), 0),
        u(x, t),
        (x, t),
        ics={"initial_profile": x**2, "initial_time_derivative": 0, "curve_value": 0},
    )
    # fitted d'Alembert-like solution should satisfy the PDE and initial data.
    expr = res.solution.rhs
    assert sp.expand(sp.diff(expr, t, 2) - sp.diff(expr, x, 2)) == 0
    assert sp.expand(expr.subs(t, 0) - x**2) == 0
    assert sp.expand(sp.diff(expr, t).subs(t, 0)) == 0


def test_simply_supported_beam_spectral_solver_returns_series():
    x, t, L = sp.symbols("x t L", positive=True, real=True)
    u = sp.Function("u")
    res = solve_simply_supported_beam_ibvp(
        u(x, t),
        x=x,
        t=t,
        length=L,
        stiffness=1,
        damping=0,
        initial_displacement=sp.sin(sp.pi * x / L),
        initial_velocity=0,
    )
    assert res.method == "simply_supported_beam_spectral"
    assert isinstance(res.solution.rhs, sp.Sum)


def test_auto_dispatch_prefers_constant_coefficient_linear_solver():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    res = pdesolve(
        sp.Eq(sp.diff(u(x, t), t, 2) - sp.diff(u(x, t), x, 2), 0),
        u(x, t),
        (x, t),
        ics={"initial_profile": x**2, "initial_time_derivative": 0, "curve_value": 0},
    )
    assert res.method in {
        "linear_constant_coefficient_factored_fitted_data",
        "linear_constant_coefficient_factored",
    }
