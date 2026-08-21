import sympy as sp

from pdesolve.classical import (
    plan_pde_solution_methods,
    solve_heat_equation_1d_laplace_fourier_formal,
    solve_rectangle_dirichlet_laplace_series,
    solve_heat_equation_1d_robin_series,
    solve_reduced_equation_auto,
    build_quasilinear_characteristic_odes,
)


def test_plan_prefers_wave_dalembert_for_wave_ivp():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")
    plan = plan_pde_solution_methods(
        sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)),
        u(x, t),
        ics={"initial_displacement": lambda z: z, "initial_velocity": lambda z: 1},
    )
    methods = [s.method for s in plan.steps]
    assert "wave_dalembert" in methods
    assert methods.index("wave_dalembert") < methods.index("classification_only")


def test_laplace_fourier_heat_formal_builds_transform():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")
    tr = solve_heat_equation_1d_laplace_fourier_formal(
        u(x, t), x=x, t=t, initial_profile=x * (1 - x)
    )
    assert tr.method == "laplace_fourier_heat_formal"
    assert tr.transformed_equation is not None


def test_rectangle_laplace_series_solver_runs():
    x, y = sp.symbols("x y", positive=True, real=True)
    u = sp.Function("u")
    res = solve_rectangle_dirichlet_laplace_series(
        u(x, y), x=x, y=y, boundary_top=x * (sp.pi - x), terms=3
    )
    assert res.method == "laplace_rectangle_dirichlet_series"
    assert isinstance(res.solution, sp.Equality)


def test_heat_robin_series_solver_runs():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")
    res = solve_heat_equation_1d_robin_series(
        u(x, t), x=x, t=t, initial_profile=x * (sp.pi - x), terms=3
    )
    assert res.method == "heat_robin_series_formal"
    assert isinstance(res.solution, sp.Equality)


def test_reduced_equation_auto_solves_simple_ode():
    z = sp.symbols("z")
    f = sp.Function("f")
    reduced = sp.Eq(sp.diff(f(z), z), f(z))
    res = solve_reduced_equation_auto(reduced)
    assert res.solved
    assert "exp" in str(res.solution)


def test_build_quasilinear_characteristic_odes_returns_three_odes():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    odes = build_quasilinear_characteristic_odes(
        sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0), u(x, t)
    )
    assert len(odes) == 3
