import sympy as sp

from pdesolve import (
    pdesolve,
    solve_fundamental_solution,
    solve_green_function,
    FundamentalSolutionResult,
    GreenFunctionResult,
)


def test_heat_fundamental_solution_whole_line():
    x, t, a = sp.symbols("x t a", positive=True)
    u = sp.Function("u")
    res = solve_fundamental_solution(
        sp.Eq(sp.diff(u(x, t), t) - a * sp.diff(u(x, t), x, 2), 0), u(x, t), (x, t)
    )
    assert isinstance(res, FundamentalSolutionResult)
    assert res.method == "kernel_fundamental_solution"
    assert "exp" in str(res.solution)
    assert "Heaviside" in str(res.solution)


def test_heat_green_function_half_line_dirichlet():
    x, t, a = sp.symbols("x t a", positive=True)
    u = sp.Function("u")
    res = solve_green_function(
        sp.Eq(sp.diff(u(x, t), t) - a * sp.diff(u(x, t), x, 2), 0),
        u(x, t),
        (x, t),
        bcs=(sp.Eq(u(0, t), 0),),
        geometry="half_line",
    )
    assert isinstance(res, GreenFunctionResult)
    s = str(res.solution)
    assert "exp" in s and "-x_" in s or "+ x_0" in s


def test_wave_green_function_interval_dirichlet_is_series():
    x, t, c, L = sp.symbols("x t c L", positive=True)
    u = sp.Function("u")
    res = solve_green_function(
        sp.Eq(sp.diff(u(x, t), t, 2) - c**2 * sp.diff(u(x, t), x, 2), 0),
        u(x, t),
        (x, t),
        bcs=(sp.Eq(u(0, t), 0), sp.Eq(u(L, t), 0)),
    )
    assert isinstance(res, GreenFunctionResult)
    assert isinstance(res.solution, sp.Expr)
    assert "Sum" in str(res.solution)
    assert "sin" in str(res.solution)


def test_laplace_green_function_half_plane_dirichlet():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    res = solve_green_function(
        sp.Eq(sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2), 0),
        u(x, y),
        (x, y),
        bcs={"type": "dirichlet_half_plane"},
        geometry="half_plane",
    )
    assert isinstance(res, GreenFunctionResult)
    assert "log" in str(res.solution).lower()


def test_pdesolve_auto_routes_to_kernel_green_function():
    x, t, a, xi, tau, L = sp.symbols("x t a xi tau L", positive=True)
    u = sp.Function("u")
    eq = sp.Eq(
        sp.diff(u(x, t), t) - a * sp.diff(u(x, t), x, 2),
        sp.DiracDelta(x - xi) * sp.DiracDelta(t - tau),
    )
    rec = pdesolve(eq, u(x, t), (x, t), bcs=(sp.Eq(u(0, t), 0), sp.Eq(u(L, t), 0)))
    assert rec.method == "kernel_green_function"
    assert "Sum" in str(rec.solution)
