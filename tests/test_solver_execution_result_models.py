import sympy as sp

from pdesolve.first_order_nonlinear import solve_first_order_nonlinear_auto
from pdesolve.problem import build_pde_problem, build_system_pde_problem
from pdesolve.results import BasePDEResult, SeriesPDEResult, SystemPDEResult, TransformPDEResult
from pdesolve.solver_execution import solve_with_canonical_problem


def test_separation_solver_returns_standardized_series_result():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))
    problem = build_pde_problem(
        eq,
        u(x, t),
        (x, t),
        ics={"initial_profile": x * (sp.pi - x)},
        bcs={"type": "dirichlet_homogeneous_interval", "length": sp.pi},
    )
    res = solve_with_canonical_problem(problem, "heat_dirichlet_series")
    assert isinstance(res, BasePDEResult)
    assert isinstance(res, SeriesPDEResult)


def test_transform_solver_returns_standardized_transform_result():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))
    problem = build_pde_problem(eq, u(x, t), (x, t), ics={"initial_profile": sp.exp(-(x**2))})
    res = solve_with_canonical_problem(problem, "structured_transform")
    assert isinstance(res, TransformPDEResult)


def test_system_solver_returns_standardized_system_result():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    v = sp.Function("v")
    eqs = (
        sp.Eq(sp.diff(u(x, t), t), sp.diff(v(x, t), x)),
        sp.Eq(sp.diff(v(x, t), t), sp.diff(u(x, t), x)),
    )
    problem = build_system_pde_problem(eqs, (u(x, t), v(x, t)), (x, t), ics=[])
    res = solve_with_canonical_problem(problem, "hyperbolic_system")
    assert isinstance(res, SystemPDEResult)


def test_first_order_nonlinear_auto_routes_through_canonical_solver_execution():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) + u(x, y) * sp.diff(u(x, y), y), 0)
    res = solve_first_order_nonlinear_auto(
        eq, u(x, y), (x, y), ics={"initial_profile": 1 / (x + 1), "curve_value": 0}
    )
    assert isinstance(res, BasePDEResult)
