import sympy as sp

from pdesolve import ClosedFormPDEResult, SeriesPDEResult, SystemPDEResult, TransformPDEResult
from pdesolve.problem import build_pde_problem, build_system_pde_problem
from pdesolve.solver_execution import solve_with_canonical_problem


def test_canonical_problem_first_order_solver_family_returns_standardized_result():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    problem = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0),
        u(x, t),
        (x, t),
        ics={"initial_profile": sp.exp(-(x**2))},
    )
    res = solve_with_canonical_problem(problem, "transport_ivp")
    assert isinstance(res, ClosedFormPDEResult)
    assert res.metadata.get("canonical_representation") is not None


def test_canonical_problem_series_solver_family_returns_series_result():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    problem = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics={"initial_profile": x * (sp.pi - x)},
        bcs={"type": "dirichlet_homogeneous_interval", "length": sp.pi},
    )
    res = solve_with_canonical_problem(problem, "heat_dirichlet_series")
    assert isinstance(res, SeriesPDEResult)


def test_canonical_problem_transform_solver_family_returns_transform_result():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    problem = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics={"initial_profile": sp.exp(-(x**2))},
    )
    res = solve_with_canonical_problem(problem, "fourier_heat")
    assert isinstance(res, TransformPDEResult)


def test_canonical_problem_system_solver_family_returns_system_result():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    v = sp.Function("v")
    eqs = [
        sp.Eq(sp.diff(u(x, t), t), sp.diff(v(x, t), x)),
        sp.Eq(sp.diff(v(x, t), t), sp.diff(u(x, t), x)),
    ]
    sys_problem = build_system_pde_problem(
        eqs, (u(x, t), v(x, t)), (x, t), ics=[sp.Eq(u(x, 0), sp.sin(x)), sp.Eq(v(x, 0), 0)]
    )
    res = solve_with_canonical_problem(sys_problem, "hyperbolic_system")
    assert isinstance(res, SystemPDEResult)
