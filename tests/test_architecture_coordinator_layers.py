import sympy as sp

from pdesolve.problem import build_pde_problem
from pdesolve.recognizers import recognize_canonical_problem
from pdesolve.planners import plan_canonical_problem
from pdesolve.solvers import execute_planned_solver
from pdesolve.results import BasePDEResult, SeriesPDEResult


def test_coordinator_layers_work_from_canonical_problem():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    problem = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics={"initial_profile": sp.sin(sp.pi * x)},
        bcs={"type": "dirichlet_homogeneous_interval", "length": 1},
    )
    recs = recognize_canonical_problem(problem)
    assert isinstance(recs, tuple)
    plan = plan_canonical_problem(problem, prefer_separation=True)
    methods = [step.method for step in plan.steps]
    assert "separation_framework" in methods or "heat_dirichlet_series" in methods
    result = execute_planned_solver(problem, methods[0])
    assert isinstance(result, BasePDEResult)


def test_execute_planned_solver_returns_standardized_series_result():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    problem = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics={"initial_profile": sp.sin(sp.pi * x)},
        bcs={"type": "dirichlet_homogeneous_interval", "length": 1},
    )
    result = execute_planned_solver(problem, "separation_framework")
    assert isinstance(result, SeriesPDEResult)
