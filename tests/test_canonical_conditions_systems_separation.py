import sympy as sp

from pdesolve.problem import build_pde_problem
from pdesolve.classification import plan_pde_solution_methods
from pdesolve.separation_framework import build_separable_geometry_plan
from pdesolve.hyperbolic_system import (
    extract_canonical_linear_system_form,
    solve_hyperbolic_system,
)


def test_canonical_problem_carries_condition_analysis():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))
    problem = build_pde_problem(
        eq,
        u(x, t),
        (x, t),
        ics=sp.Eq(u(x, 0), sp.sin(x)),
        bcs=[sp.Eq(u(0, t), 0), sp.Eq(u(sp.pi, t), 0)],
    )
    report = problem.details.get("condition_analysis")
    assert report is not None
    assert report.ok is True


def test_planner_uses_separation_framework_metadata():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))
    plan = plan_pde_solution_methods(
        eq,
        u(x, t),
        (x, t),
        ics=sp.Eq(u(x, 0), sp.sin(x)),
        bcs=[sp.Eq(u(0, t), 0), sp.Eq(u(sp.pi, t), 0)],
        prefer_separation=True,
    )
    assert plan.details["separation_plan"] is not None
    assert plan.details["separation_plan"].eigenbasis == "sine"
    assert any(step.method == "separation_framework" for step in plan.steps)


def test_compatibility_engine_flags_incomplete_rectangle_boundary():
    x, y, t = sp.symbols("x y t", real=True)
    u = sp.Function("u")
    problem = build_pde_problem(
        sp.Eq(
            sp.diff(u(x, y, t), t),
            sp.diff(u(x, y, t), x, 2) + sp.diff(u(x, y, t), y, 2),
        ),
        u(x, y, t),
        (x, y, t),
        ics=sp.Eq(u(x, y, 0), sp.sin(sp.pi * x) * sp.sin(sp.pi * y)),
        bcs=[sp.Eq(u(0, y, t), 0), sp.Eq(u(1, y, t), 0)],
    )
    report = problem.details["condition_analysis"]
    assert any(issue.code == "rectangle_boundary_incomplete" for issue in report.issues)


def test_system_extraction_and_solution_metadata():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    v = sp.Function("v")
    eqs = [
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x) + 0 * sp.diff(v(x, t), x)),
        sp.Eq(sp.diff(v(x, t), t), -sp.diff(v(x, t), x)),
    ]
    canonical = extract_canonical_linear_system_form(eqs, (u, v), (x, t))
    assert canonical.diagonalizable is True
    res = solve_hyperbolic_system(
        eqs,
        [sp.Eq(u(x, 0), sp.sin(x)), sp.Eq(v(x, 0), sp.cos(x))],
        (u, v),
        vars=(x, t),
    )
    assert res.system_size == 2
    assert res.details["solver"] in {"diagonalization", "matrix_exponential"}


def test_separation_plan_for_rectangle_dirichlet():
    x, y, t = sp.symbols("x y t", real=True)
    u = sp.Function("u")
    problem = build_pde_problem(
        sp.Eq(
            sp.diff(u(x, y, t), t),
            sp.diff(u(x, y, t), x, 2) + sp.diff(u(x, y, t), y, 2),
        ),
        u(x, y, t),
        (x, y, t),
        ics=sp.Eq(u(x, y, 0), x * (1 - x) * y * (1 - y)),
        bcs=[
            sp.Eq(u(0, y, t), 0),
            sp.Eq(u(1, y, t), 0),
            sp.Eq(u(x, 0, t), 0),
            sp.Eq(u(x, 1, t), 0),
        ],
    )
    geom = problem.details["domain_geometry"]
    cmodel = problem.details["condition_model"]
    sep = build_separable_geometry_plan(geom, cmodel, family="heat_like")
    assert sep is not None
    assert sep.eigenbasis == "tensor_sine"
