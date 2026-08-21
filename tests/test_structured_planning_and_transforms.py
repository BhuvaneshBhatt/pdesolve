from __future__ import annotations

import sympy as sp

from pdesolve import build_pde_problem, pdesolve, canonicalize_first_order_nonlinear_pde
from pdesolve.classification import plan_pde_solution_methods
from pdesolve.benchmark_suite import get_method_family_regression_cases


def test_separation_framework_solver_consumes_plan_for_interval_heat():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eqs = [
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        sp.Eq(u(x, 0), sp.sin(x)),
        sp.Eq(u(0, t), 0),
        sp.Eq(u(sp.pi, t), 0),
    ]
    res = pdesolve(eqs, u(x, t), (x, t), method="separation_framework")
    assert getattr(res, "method", None) in {
        "heat_dirichlet_series",
        "series_solution",
        "closed_form",
        "heat_dirichlet_series",
    } or "heat" in str(getattr(res, "method", ""))


def test_structured_transform_plan_is_built_for_whole_line_heat():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    prob = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics={"equation": sp.Eq(u(x, 0), sp.exp(-(x**2)))},
    )
    tp = prob.details.get("transform_plan")
    assert tp is not None
    assert tp.method == "structured_transform"


def test_first_order_framework_canonicalization_generalized_clairaut():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(
        u(x, y),
        x * sp.diff(u(x, y), x)
        + y * sp.diff(u(x, y), y)
        + sp.sin(sp.diff(u(x, y), x) + sp.diff(u(x, y), y)),
    )
    can = canonicalize_first_order_nonlinear_pde(eq, u(x, y), (x, y))
    assert can.recognized_family in {"generalized_clairaut", "generic_first_order"}


def test_boundary_model_is_populated():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    prob = build_pde_problem(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics={"equation": sp.Eq(u(x, 0), x)},
        bcs=[sp.Eq(u(0, t), 0), sp.Eq(u(1, t), 0)],
    )
    bm = prob.details.get("boundary_model")
    assert bm is not None
    assert len(bm.bindings) >= 2


def test_method_family_regression_suite_exposes_cases():
    suite = get_method_family_regression_cases()
    assert "heat_like" in suite and suite["heat_like"]


def test_planner_adds_structured_transform_candidate():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    plan = plan_pde_solution_methods(
        sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
        ics={"equation": sp.Eq(u(x, 0), sp.exp(-(x**2)))},
        prefer_transform=True,
    )
    methods = {c.method for c in plan.steps}
    assert "structured_transform" in methods
