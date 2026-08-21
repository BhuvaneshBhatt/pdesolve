import sympy as sp

from pdesolve import build_pde_problem, pdesolve, plan_canonical_problem
from pdesolve.classical import solve_heat_equation_1d_half_line_transform
from pdesolve.conditions import (
    DirichletCondition,
    NeumannCondition,
    PeriodicCondition,
    RobinCondition,
    classify_condition_equation,
    parse_conditions,
)
from pdesolve.domains import HalfLineDomain, IntervalDomain, RectangleDomain, infer_domain_geometry
from pdesolve.normalization import NormalizationPolicy, normalize_solution
from pdesolve.solver_execution import is_registered_method


def test_canonical_condition_parser_distinguishes_dirichlet_neumann_robin_and_periodic():
    x, t = sp.symbols("x t", real=True)
    alpha, beta = sp.symbols("alpha beta", nonzero=True)
    u = sp.Function("u")
    model = parse_conditions(
        bcs=[
            sp.Eq(u(0, t), 0),
            sp.Eq(sp.diff(u(x, t), x).subs(x, 1), 0),
            sp.Eq(alpha * u(2, t) + beta * sp.diff(u(x, t), x).subs(x, 2), 0),
            sp.Eq(u(3, t), u(4, t)),
        ],
        dep_expr=u(x, t),
        indep_vars=(x, t),
    )
    assert isinstance(model.boundary_conditions[0], DirichletCondition)
    assert isinstance(model.boundary_conditions[1], NeumannCondition)
    assert isinstance(model.boundary_conditions[2], RobinCondition)
    assert isinstance(model.boundary_conditions[3], PeriodicCondition)
    kinds = [
        classify_condition_equation(c, time_variable=t, spatial_variables=(x,))
        for c in model.boundary_conditions
    ]
    assert kinds == ["dirichlet", "neumann", "robin", "periodic"]
    assert model.boundary_conditions[2].metadata["alpha"] == alpha
    assert model.boundary_conditions[2].metadata["beta"] == beta
    assert model.boundary_conditions[3].metadata["paired_location"] == 4


def test_domain_inference_accepts_explicit_sympy_domains():
    x, y = sp.symbols("x y", real=True)
    half = infer_domain_geometry(indep_vars=(x,), domain=sp.Interval(2, sp.oo))
    interval = infer_domain_geometry(indep_vars=(x,), domain=sp.Interval(-1, 3))
    rect = infer_domain_geometry(
        indep_vars=(x, y), domain=sp.ProductSet(sp.Interval(0, 1), sp.Interval(-2, 2))
    )
    assert isinstance(half, HalfLineDomain) and half.extents["x"] == (2, sp.oo)
    assert isinstance(interval, IntervalDomain) and interval.extents["x"] == (-1, 3)
    assert isinstance(rect, RectangleDomain)
    assert rect.extents == {"x": (0, 1), "y": (-2, 2)}


def test_single_boundary_plus_initial_slice_infers_half_line_and_routes_heat_transform():
    x, t = sp.symbols("x t", nonnegative=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))
    ics = [sp.Eq(u(x, 0), 0)]
    bcs = [sp.Eq(u(0, t), 0)]
    problem = build_pde_problem(eq, u(x, t), (x, t), ics=ics, bcs=bcs)
    assert problem.details["domain_geometry"].kind == "half_line"
    plan = plan_canonical_problem(problem)
    assert plan.steps[0].method == "heat_half_line_transform"
    assert is_registered_method(plan.steps[0].method)
    result = pdesolve(eq, u(x, t), (x, t), ics=ics, bcs=bcs)
    assert result.method == "heat_half_line_dirichlet_transform"
    assert sp.simplify(result.solution.rhs) == 0


def test_interval_wave_ivp_routes_to_dirichlet_series_not_dalembert():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2))
    ics = [sp.Eq(u(x, 0), sp.sin(x)), sp.Eq(sp.diff(u(x, t), t).subs(t, 0), 0)]
    bcs = [sp.Eq(u(0, t), 0), sp.Eq(u(sp.pi, t), 0)]
    problem = build_pde_problem(eq, u(x, t), (x, t), ics=ics, bcs=bcs)
    plan = plan_canonical_problem(problem)
    assert plan.steps[0].method == "wave_dirichlet_series"
    result = pdesolve(eq, u(x, t), (x, t), ics=ics, bcs=bcs, terms=2)
    assert result.method == "wave_dirichlet_sine_series"
    assert sp.simplify(result.solution.rhs - sp.sin(x) * sp.cos(t)) == 0


def test_explicit_domain_participates_in_problem_geometry_and_planning():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))
    problem = build_pde_problem(
        eq, u(x, t), (x, t), ics=[sp.Eq(u(x, 0), 0)], domain=sp.Interval(0, sp.oo)
    )
    assert problem.domain == sp.Interval(0, sp.oo)
    assert problem.details["domain_geometry"].kind == "half_line"
    assert problem.canonical_representation.domain_metadata["explicit"] is True


def test_bounded_normalization_reduces_simple_closed_form_but_skips_formal_integrals():
    x = sp.symbols("x")
    raw = sp.Eq(sp.Function("u")(x), (x**2 - 1) / (x - 1), evaluate=False)
    normalized, report = normalize_solution(
        raw, method="first_order", policy=NormalizationPolicy(max_ops=50)
    )
    assert report.attempted is True
    assert report.changed is True
    assert sp.simplify(normalized.rhs - (x + 1)) == 0

    formal = sp.Eq(sp.Function("u")(x), sp.Integral(sp.exp(-(x**2)), x), evaluate=False)
    untouched, formal_report = normalize_solution(formal, method="first_order")
    assert untouched == formal
    assert formal_report.attempted is False
    assert formal_report.skipped_reason == "formal_or_piecewise_expression"


def test_dispatch_records_normalization_metadata_and_supports_opt_out():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y), x) + sp.diff(u(x, y), y), 0)
    result = pdesolve(eq, u(x, y), (x, y), method="first_order", normalize_result=False)
    assert result.details["normalization"]["attempted"] is False
    assert result.details["normalization"]["skipped_reason"] == "disabled"


def test_half_line_heat_transform_construction_is_bounded_and_unevaluated():
    import time

    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")(x, t)

    start = time.perf_counter()
    result = solve_heat_equation_1d_half_line_transform(
        u,
        x=x,
        t=t,
        initial_profile=sp.exp(-x),
        boundary="dirichlet",
    )
    elapsed = time.perf_counter() - start

    # This regression used to enter an expensive sp.simplify() call over nested
    # improper integrals.  Construction should remain a cheap formal operation.
    assert elapsed < 2.0
    assert result.method == "heat_half_line_dirichlet_transform"
    assert result.details["transform_evaluated"] is False
    assert isinstance(result.details["profile_transform"], sp.Integral)
    assert result.solution.rhs.has(sp.Integral)
