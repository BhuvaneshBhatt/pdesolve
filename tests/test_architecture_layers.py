import sympy as sp

from pdesolve import (
    build_pde_problem,
    parse_conditions,
    infer_domain_geometry,
    extract_canonical_linear_system_form,
    solve_hyperbolic_system,
    pdesolve,
)


def test_condition_parser_handles_arbitrary_constant_time_and_boundary_slices():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    model = parse_conditions(
        ics=[
            sp.Eq(u(x, 2), sp.sin(x)),
            sp.Eq(sp.diff(u(x, t), t).subs(t, 2), sp.cos(x)),
        ],
        bcs=[sp.Eq(u(0, t), 0), sp.Eq(sp.diff(u(x, t), x).subs(x, 1), 0)],
        dep_expr=u(x, t),
        indep_vars=(x, t),
    )
    assert len(model.initial_conditions) == 2
    assert {c.location for c in model.initial_conditions} == {2}
    assert len(model.boundary_conditions) == 2


def test_domain_inference_detects_interval_from_boundary_conditions():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    model = parse_conditions(
        bcs=[sp.Eq(u(0, t), 0), sp.Eq(u(sp.pi, t), 0)],
        dep_expr=u(x, t),
        indep_vars=(x, t),
    )
    geom = infer_domain_geometry(indep_vars=(x, t), condition_model=model)
    assert geom.kind == "interval"
    assert geom.extents["x"] == (0, sp.pi)


def test_build_problem_canonical_layer_uses_domain_and_condition_metadata():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))
    prob = build_pde_problem(
        eq,
        u(x, t),
        (x, t),
        ics=[sp.Eq(u(x, 2), sp.sin(x))],
        bcs=[sp.Eq(u(0, t), 0), sp.Eq(u(sp.pi, t), 0)],
    )
    can = prob.canonical_representation
    assert can.time_slice_metadata["constant_time_values"] == (2,)
    assert can.domain_metadata["geometry"] == "interval"


def test_canonical_linear_system_extraction_and_solver_metadata():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    v = sp.Function("v")
    eqs = [
        sp.Eq(sp.diff(u(x, t), t), sp.diff(v(x, t), x)),
        sp.Eq(sp.diff(v(x, t), t), sp.diff(u(x, t), x)),
    ]
    can = extract_canonical_linear_system_form(eqs, (u(x, t), v(x, t)), (x, t))
    assert can.diagonalizable is True
    sol = solve_hyperbolic_system(
        eqs, [sp.Eq(u(x, 0), sp.sin(x)), sp.Eq(v(x, 0), 0)], (u(x, t), v(x, t)), (x, t)
    )
    assert sol.canonical_system is not None
    assert len(sol.characteristic_variables) == 2


def test_nonautonomous_quasilinear_path_attempts_explicit_characteristics():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq((1 + x) * sp.diff(u(x, t), x) + sp.diff(u(x, t), t), 0)
    res = pdesolve(
        eq,
        u(x, t),
        (x, t),
        method="quasilinear_implicit",
        ics={"initial_profile": x**2},
    )
    raw = res.details["raw_result"]
    assert raw.method == "quasilinear_implicit_characteristics"
    assert raw.details.get("explicit_characteristic_system") in {True, False}
