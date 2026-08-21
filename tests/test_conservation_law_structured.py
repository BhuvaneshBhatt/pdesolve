import sympy as sp

from pdesolve.classical import (
    canonicalize_scalar_conservation_law_1d,
    parse_scalar_conservation_law_initial_data,
    solve_scalar_conservation_law_ivp,
    verify_piecewise_conservation_law_solution,
    verify_weak_conservation_law_solution,
)


def test_canonical_conservation_law_normalization_burgers():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)
    can = canonicalize_scalar_conservation_law_1d(eq, u, (x, t))
    assert can.family in {"inviscid_burgers", "scalar_conservation_law"}
    assert sp.simplify(can.flux - u(x, t) ** 2 / 2) == 0
    assert sp.simplify(can.source) == 0
    assert sp.simplify(can.autonomous_flux - sp.Symbol("u", real=True) ** 2 / 2) == 0


def test_parse_conservation_law_initial_data_profile_and_riemann():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    general = parse_scalar_conservation_law_initial_data(
        {"initial_profile": sp.sin(x)}, u, (x, t)
    )
    assert general.kind == "general_profile"
    riemann = parse_scalar_conservation_law_initial_data(
        {"riemann_data": (1, 0)}, u, (x, t)
    )
    assert riemann.kind == "riemann"
    assert riemann.left_state == 1
    assert riemann.right_state == 0


def test_structured_propagation_result_for_linear_flux():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + 3 * sp.diff(u(x, t), x), 0)
    res = solve_scalar_conservation_law_ivp(
        eq, u, (x, t), initial_conditions={"initial_profile": sp.sin(x)}
    )
    assert res.method == "scalar_conservation_profile_propagation"
    assert isinstance(res.solution, sp.Equality)
    assert sp.simplify(res.speed - 3) == 0


def test_structured_riemann_shock_and_weak_verification():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)
    res = solve_scalar_conservation_law_ivp(
        eq, u, (x, t), initial_conditions={"riemann_data": (1, 0)}
    )
    assert res.method == "scalar_conservation_riemann_shock"
    weak = verify_weak_conservation_law_solution(res)
    assert weak.verified is True
    piecewise = verify_piecewise_conservation_law_solution(
        eq, res, u, (x, t), initial_conditions={"riemann_data": (1, 0)}
    )
    assert piecewise.verified is True


def test_structured_riemann_rarefaction_and_weak_verification():
    x, t = sp.symbols("x t", positive=True, real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)
    res = solve_scalar_conservation_law_ivp(
        eq, u, (x, t), initial_conditions={"riemann_data": (0, 1)}
    )
    assert res.method == "scalar_conservation_riemann_rarefaction"
    weak = verify_weak_conservation_law_solution(res)
    assert weak.verified is True


def test_structured_implicit_characteristics_for_general_autonomous_flux():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + 3 * u(x, t) ** 2 * sp.diff(u(x, t), x), 0)
    res = solve_scalar_conservation_law_ivp(
        eq, u, (x, t), initial_conditions={"initial_profile": sp.sin(x)}
    )
    assert res.method == "scalar_conservation_implicit_characteristics"
    assert isinstance(res.solution, tuple)
    assert len(res.solution) == 2
    xi = res.characteristic_parameter
    assert sp.simplify(res.characteristic_relation.lhs - x) == 0
    assert sp.simplify(res.profile_relation.rhs - sp.sin(xi)) == 0
    weak = verify_weak_conservation_law_solution(res)
    assert weak.verified is True
