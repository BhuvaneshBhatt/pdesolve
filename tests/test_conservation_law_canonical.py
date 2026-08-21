import sympy as sp

from pdesolve import (
    ConservationLawRarefactionResult,
    ConservationLawShockResult,
    analyze_conservation_law,
    canonicalize_scalar_conservation_law,
    parse_conservation_law_initial_data,
    verify_conservation_law_solution,
)


def test_canonicalize_linear_advection_conservation_law():
    x, t, c = sp.symbols("x t c", real=True)
    u = sp.Function("u")(x, t)
    can = canonicalize_scalar_conservation_law(
        sp.Eq(sp.diff(u, t) + c * sp.diff(u, x), 0), u, (x, t)
    )
    usym = sp.Symbol("u", real=True)
    assert can.autonomous_flux == c * usym
    assert sp.simplify(can.characteristic_speed - c) == 0


def test_parse_piecewise_riemann_initial_data():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")(x, t)
    ic = sp.Eq(u.subs(t, 0), sp.Piecewise((1, x < 0), (0, True)))
    parsed = parse_conservation_law_initial_data(ic, u, (x, t))
    assert parsed.kind == "riemann"
    assert parsed.left_state == 1
    assert parsed.right_state == 0


def test_analyze_conservation_law_returns_structured_riemann_result():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")(x, t)
    eq = sp.Eq(sp.diff(u, t) + u * sp.diff(u, x), 0)
    ics = {"initial_profile": sp.Piecewise((1, x < 0), (0, True))}
    res = analyze_conservation_law(eq, u, (x, t), ics=ics)
    assert res.method == "scalar_conservation_riemann_shock"
    assert isinstance(res.details["structured_result"], ConservationLawShockResult)


def test_analyze_conservation_law_returns_structured_propagation_result():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")(x, t)
    eq = sp.Eq(sp.diff(u, t) + 2 * sp.diff(u, x), 0)
    ics = {"initial_profile": sp.sin(x)}
    res = analyze_conservation_law(eq, u, (x, t), ics=ics)
    assert res.method == "scalar_conservation_propagation"
    assert sp.simplify(res.solution.rhs - sp.sin(x - 2 * t)) == 0


def test_verify_conservation_rarefaction_solution_smoke():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")(x, t)
    eq = sp.Eq(sp.diff(u, t) + u * sp.diff(u, x), 0)
    res = analyze_conservation_law(
        eq, u, (x, t), ics={"initial_profile": sp.Piecewise((0, x < 0), (1, True))}
    )
    assert isinstance(res.details["structured_result"], ConservationLawRarefactionResult)
    report = verify_conservation_law_solution(
        eq, sp.Eq(u, res.solution), u, (x, t), structured_result=res.details["structured_result"]
    )
    assert report.pde_verified is True
