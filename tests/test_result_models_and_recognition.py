import sympy as sp

from pdesolve import (
    CanonicalPDERepresentation,
    ClosedFormPDEResult,
    ImplicitPDEResult,
    build_pde_problem,
    pdesolve,
)
from pdesolve.classical_methods import PDEIVPResult
from pdesolve.complete_integral_helpers import (
    recognize_generalized_clairaut_pde,
    solve_generalized_clairaut_complete_integral,
)
from pdesolve.conservation_laws import (
    entropy_admissibility_scalar_riemann,
    solve_scalar_conservation_law_ivp,
    verify_weak_conservation_law_solution,
)
from pdesolve.dispatcher_support import coerce_result
from pdesolve.results import ConservationLawShockResult


def test_canonical_representation_layer_exposes_metadata():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))
    prob = build_pde_problem(
        eq, u(x, t), (x, t), ics={"initial_profile": sp.sin(x)}, bcs={"left": 0}
    )
    can = prob.canonical_representation
    assert isinstance(can, CanonicalPDERepresentation)
    assert can.order == 2
    assert can.linearity in {"linear", "quasilinear"}
    assert "integral_transform" in can.transformability_tags
    assert can.ic_metadata["provided"] is True


def test_recognition_is_separated_from_solution_construction():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    eq = sp.Eq(
        u(x, y),
        x * sp.diff(u(x, y), x)
        + y * sp.diff(u(x, y), y)
        + sp.sin(sp.diff(u(x, y), x) + sp.diff(u(x, y), y)),
    )
    rec = recognize_generalized_clairaut_pde(eq, u(x, y), (x, y))
    assert rec.recognized is True
    assert sp.simplify(rec.phi - sp.sin(rec.gradients[0] + rec.gradients[1])) == 0


def test_result_object_model_has_closed_and_implicit_forms():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    raw = PDEIVPResult(
        method="quasilinear_implicit_characteristics",
        solution=(sp.Eq(sp.Symbol("X0"), x),),
        details={"implicit": True},
    )
    rec = coerce_result(
        raw,
        attempted_methods=["quasilinear_implicit_characteristics"],
        trace_steps=[],
        canonical_eq=sp.Eq(sp.diff(u(x, t), t), 0),
        assumptions=True,
        dep_expr_or_func=u(x, t),
        indep_vars=(x, t),
    )
    assert isinstance(rec, ClosedFormPDEResult)
    assert rec.metadata["implicit"] is True
    implicit_view = ImplicitPDEResult(
        method=rec.method,
        solution=rec.solution,
        classification=None,
        assumptions=True,
        verification=rec.verification,
        metadata=rec.metadata,
    )
    assert implicit_view.method == rec.method


def test_entropy_admissibility_and_weak_verification_for_burgers_shock():
    u = sp.Symbol("u", real=True)
    flux = u**2 / 2
    admiss = entropy_admissibility_scalar_riemann(flux, 2, 0, "shock", shock_speed=1, u_symbol=u)
    assert admiss["admissible"] is True
    x, t = sp.symbols("x t", real=True)
    sol = ConservationLawShockResult(
        method="scalar_conservation_riemann_shock",
        solution=sp.Eq(sp.Function("w")(x, t), sp.Piecewise((2, x < t), (0, True))),
        flux=flux,
        left_state=2,
        right_state=0,
        shock_speed=1,
        details={"admissibility": admiss},
    )
    ver = verify_weak_conservation_law_solution(sol)
    assert ver.verified is True


def test_general_quasilinear_implicit_returns_formal_characteristic_system_for_nonautonomous_case():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq((1 + x) * sp.diff(u(x, t), x) + sp.diff(u(x, t), t), 0)
    res = pdesolve(
        eq, u(x, t), (x, t), method="quasilinear_implicit", ics={"initial_profile": x**2}
    )
    raw = res.details["raw_result"]
    assert raw.details.get("formal_characteristic_system") is True
    assert len(raw.solution) == 6


def test_generalized_clairaut_solver_exposes_singular_envelope_solution():
    x, y = sp.symbols("x y", real=True)
    u = sp.Function("u")
    ux = sp.diff(u(x, y), x)
    uy = sp.diff(u(x, y), y)
    eq = sp.Eq(u(x, y), x * ux + y * uy + (ux**2 + uy**2) / 2)
    res = solve_generalized_clairaut_complete_integral(eq, u(x, y), (x, y))
    sing = res.details.get("singular_solution")
    assert sing is not None
    rhs = sing.rhs if isinstance(sing, sp.Equality) else sing[0].rhs
    assert sp.simplify(rhs + (x**2 + y**2) / 2) == 0


def test_structured_conservation_solver_attaches_admissibility_metadata():
    x, t = sp.symbols("x t", real=True)
    ufun = sp.Function("u")
    eq = sp.Eq(sp.diff(ufun(x, t), t) + ufun(x, t) * sp.diff(ufun(x, t), x), 0)
    structured = solve_scalar_conservation_law_ivp(
        eq, ufun(x, t), (x, t), initial_conditions={"riemann_data": (2, 0)}
    )
    assert structured.details.get("admissibility", {}).get("admissible") is True


def test_planner_details_use_condition_and_domain_models():
    import sympy as sp

    from pdesolve.classification import plan_pde_solution_methods

    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))
    plan = plan_pde_solution_methods(
        eq,
        u(x, t),
        (x, t),
        ics=[sp.Eq(u(x, 0), sp.sin(x))],
        bcs=[sp.Eq(u(0, t), 0), sp.Eq(u(sp.pi, t), 0)],
    )
    assert plan.details["condition_model"] is not None
    assert plan.details["domain_geometry"] is not None
    assert getattr(plan.details["domain_geometry"], "kind", None) == "interval"
