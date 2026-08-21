import pytest
import sympy as sp

from pdesolve.classification import preprocess_pde_problem
from pdesolve.conditions import parse_conditions, summarize_condition_model
from pdesolve.dispatcher_support import as_verification_summary
from pdesolve.domains import DomainGeometry, infer_domain_geometry
from pdesolve.performance import clear_all_caches
from pdesolve.results import PDESolutionRecord
from pdesolve.solver_execution import temporary_method_handler
from pdesolve.solvers.coordinator import execute_planned_solver
from pdesolve.errors import PDEMethodNotApplicable


def _heat_problem_vars(order="xt"):
    x, t = sp.symbols("x t", real=True)
    ufun = sp.Function("u")
    if order == "xt":
        u = ufun(x, t)
        variables = (x, t)
        eq = sp.Eq(sp.diff(u, t), sp.diff(u, x, 2))
    else:
        u = ufun(t, x)
        variables = (t, x)
        eq = sp.Eq(sp.diff(u, t), sp.diff(u, x, 2))
    return x, t, u, variables, eq


def test_preprocess_cache_does_not_leak_details_mutation():
    clear_all_caches()
    x, t, u, variables, eq = _heat_problem_vars()
    first = preprocess_pde_problem(eq, u, variables)
    first.details["caller_marker"] = 123
    second = preprocess_pde_problem(eq, u, variables)
    assert first is not second
    assert "caller_marker" not in second.details


def test_clear_all_caches_clears_preprocess_cache():
    import pdesolve.classification as classification

    clear_all_caches()
    x, t, u, variables, eq = _heat_problem_vars()
    preprocess_pde_problem(eq, u, variables)
    assert classification._PREPROCESS_CACHE
    clear_all_caches()
    assert not classification._PREPROCESS_CACHE


def test_preprocess_cache_is_bounded():
    import pdesolve.classification as classification

    clear_all_caches()
    x = sp.symbols("x", real=True)
    u = sp.Function("u")(x)
    for index in range(classification._PREPROCESS_CACHE_MAXSIZE + 4):
        eq = sp.Eq(sp.diff(u, x), sp.Integer(index) * u)
        preprocess_pde_problem(eq, u, (x,))
    assert (
        len(classification._PREPROCESS_CACHE)
        <= classification._PREPROCESS_CACHE_MAXSIZE
    )


def test_initial_condition_kind_is_independent_of_variable_order():
    x, t, u, variables, _ = _heat_problem_vars(order="tx")
    profile = sp.Eq(u.subs(t, 0), sp.sin(x))
    velocity = sp.Eq(sp.Subs(sp.diff(u, t), t, 0), sp.cos(x))
    model = parse_conditions([profile, velocity], dep_expr=u, indep_vars=variables)
    summary = summarize_condition_model(model)
    assert model.metadata["time_variable"] == t
    assert summary["initial_kinds"] == ("profile", "velocity")


def test_verification_requires_all_supplied_obligations():
    summary = as_verification_summary(
        {"pde_verified": True, "verified": True},
        require_pde=True,
        require_initial=True,
        require_boundary=True,
    )
    assert summary.pde_verified is True
    assert summary.initial_verified is None
    assert summary.boundary_verified is None
    assert summary.verified is None
    assert summary.status == "unverified"


def test_unspecified_one_dimensional_domain_stays_unspecified():
    x = sp.symbols("x", real=True)
    domain = infer_domain_geometry(indep_vars=(x,))
    assert type(domain) is DomainGeometry
    assert domain.kind == "unspecified"


def test_expected_method_failure_uses_method_exception():
    class Problem:
        equation = sp.Eq(0, 0)
        assumptions = True
        details = {}

    def fail(_ctx):
        raise ValueError("not applicable")

    with temporary_method_handler("classification_only", fail):
        with pytest.raises(PDEMethodNotApplicable, match="not applicable"):
            execute_planned_solver(Problem(), "classification_only")


def test_unexpected_programming_error_is_not_swallowed():
    class Problem:
        equation = sp.Eq(0, 0)
        assumptions = True
        details = {}

    def fail(_ctx):
        raise TypeError("programming defect")

    with temporary_method_handler("classification_only", fail):
        with pytest.raises(TypeError, match="programming defect"):
            execute_planned_solver(Problem(), "classification_only")


def test_solution_record_uses_single_immutable_metadata_store():
    record = PDESolutionRecord(
        method="test",
        solution=sp.Integer(1),
        metadata={"value": 2},
        verification={"verified": None},
    )
    assert record.details is record.metadata
    assert record.details["value"] == 2
    with pytest.raises(TypeError):
        record.metadata["value"] = 3
