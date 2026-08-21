import sympy as sp

from pdesolve.geometry import VectorFieldKD, DistributionKD
from pdesolve.diffinv import (
    differential_invariants_scalar_up_to_order,
    differential_invariants_commuting_distribution_scalar_up_to_order,
)
from pdesolve.workflows import repeated_reduction_workflow_scalar_kd_managed
from pdesolve.pde import ScalarJetSpaceKD, build_scalar_general_solved_pde_from_equation
from pdesolve.performance import clear_all_caches, cache_stats
from pdesolve.benchmarks import run_benchmark_case


def _heat_eq_obj():
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    pde = sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)))
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(
        jet, pde, max_principal_order=2
    )
    return eq_obj


def test_higher_order_scalar_differential_invariant_tower():
    x, t = sp.symbols("x t", real=True)
    X = VectorFieldKD((x, t), (1, 1))
    res = differential_invariants_scalar_up_to_order(X, 0, sp.Symbol("u"), max_order=3)
    assert 0 in res.invariants_by_order and 3 in res.invariants_by_order
    assert len(res.invariants_by_order[1]) >= 1


def test_higher_order_commuting_distribution_differential_invariant_tower():
    x, y, t = sp.symbols("x y t", real=True)
    dist = DistributionKD(
        (x, y, t),
        (VectorFieldKD((x, y, t), (1, 0, 0)), VectorFieldKD((x, y, t), (0, 1, 0))),
    )
    res = differential_invariants_commuting_distribution_scalar_up_to_order(
        dist, sp.Symbol("u"), max_order=2
    )
    assert 2 in res.invariants_by_order
    assert len(res.base_invariants) == 1


def test_managed_repeated_reduction_workflow_tracks_steps_and_seen_signatures():
    eq_obj = _heat_eq_obj()
    out = repeated_reduction_workflow_scalar_kd_managed(eq_obj, max_steps=2)
    assert len(out.history) >= 1
    assert len(out.seen_signatures) >= 1


def test_memoization_smoke_for_brackets_and_diagnostics():
    x, y = sp.symbols("x y", real=True)
    clear_all_caches()
    X1 = VectorFieldKD((x, y), (1, 0))
    X2 = VectorFieldKD((x, y), (x, 1))
    dist = DistributionKD((x, y), (X1, X2))
    _ = X1.bracket(X2)
    _ = dist.diagnostics()
    before = cache_stats()
    _ = X1.bracket(X2)
    _ = dist.diagnostics()
    after = cache_stats()
    assert after.bracket_cache_info.hits >= before.bracket_cache_info.hits
    assert after.diagnostics_cache_info.hits >= before.diagnostics_cache_info.hits


def test_benchmark_cases_run():
    for name in [
        "noncommuting_involutive_chart",
        "higher_order_differential_invariants",
        "managed_repeated_reduction_workflow",
        "cache_smoke",
    ]:
        out = run_benchmark_case(name)
        assert out.success is True
