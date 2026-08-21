from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import sympy as sp

from .coordinates import solve_with_diagnostics
from .geometry import DistributionKD, VectorFieldKD
from .jet_space import ScalarJetSpaceKD, build_scalar_general_solved_pde_from_equation
from .reduction import (
    reduce_scalar_by_translation_affine_kd,
    reduce_scalar_by_translation_subalgebra_kd,
)
from .symmetry import solve_determining_equations_with_polynomial_ansatz_scalar_general_kd
from .workflows import repeated_reduction_workflow_scalar_kd


@dataclass(frozen=True)
class BenchmarkOutcome:
    name: str
    category: str
    success: bool
    details: dict


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    category: str
    description: str
    runner: Callable[[], dict]

    def run(self) -> BenchmarkOutcome:
        details = self.runner()
        return BenchmarkOutcome(self.name, self.category, True, details)


def _is_scalar_multiplier(expr: sp.Expr) -> bool:
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Derivative):
            return False
        if getattr(node, "is_Function", False) and not isinstance(node, sp.Symbol):
            return False
    return sp.simplify(expr) != 0


def _equations_equivalent(lhs: sp.Equality, rhs: sp.Equality) -> bool:
    L = sp.expand(lhs.lhs - lhs.rhs)
    R = sp.expand(rhs.lhs - rhs.rhs)
    diff = sp.simplify(L - R)
    if diff == 0:
        return True
    for A, B in ((L, R), (R, L)):
        try:
            q = sp.simplify(sp.cancel(sp.together(A / B)))
            if _is_scalar_multiplier(q):
                return True
        except Exception:
            pass
    return False


def _assert_eq(actual, expected, message: str):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def _assert_equation_equivalent(actual: sp.Equality, expected: sp.Equality, message: str):
    if not _equations_equivalent(actual, expected):
        raise AssertionError(f"{message}: expected {expected}, got {actual}")


# -------------------------
# Benchmark case builders
# -------------------------


def case_heat_principal_selection() -> dict:
    x, y, t = sp.symbols("x y t", real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    pde = sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)))
    eq_obj, info = build_scalar_general_solved_pde_from_equation(jet, pde, max_principal_order=3)
    _assert_eq(info.principal_multiindex, (0, 0, 1), "Heat equation principal multi-index")
    _assert_eq(eq_obj.G, jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)), "Heat equation solved RHS")
    return {"principal_multiindex": info.principal_multiindex, "solved_rhs": eq_obj.G}


def case_wave_second_order_selection() -> dict:
    x, y, t = sp.symbols("x y t", real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    pde = sp.Eq(jet.coord((0, 0, 2)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)))
    eq_obj, info = build_scalar_general_solved_pde_from_equation(jet, pde, max_principal_order=3)
    _assert_eq(info.principal_multiindex, (0, 0, 2), "Wave equation principal multi-index")
    _assert_eq(eq_obj.G, jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)), "Wave equation solved RHS")
    return {"principal_multiindex": info.principal_multiindex, "solved_rhs": eq_obj.G}


def case_transport_advection_reduction() -> dict:
    x, t, a = sp.symbols("x t a", real=True)
    jet = ScalarJetSpaceKD((x, t), dep_name="u", max_order=1)
    pde = sp.Eq(jet.coord((0, 1)) + a * jet.coord((1, 0)), 0)
    eq_obj, info = build_scalar_general_solved_pde_from_equation(jet, pde, max_principal_order=2)
    _assert_eq(info.principal_multiindex, (0, 1), "Advection equation principal multi-index")
    red = reduce_scalar_by_translation_affine_kd(eq_obj, (1, a), a=0, b=0)
    _assert_eq(red.invariants, (x - t / a,), "Advection invariant")
    z1 = sp.Symbol("z1", real=True)
    f = sp.Function("f")
    expected = sp.Eq(sp.diff(f(z1), z1), 0)
    _assert_equation_equivalent(red.reduced_equation, expected, "Advection reduced equation")
    return {"invariants": red.invariants, "reduced_equation": red.reduced_equation}


def case_reaction_diffusion_reduction() -> dict:
    x, t, c = sp.symbols("x t c", real=True)
    jet = ScalarJetSpaceKD((x, t), dep_name="u", max_order=2)
    u = jet.u
    pde = sp.Eq(jet.coord((0, 1)), jet.coord((2, 0)) + u * (1 - u))
    eq_obj, info = build_scalar_general_solved_pde_from_equation(jet, pde, max_principal_order=2)
    _assert_eq(info.principal_multiindex, (0, 1), "Reaction-diffusion principal multi-index")
    red = reduce_scalar_by_translation_affine_kd(eq_obj, (1, c), a=0, b=0)
    _assert_eq(red.invariants, (x - t / c,), "Reaction-diffusion invariant")
    z1 = sp.Symbol("z1", real=True)
    f = sp.Function("f")
    expected = sp.Eq(f(z1) ** 2 - f(z1) - sp.diff(f(z1), (z1, 2)) - sp.diff(f(z1), z1) / c, 0)
    _assert_equation_equivalent(
        red.reduced_equation, expected, "Reaction-diffusion reduced equation"
    )
    return {"invariants": red.invariants, "reduced_equation": red.reduced_equation}


def case_heat_translation_reduction() -> dict:
    x, y, t, c = sp.symbols("x y t c", positive=True, real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    pde = sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)))
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(jet, pde, max_principal_order=3)
    red = reduce_scalar_by_translation_affine_kd(eq_obj, (1, 0, c), a=0, b=0)
    _assert_eq(red.invariants, (y, x - t / c), "Heat translation invariants")
    z1, z2 = sp.symbols("z1 z2", real=True)
    f = sp.Function("f")
    expected = sp.Eq(
        sp.diff(f(z1, z2), (z1, 2)) + sp.diff(f(z1, z2), (z2, 2)) + sp.diff(f(z1, z2), z2) / c, 0
    )
    _assert_equation_equivalent(red.reduced_equation, expected, "Heat translation reduced equation")
    return {"invariants": red.invariants, "reduced_equation": red.reduced_equation}


def case_scaling_coordinates() -> dict:
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    dist = DistributionKD((x, y, t), (VectorFieldKD((x, y, t), (x, y, 2 * t)),))
    coords, diag = solve_with_diagnostics(dist)
    _assert_eq(coords.invariants, (y / x, t / x**2), "Scaling invariants")
    _assert_eq(coords.transverse, (sp.log(x),), "Scaling transverse coordinate")
    return {
        "method": coords.method,
        "invariants": coords.invariants,
        "transverse": coords.transverse,
        "diagnostics": diag,
    }


def case_affine_generator_coordinates() -> dict:
    x, y, t = sp.symbols("x y t", real=True)
    dist = DistributionKD((x, y, t), (VectorFieldKD((x, y, t), (1, 0, y)),))
    coords, diag = solve_with_diagnostics(dist)
    if len(coords.invariants) != 2:
        raise AssertionError(f"Expected 2 invariants, got {coords.invariants}")
    if coords.method is None:
        raise AssertionError("Expected a coordinate-construction method for affine generator")
    return {
        "method": coords.method,
        "invariants": coords.invariants,
        "transverse": coords.transverse,
        "diagnostics": diag,
    }


def case_commuting_multi_symmetry_reduction() -> dict:
    x, y, t = sp.symbols("x y t", real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    pde = sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)))
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(jet, pde, max_principal_order=3)
    red = reduce_scalar_by_translation_subalgebra_kd(
        eq_obj, [(1, 0, 0), (0, 1, 0)], a_list=[0, 0], beta_list=[0, 0]
    )
    _assert_eq(red.invariants, (t,), "Commuting translation-subalgebra invariant")
    z1 = sp.Symbol("z1", real=True)
    f = sp.Function("f")
    expected = sp.Eq(sp.diff(f(z1), z1), 0)
    _assert_equation_equivalent(
        red.reduced_equation, expected, "Commuting translation-subalgebra reduced equation"
    )
    return {
        "invariants": red.invariants,
        "transverse": red.transverse_parameters,
        "reduced_equation": red.reduced_equation,
    }


def case_mixed_principal_manual_support() -> dict:
    x, y, t = sp.symbols("x y t", real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    pde = sp.Eq(jet.coord((1, 1, 0)), jet.coord((0, 0, 1)))
    eq_obj, info = build_scalar_general_solved_pde_from_equation(
        jet, pde, principal_multiindex=(1, 1, 0), max_principal_order=3
    )
    _assert_eq(info.principal_multiindex, (1, 1, 0), "Mixed-principal manual solved form")
    poly = solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(
        eq_obj, degree=1, include_dependent_var=True
    )
    if len(poly.xi_solutions) != 3:
        raise AssertionError("Expected three xi solutions in 3D mixed-principal problem")
    return {
        "principal_multiindex": info.principal_multiindex,
        "symmetry_family": tuple(poly.xi_solutions),
        "phi": poly.phi_solution,
    }


def case_higher_order_manual_support() -> dict:
    x, y, t = sp.symbols("x y t", real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=3)
    pde = sp.Eq(jet.coord((0, 0, 3)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)))
    eq_obj, info = build_scalar_general_solved_pde_from_equation(
        jet, pde, principal_multiindex=(0, 0, 3), max_principal_order=3
    )
    _assert_eq(info.principal_multiindex, (0, 0, 3), "Higher-order manual solved form")
    poly = solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(
        eq_obj, degree=1, include_dependent_var=True
    )
    if len(poly.xi_solutions) != 3:
        raise AssertionError("Expected three xi solutions in 3D higher-order problem")
    return {
        "principal_multiindex": info.principal_multiindex,
        "symmetry_family": tuple(poly.xi_solutions),
        "phi": poly.phi_solution,
    }


def case_repeated_reduction_workflow() -> dict:
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    pde = sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)))
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(jet, pde, max_principal_order=3)
    result = repeated_reduction_workflow_scalar_kd(
        eq_obj, max_steps=2, symmetry_degree=1, max_subset_size=2, prefer_commuting=True
    )
    if not result.steps:
        raise AssertionError("Expected at least one repeated-reduction step")
    first = result.steps[0]
    return {
        "num_steps": len(result.steps),
        "num_matches_first_step": len(first.matches),
        "final_equation": result.final_equation,
    }


BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "heat_principal_selection",
        "heat",
        "Heat-type equation principal solved-form selection",
        case_heat_principal_selection,
    ),
    BenchmarkCase(
        "wave_second_order_selection",
        "wave",
        "Wave-type equation second-order principal solved-form selection",
        case_wave_second_order_selection,
    ),
    BenchmarkCase(
        "transport_advection_reduction",
        "transport",
        "Transport/advection characteristic reduction",
        case_transport_advection_reduction,
    ),
    BenchmarkCase(
        "reaction_diffusion_reduction",
        "reaction_diffusion",
        "Reaction-diffusion traveling-wave reduction",
        case_reaction_diffusion_reduction,
    ),
    BenchmarkCase(
        "heat_translation_reduction",
        "heat",
        "Heat-type affine translation reduction with expected invariants and reduced PDE",
        case_heat_translation_reduction,
    ),
    BenchmarkCase(
        "scaling_coordinates",
        "affine_scaling",
        "Diagonal scaling invariant/transverse coordinate construction",
        case_scaling_coordinates,
    ),
    BenchmarkCase(
        "affine_generator_coordinates",
        "affine",
        "Affine local engine invariant construction",
        case_affine_generator_coordinates,
    ),
    BenchmarkCase(
        "commuting_multi_symmetry_reduction",
        "multi_symmetry",
        "Commuting multi-symmetry reduction to one variable",
        case_commuting_multi_symmetry_reduction,
    ),
    BenchmarkCase(
        "mixed_principal_manual_support",
        "mixed_principal",
        "Mixed-principal solved-form support",
        case_mixed_principal_manual_support,
    ),
    BenchmarkCase(
        "higher_order_manual_support",
        "higher_order",
        "Higher-order solved-form support",
        case_higher_order_manual_support,
    ),
    BenchmarkCase(
        "repeated_reduction_workflow",
        "workflow",
        "Repeated reduction workflow regression",
        case_repeated_reduction_workflow,
    ),
)


def iter_benchmark_cases() -> Iterable[BenchmarkCase]:
    return BENCHMARK_CASES


def get_benchmark_case(name: str) -> BenchmarkCase:
    for case in BENCHMARK_CASES:
        if case.name == name:
            return case
    raise KeyError(name)


def run_benchmark_case(name: str) -> BenchmarkOutcome:
    return get_benchmark_case(name).run()


def run_benchmark_suite() -> tuple[BenchmarkOutcome, ...]:
    return tuple(case.run() for case in BENCHMARK_CASES)


# ---------------- Additional reduction benchmarks ----------------


def case_noncommuting_involutive_chart():
    x, y = sp.symbols("x y", real=True)
    from .frobenius import local_frobenius_chart
    from .geometry import DistributionKD, VectorFieldKD

    dist = DistributionKD((x, y), (VectorFieldKD((x, y), (1, 0)), VectorFieldKD((x, y), (x, 1))))
    chart = local_frobenius_chart(dist)
    if len(chart.transverse) != 2:
        raise AssertionError("Expected full local chart for rank-2 involutive distribution.")
    return {"method": chart.method, "coords": chart.invariants + chart.transverse}


def case_higher_order_differential_invariants():
    x, t = sp.symbols("x t", real=True)
    from .diffinv import differential_invariants_scalar_up_to_order
    from .geometry import VectorFieldKD

    X = VectorFieldKD((x, t), (1, 1))
    res = differential_invariants_scalar_up_to_order(X, 0, sp.Symbol("u"), max_order=3)
    if 3 not in res.invariants_by_order:
        raise AssertionError("Missing third-order invariant tower.")
    return {
        "orders": tuple(sorted(res.invariants_by_order.keys())),
        "num_order3": len(res.invariants_by_order[3]),
    }


def case_managed_repeated_reduction_workflow():
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    from .jet_space import ScalarJetSpaceKD, build_scalar_general_solved_pde_from_equation
    from .workflows import repeated_reduction_workflow_scalar_kd_managed

    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    pde = sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)))
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(jet, pde, max_principal_order=2)
    out = repeated_reduction_workflow_scalar_kd_managed(eq_obj, max_steps=2)
    if len(out.history) < 1:
        raise AssertionError("Expected at least one managed history entry.")
    return {
        "num_steps": len(out.steps),
        "history_len": len(out.history),
        "seen": len(out.seen_signatures),
    }


def case_cache_smoke():
    from .geometry import DistributionKD, VectorFieldKD
    from .performance import cache_stats, clear_all_caches

    x, y = sp.symbols("x y", real=True)
    clear_all_caches()
    X1 = VectorFieldKD((x, y), (1, 0))
    X2 = VectorFieldKD((x, y), (x, 1))
    _ = X1.bracket(X2)
    dist = DistributionKD((x, y), (X1, X2))
    _ = dist.diagnostics()
    before = cache_stats()
    _ = X1.bracket(X2)
    _ = dist.diagnostics()
    after = cache_stats()
    if after.bracket_cache_info.hits < before.bracket_cache_info.hits:
        raise AssertionError("Bracket cache hits did not behave monotonically.")
    return {
        "bracket_hits": after.bracket_cache_info.hits,
        "diagnostics_hits": after.diagnostics_cache_info.hits,
    }


BENCHMARK_CASES = BENCHMARK_CASES + (
    BenchmarkCase(
        "noncommuting_involutive_chart",
        "reduction_geometry",
        "Noncommuting involutive chart construction",
        case_noncommuting_involutive_chart,
    ),
    BenchmarkCase(
        "higher_order_differential_invariants",
        "reduction_geometry",
        "Higher-order differential invariant tower",
        case_higher_order_differential_invariants,
    ),
    BenchmarkCase(
        "managed_repeated_reduction_workflow",
        "reduction_geometry",
        "Managed repeated reduction workflow",
        case_managed_repeated_reduction_workflow,
    ),
    BenchmarkCase(
        "cache_smoke",
        "reduction_geometry",
        "Cache smoke test for memoized kernels",
        case_cache_smoke,
    ),
)
