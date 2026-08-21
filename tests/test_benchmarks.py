import sympy as sp

from pdesolve.benchmarks import BENCHMARK_CASES, run_benchmark_suite, run_benchmark_case


def _eq_expr(eq):
    return sp.expand(eq.lhs - eq.rhs)


def _eq_equiv(a, b):
    diff = sp.expand(_eq_expr(a) - _eq_expr(b))
    num, _ = sp.together(diff).as_numer_denom()
    num = sp.expand(num)
    return num == 0 or num.equals(0)


def test_benchmark_registry_has_required_categories():
    categories = {case.category for case in BENCHMARK_CASES}
    required = {
        "heat",
        "wave",
        "transport",
        "reaction_diffusion",
        "affine_scaling",
        "affine",
        "multi_symmetry",
        "mixed_principal",
        "higher_order",
        "workflow",
    }
    assert required.issubset(categories)


def test_full_benchmark_suite_runs():
    outcomes = run_benchmark_suite()
    assert len(outcomes) == len(BENCHMARK_CASES)
    assert all(out.success for out in outcomes)


def test_expected_reduced_equations_regressions():
    advection = run_benchmark_case("transport_advection_reduction")
    reaction = run_benchmark_case("reaction_diffusion_reduction")
    heat = run_benchmark_case("heat_translation_reduction")
    multi = run_benchmark_case("commuting_multi_symmetry_reduction")

    # Reuse the same symbols/assumptions that appear in the actual reduced equations.
    z1_adv = next(
        s for s in advection.details["reduced_equation"].free_symbols if s.name == "z1"
    )
    z1_re = next(
        s for s in reaction.details["reduced_equation"].free_symbols if s.name == "z1"
    )
    z1_h = next(
        s for s in heat.details["reduced_equation"].free_symbols if s.name == "z1"
    )
    z2_h = next(
        s for s in heat.details["reduced_equation"].free_symbols if s.name == "z2"
    )
    z1_m = next(
        s for s in multi.details["reduced_equation"].free_symbols if s.name == "z1"
    )
    c_re = next(
        s for s in reaction.details["reduced_equation"].free_symbols if s.name == "c"
    )
    c_h = next(
        s for s in heat.details["reduced_equation"].free_symbols if s.name == "c"
    )

    f = sp.Function("f")

    assert _eq_equiv(
        advection.details["reduced_equation"], sp.Eq(sp.diff(f(z1_adv), z1_adv), 0)
    )
    expected_reaction = sp.Eq(
        f(z1_re) ** 2
        - f(z1_re)
        - sp.diff(f(z1_re), (z1_re, 2))
        - sp.diff(f(z1_re), z1_re) / c_re,
        0,
    )
    assert _eq_equiv(reaction.details["reduced_equation"], expected_reaction)
    expected_heat = sp.Eq(
        sp.diff(f(z1_h, z2_h), (z1_h, 2))
        + sp.diff(f(z1_h, z2_h), (z2_h, 2))
        + sp.diff(f(z1_h, z2_h), z2_h) / c_h,
        0,
    )
    assert _eq_equiv(heat.details["reduced_equation"], expected_heat)
    assert _eq_equiv(
        multi.details["reduced_equation"], sp.Eq(sp.diff(f(z1_m), z1_m), 0)
    )
