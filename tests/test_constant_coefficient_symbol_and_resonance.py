import sympy as sp

from pdesolve.constant_coeff import (
    _cc_operator_apply_from_terms,
    build_constant_coefficient_homogeneous_solution,
    build_constant_coefficient_operator_profile,
    detect_linear_constant_coefficient_pde,
    invert_constant_coefficient_operator_on_forcing,
    invert_factored_constant_coefficient_operator_on_forcing,
)


def _residual(eq, expr, uexpr, vars_):
    ccpde = detect_linear_constant_coefficient_pde(eq, uexpr, vars_)
    return sp.simplify(_cc_operator_apply_from_terms(ccpde.operator_terms, expr, vars_) - ccpde.rhs)


def test_symbol_layer_reports_shift_and_lowest_term():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t, 2) - 2 * sp.diff(u(x, t), t, x) + sp.diff(u(x, t), x, 2), 0)
    profile = build_constant_coefficient_operator_profile(eq, u(x, t), (x, t))
    shifted = profile.symbol.shift((1, 1))
    assert profile.symbol.evaluate((1, 1)) == 0
    assert shifted.resonance_multiplicity == 2
    assert shifted.lowest_term is not None
    assert shifted.lowest_term.total_degree == 2


def test_resonance_multiplicity_lifts_exponential_ansatz():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    eq = sp.Eq(
        sp.diff(u(x, t), t, 2) - 2 * sp.diff(u(x, t), t, x) + sp.diff(u(x, t), x, 2), sp.exp(x + t)
    )
    result = invert_constant_coefficient_operator_on_forcing(eq, u(x, t), (x, t))
    assert result.details["resonant"] is True
    assert result.details["resonance_multiplicity"] == 2
    assert sp.simplify(_residual(eq, result.solution, u(x, t), (x, t))) == 0


def test_factor_pipeline_metadata_is_attached():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    eq = sp.Eq((sp.diff(u(x, t), t) + 2 * u(x, t)) - (sp.diff(u(x, t), x) - u(x, t)), sp.exp(t))
    result = invert_factored_constant_coefficient_operator_on_forcing(eq, u(x, t), (x, t))
    pipeline = result.details["factor_pipeline"]
    assert pipeline
    assert all("factor" in stage for stage in pipeline)
    assert sp.simplify(_residual(eq, result.solution, u(x, t), (x, t))) == 0


def test_multivariate_polynomial_exponential_forcing_is_supported():
    x, y, z = sp.symbols("x y z")
    u = sp.Function("u")
    rhs = x * y * sp.exp(x + y + z)
    eq = sp.Eq(
        sp.diff(u(x, y, z), x) + sp.diff(u(x, y, z), y) + sp.diff(u(x, y, z), z) + u(x, y, z), rhs
    )
    result = invert_constant_coefficient_operator_on_forcing(eq, u(x, y, z), (x, y, z))
    assert result.details["resonance_multiplicity"] == 0
    assert sp.simplify(_residual(eq, result.solution, u(x, y, z), (x, y, z))) == 0


def test_multivariate_polynomial_trig_forcing_is_supported():
    x, y, z = sp.symbols("x y z")
    u = sp.Function("u")
    rhs = x * sp.cos(x + y + z)
    eq = sp.Eq(
        sp.diff(u(x, y, z), x) + sp.diff(u(x, y, z), y) + sp.diff(u(x, y, z), z) + 2 * u(x, y, z),
        rhs,
    )
    result = invert_constant_coefficient_operator_on_forcing(eq, u(x, y, z), (x, y, z))
    assert (
        sp.simplify(
            _residual(eq, sp.expand(result.solution.rewrite(sp.exp)), u(x, y, z), (x, y, z))
            .rewrite(sp.sin)
            .rewrite(sp.cos)
        )
        == 0
    )


def test_homogeneous_family_generalizes_to_three_variables():
    x, y, z = sp.symbols("x y z")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, y, z), x) + 2 * sp.diff(u(x, y, z), y) + 3 * sp.diff(u(x, y, z), z), 0)
    result = build_constant_coefficient_homogeneous_solution(eq, u(x, y, z), (x, y, z))
    family = result.details["families"][0]
    assert len(family.invariants) == 2
    assert len(family.generators) == 1
    assert sp.simplify(_residual(eq, result.solution.rhs, u(x, y, z), (x, y, z))) == 0


def test_polynomial_forcing_uses_structured_ansatz_or_inverse():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    rhs = x**2 + x * t
    eq = sp.Eq(sp.diff(u(x, t), x) + sp.diff(u(x, t), t) + 3 * u(x, t), rhs)
    result = invert_constant_coefficient_operator_on_forcing(eq, u(x, t), (x, t))
    if result.method == "constant_coefficient_sum_split":
        styles = {part.details["annihilator_style"] for part in result.details["parts"]}
        assert styles <= {"truncated_inverse", "polynomial_ansatz"}
    else:
        assert result.details["annihilator_style"] in {"truncated_inverse", "polynomial_ansatz"}
    assert sp.simplify(_residual(eq, result.solution, u(x, t), (x, t))) == 0
