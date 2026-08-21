import sympy as sp

from pdesolve.geometry import (
    VectorFieldKD,
    DistributionKD,
    CharacteristicCoordinatesResult,
)
from pdesolve.frobenius import local_frobenius_chart
from pdesolve.diffinv import invariant_jet_bundle
from pdesolve.canonical import (
    canonicalize_coordinate_chart,
    canonicalize_reduced_equation,
)
from pdesolve.lie import (
    lie_algebra_structure_summary,
    choose_frobenius_friendly_subalgebras,
)
from pdesolve.pde import (
    build_scalar_jet_equation_from_sympy_pde,
    build_scalar_general_solved_pde_from_equation,
)
from pdesolve.reduction import reduce_scalar_by_frobenius_chart


def test_invariant_jet_bundle_for_translation_distribution():
    x, y, t = sp.symbols("x y t", real=True)
    dist = DistributionKD(
        (x, y, t),
        (
            VectorFieldKD((x, y, t), (1, 0, 0)),
            VectorFieldKD((x, y, t), (0, 1, 0)),
        ),
    )
    chart = local_frobenius_chart(dist)
    bundle = invariant_jet_bundle(dist, chart=chart, dep_name="w", max_order=2)
    assert len(bundle.chart.invariants) == 1
    assert bundle.reduced_jet.u == sp.Symbol("w")
    assert len(bundle.invariant_diff_ops) == 1
    pulled = bundle.pullback_reduced_jet_symbol((1,))
    assert pulled != 0


def test_canonicalize_coordinate_chart_normalizes_linear_coordinates():
    x, y = sp.symbols("x y", real=True)
    chart = CharacteristicCoordinatesResult(
        invariants=(2 * x + 4,),
        transverse=(3 * y + 6,),
        jacobian=sp.Integer(6),
        method="test",
        validity_conditions=(x, x),
    )
    out = canonicalize_coordinate_chart(chart, (x, y))
    assert sp.expand(out.invariants[0] - x) == 0
    assert sp.expand(out.transverse[0] - y) == 0
    assert len(out.validity_conditions) == 1


def test_canonicalize_reduced_equation_strips_denominator_and_prefactor():
    z = sp.symbols("z")
    f = sp.Function("f")
    eq = sp.Eq((2 * (f(z) + 1)) / 3, 0)
    out = canonicalize_reduced_equation(eq)
    assert sp.expand(out.lhs - (f(z) + 1)) == 0


def test_lie_structure_summary_and_frobenius_friendly_selection():
    x, y = sp.symbols("x y", real=True)
    X1 = VectorFieldKD((x, y), (1, 0))
    X2 = VectorFieldKD((x, y), (x, 1))
    dist = DistributionKD((x, y), (X1, X2))
    summary = lie_algebra_structure_summary(dist)
    assert summary.closure_closed
    assert summary.derived_dims[0] == 2
    cands = choose_frobenius_friendly_subalgebras(dist, max_dim=2)
    assert len(cands) >= 1
    assert cands[0].distribution.size >= 1


def test_frobenius_reduction_canonicalizes_output():
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    ufun = sp.Function("u")
    pde = sp.Eq(
        sp.diff(ufun(x, y, t), t),
        sp.diff(ufun(x, y, t), x, 2) + sp.diff(ufun(x, y, t), y, 2),
    )
    jet, jpde = build_scalar_jet_equation_from_sympy_pde(
        (x, y, t), ufun, pde, max_order=2, dep_name="u"
    )
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(
        jet, jpde, max_principal_order=2
    )
    dist = DistributionKD(
        (x, y, t),
        (
            VectorFieldKD((x, y, t), (1, 0, 0)),
            VectorFieldKD((x, y, t), (0, 1, 0)),
        ),
    )
    chart = local_frobenius_chart(dist)
    red = reduce_scalar_by_frobenius_chart(eq_obj, chart)
    # canonicalized chart should reduce invariants to just t up to simplification
    assert len(red.invariants) == 1
    assert sp.simplify(red.invariants[0] - t) == 0
    assert red.reduced_equation.rhs == 0
