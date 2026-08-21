import sympy as sp

from pdesolve.geometry import VectorFieldKD, DistributionKD
from pdesolve.frobenius import (
    restricted_local_frobenius_atlas,
    local_frobenius_chart_explain,
    local_frobenius_chart,
)
from pdesolve.verify import verify_frobenius_chart, verify_reduction
from pdesolve.pde import (
    build_scalar_jet_equation_from_sympy_pde,
    build_scalar_general_solved_pde_from_equation,
)
from pdesolve.reduction import (
    reduce_scalar_by_frobenius_chart,
    reduce_scalar_by_commuting_affine_subalgebra_kd,
)
from pdesolve.workflows import repeated_reduction_workflow_scalar_kd_frobenius_default


def _heat_eq_obj():
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    u = sp.Function("u")
    pde = sp.Eq(
        sp.diff(u(x, y, t), t), sp.diff(u(x, y, t), x, 2) + sp.diff(u(x, y, t), y, 2)
    )
    jet, jpde = build_scalar_jet_equation_from_sympy_pde(
        (x, y, t), u, pde, max_order=2, dep_name="u"
    )
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(
        jet, jpde, max_principal_order=2
    )
    return eq_obj


def test_chart_atlas_and_explainability_for_involutive_affine_distribution():
    x, y = sp.symbols("x y", real=True)
    dist = DistributionKD(
        (x, y), (VectorFieldKD((x, y), (1, 0)), VectorFieldKD((x, y), (x, 1)))
    )
    atlas = restricted_local_frobenius_atlas(dist)
    assert atlas.best() is not None
    report = local_frobenius_chart_explain(dist)
    assert report["success"] is True
    assert report["chart_count"] >= 1


def test_verify_frobenius_chart_for_translation_distribution():
    x, y, t = sp.symbols("x y t", real=True)
    dist = DistributionKD(
        (x, y, t),
        (VectorFieldKD((x, y, t), (1, 0, 0)), VectorFieldKD((x, y, t), (0, 1, 0))),
    )
    chart = local_frobenius_chart(dist)
    ver = verify_frobenius_chart(dist, chart)
    assert ver.valid is True
    assert sp.simplify(ver.jacobian) != 0


def test_verify_reduction_for_frobenius_reduced_heat_equation():
    eq_obj = _heat_eq_obj()
    x, y, t = eq_obj.jet.xs
    dist = DistributionKD(
        (x, y, t),
        (VectorFieldKD((x, y, t), (1, 0, 0)), VectorFieldKD((x, y, t), (0, 1, 0))),
    )
    chart = local_frobenius_chart(dist)
    red = reduce_scalar_by_frobenius_chart(eq_obj, chart)
    ver = verify_reduction(eq_obj, red, chart)
    assert ver.valid is True


def test_general_commuting_affine_subalgebra_reduction_runs():
    eq_obj = _heat_eq_obj()
    _, _, t = eq_obj.jet.xs
    red = reduce_scalar_by_commuting_affine_subalgebra_kd(
        eq_obj,
        Xis_list=[(1, 0, 0), (0, 1, t)],
        a_list=[0, 0],
        beta_list=[0, 0],
    )
    assert len(red.invariants) == 1
    assert red.reduced_equation.rhs == 0


def test_repeated_workflow_records_verification_or_diagnostics():
    eq_obj = _heat_eq_obj()
    out = repeated_reduction_workflow_scalar_kd_frobenius_default(
        eq_obj, max_steps=1, symmetry_degree=1, max_subset_size=2
    )
    assert len(out.steps) == 1
    step = out.steps[0]
    assert step.reduced is not None
    assert step.verification is not None
