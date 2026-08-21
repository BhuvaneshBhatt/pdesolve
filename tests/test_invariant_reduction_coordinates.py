import sympy as sp

from pdesolve.geometry import VectorFieldKD, DistributionKD
from pdesolve.frobenius import local_frobenius_chart, adapted_basis_in_chart
from pdesolve.diffinv import (
    second_order_differential_invariants_scalar,
    commuting_distribution_differential_invariants_scalar,
)
from pdesolve.pde import (
    build_scalar_jet_equation_from_sympy_pde,
    build_scalar_general_solved_pde_from_equation,
)
from pdesolve.reduction import reduce_scalar_by_frobenius_chart
from pdesolve.workflows import repeated_reduction_workflow_scalar_kd_frobenius_default


def test_second_order_differential_invariants_single_generator():
    x, t, u = sp.symbols("x t u", real=True)
    X = VectorFieldKD((x, t), (1, 1))
    out = second_order_differential_invariants_scalar(X, 0, u)
    assert len(out.base_invariants) == 1
    assert len(out.first_order_invariants) == 1
    assert len(out.second_order_invariants) == 1
    assert len(out.second_order_invariants[0]) == 1


def test_multi_generator_commuting_differential_invariants():
    x, y, t, u = sp.symbols("x y t u", positive=True, real=True)
    dist = DistributionKD(
        (x, y, t),
        (
            VectorFieldKD((x, y, t), (1, 0, 0)),
            VectorFieldKD((x, y, t), (0, 1, 0)),
        ),
    )
    out = commuting_distribution_differential_invariants_scalar(dist, u)
    assert len(out.base_invariants) == 1
    assert len(out.first_order_invariants) == 1
    assert len(out.second_order_invariants) == 1


def test_adapted_basis_noncommuting_involutive():
    x, y = sp.symbols("x y", real=True)
    X1 = VectorFieldKD((x, y), (1, 0))
    X2 = VectorFieldKD((x, y), (x, 1))
    dist = DistributionKD((x, y), (X1, X2))
    chart = local_frobenius_chart(dist)
    adapted = adapted_basis_in_chart(dist, chart.invariants + chart.transverse)
    assert len(adapted.adapted_fields) == 2
    assert adapted.transformation_matrix.shape == (2, 2)


def test_frobenius_backend_reduction_chart():
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
    assert len(red.invariants) == 1
    assert red.reduced_equation is not None


def test_repeated_workflow_frobenius_default_runs():
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
    res = repeated_reduction_workflow_scalar_kd_frobenius_default(
        eq_obj, max_steps=1, symmetry_degree=1, max_subset_size=2
    )
    assert len(res.steps) == 1
