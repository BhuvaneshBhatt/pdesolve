import sympy as sp

from pdesolve.geometry import DistributionKD, VectorFieldKD, distribution_closure
from pdesolve.frobenius import local_frobenius_chart
from pdesolve.diffinv import first_order_differential_invariants_scalar


def test_involutive_noncommuting_affine_distribution_supported():
    x, y = sp.symbols("x y", real=True)
    X1 = VectorFieldKD((x, y), (1, 0))
    X2 = VectorFieldKD((x, y), (x, 1))
    dist = DistributionKD((x, y), (X1, X2))
    clo = distribution_closure(dist)
    assert clo.closed is True
    assert dist.is_commuting() is False
    chart = local_frobenius_chart(dist)
    assert len(chart.invariants) == 0
    assert len(chart.transverse) == 2
    assert chart.method in {
        "involutive_affine_constant_derivative_coords",
        "involutive_affine_rectified_coords",
        "involutive_affine_full_rank_identity_chart",
    }


def test_frobenius_engine_for_commuting_affine_distribution():
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    dist = DistributionKD(
        (x, y, t),
        (
            VectorFieldKD((x, y, t), (x, 0, 0)),
            VectorFieldKD((x, y, t), (0, 0, 1)),
        ),
    )
    chart = local_frobenius_chart(dist)
    assert len(chart.invariants) == 1
    assert len(chart.transverse) == 2
    assert chart.jacobian != 0


def test_first_order_differential_invariants_scalar_translation():
    x, t, u = sp.symbols("x t u", real=True)
    X = VectorFieldKD((x, t), (1, 1))
    out = first_order_differential_invariants_scalar(X, 0, u)
    assert len(out.base_invariants) == 1
    # For translation in x+t, the base invariant should be equivalent to x - t (up to scale/sign/add const)
    z = sp.expand(out.base_invariants[0])
    assert sp.diff(z, x) != 0 or sp.diff(z, t) != 0
    assert len(out.first_order_invariants) == 1
