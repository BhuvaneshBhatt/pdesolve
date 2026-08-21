import sympy as sp

from pdesolve.geometry import DistributionKD, VectorFieldKD
from pdesolve.coordinates import pdesolve, solve_with_diagnostics
from pdesolve.pde import ScalarJetSpaceKD, build_scalar_general_solved_pde_from_equation
from pdesolve.reduction import (
    reduce_scalar_by_commuting_affine_subalgebra_kd,
    reduce_scalar_by_projectable_affine_generator_transport,
)


def test_commuting_affine_distribution_coordinates():
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    dist = DistributionKD(
        (x, y, t),
        (
            VectorFieldKD((x, y, t), (x, 0, 0)),
            VectorFieldKD((x, y, t), (0, 0, 1)),
        ),
    )
    coords = pdesolve(dist)
    assert len(coords.invariants) == 1
    assert len(coords.transverse) == 2
    assert coords.method in {
        "commuting_affine_constant_derivative_coords",
        "common_linear_first_integrals",
    }


def test_commuting_affine_subalgebra_reduction_runs():
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(
        jet,
        sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0))),
        max_principal_order=3,
    )
    red = reduce_scalar_by_commuting_affine_subalgebra_kd(
        eq_obj,
        Xis_list=[(x, 0, 0), (0, 0, 1)],
        a_list=[0, 0],
        beta_list=[0, 0],
    )
    assert red.reduced_equation is not None


def test_local_engine_with_diagnostics():
    x, y = sp.symbols("x y", real=True)
    dist = DistributionKD((x, y), (VectorFieldKD((x, y), (1, 0)),))
    coords, diag = solve_with_diagnostics(dist)
    assert coords is not None
    assert diag.commuting is True
    assert diag.chosen_method is not None


def test_general_projectable_affine_transport_runs():
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(
        jet,
        sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0))),
        max_principal_order=3,
    )
    red = reduce_scalar_by_projectable_affine_generator_transport(eq_obj, (x, 0, 0), 0)
    assert red.reduced_equation is not None
