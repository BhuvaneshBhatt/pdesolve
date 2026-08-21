import sympy as sp

from pdesolve.coordinates import pdesolve
from pdesolve.geometry import DistributionKD, VectorFieldKD
from pdesolve.jet_space import ScalarJetSpaceKD, build_scalar_general_solved_pde_from_equation
from pdesolve.reduction import reduce_scalar_by_translation_affine_kd
from pdesolve.symmetry import solve_determining_equations_with_polynomial_ansatz_scalar_general_kd


def test_translation_coordinates():
    x, y, t = sp.symbols("x y t", real=True)
    dist = DistributionKD((x, y, t), (VectorFieldKD((x, y, t), (1, 0, 0)),))
    coords = pdesolve(dist)
    assert len(coords.invariants) == 2
    assert len(coords.transverse) == 1


def test_heat_equation_build_and_symmetry():
    x, y, t = sp.symbols("x y t", real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    pde = sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)))
    eq_obj, info = build_scalar_general_solved_pde_from_equation(jet, pde, max_principal_order=3)
    assert info.principal_multiindex == (0, 0, 1)
    poly = solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(eq_obj, degree=1)
    assert len(poly.xi_solutions) == 3


def test_translation_reduction_runs():
    x, y, t, c = sp.symbols("x y t c", real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    eq_obj, _ = build_scalar_general_solved_pde_from_equation(
        jet,
        sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0))),
        max_principal_order=3,
    )
    red = reduce_scalar_by_translation_affine_kd(eq_obj, (1, 0, c), a=0, b=0)
    assert red.reduced_equation is not None
