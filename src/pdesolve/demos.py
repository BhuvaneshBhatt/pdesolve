from __future__ import annotations

import sympy as sp

from .coordinates import pdesolve
from .geometry import DistributionKD, VectorFieldKD
from .pde import ScalarJetSpaceKD, build_scalar_general_solved_pde_from_equation
from .reduction import (
    auto_reduce_best_commuting_subalgebra_scalar_kd,
    search_symbolic_linear_combinations_for_reduction_scalar_kd,
)
from .symmetry import (
    solve_determining_equations_with_polynomial_ansatz_scalar_general_kd,
)


def demo_characteristic_engine():
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    dist = DistributionKD(
        (x, y, t),
        (
            VectorFieldKD((x, y, t), (1, 0, 0)),
            VectorFieldKD((x, y, t), (0, 1, 0)),
        ),
    )
    coords = pdesolve(dist)
    return coords


def demo_heat_equation_pipeline():
    x, y, t = sp.symbols("x y t", positive=True, real=True)
    jet = ScalarJetSpaceKD((x, y, t), dep_name="u", max_order=2)
    pde = sp.Eq(jet.coord((0, 0, 1)), jet.coord((2, 0, 0)) + jet.coord((0, 2, 0)))
    eq_obj, info = build_scalar_general_solved_pde_from_equation(
        jet, pde, max_principal_order=3
    )
    poly = solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(
        eq_obj, degree=1
    )
    basis = poly.basis_vectors()
    matches = search_symbolic_linear_combinations_for_reduction_scalar_kd(
        eq_obj, basis, max_subset_size=2
    )
    reduced = auto_reduce_best_commuting_subalgebra_scalar_kd(
        eq_obj, matches, max_generators=2
    )
    return {
        "principal": info,
        "basis": basis,
        "matches": matches,
        "reduced": reduced,
    }
