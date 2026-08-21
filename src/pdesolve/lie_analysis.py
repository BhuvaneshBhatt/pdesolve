from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .frobenius import local_frobenius_chart
from .geometry import DistributionKD, VectorFieldKD
from .jet_space import (
    build_scalar_general_solved_pde_from_equation,
    build_scalar_jet_equation_from_sympy_pde,
)
from .symmetry import (
    determining_equations_for_scalar_general_solved_pde_kd,
    solve_determining_equations_with_polynomial_ansatz_scalar_general_kd,
)


@dataclass(frozen=True)
class LiePointSymmetryAnalysis:
    determining_equations: tuple[sp.Equality, ...]
    xi_functions: tuple[sp.Expr, ...]
    phi_function: sp.Expr
    polynomial_solution: object | None
    generators: tuple[tuple[tuple[sp.Expr, ...], sp.Expr], ...]


def analyze_lie_point_symmetries(
    equation,
    dep_function,
    indep_vars,
    *,
    polynomial_degree: int = 1,
    include_dependent_var: bool = True,
):
    """High-level Lie point-symmetry analysis for a scalar solved PDE.

    The determining equations are exact.  Solving them is presently restricted
    to a polynomial ansatz, which makes the scope explicit and reproducible.
    """
    vars_ = tuple(indep_vars)
    dep = (
        dep_function.func
        if getattr(dep_function, "is_Function", False) and dep_function.args
        else dep_function
    )
    zero = (
        equation.lhs - equation.rhs if isinstance(equation, sp.Equality) else sp.sympify(equation)
    )
    max_order = max(
        (
            sum(count for _, count in node.variable_count)
            for node in sp.preorder_traversal(zero)
            if isinstance(node, sp.Derivative)
        ),
        default=1,
    )
    jet, jet_eq = build_scalar_jet_equation_from_sympy_pde(
        vars_, dep, equation, max_order=max_order, dep_name=getattr(dep, "__name__", "u")
    )
    solved, _ = build_scalar_general_solved_pde_from_equation(jet, jet_eq)
    xis, phi, det = determining_equations_for_scalar_general_solved_pde_kd(solved)
    poly = None
    gens = ()
    try:
        poly = solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(
            solved,
            degree=polynomial_degree,
            include_dependent_var=include_dependent_var,
            preserve_free_parameters=True,
        )
        gens = tuple((tuple(xis_), phi_) for xis_, phi_, _ in poly.basis_vectors())
    except Exception:
        pass
    return LiePointSymmetryAnalysis(tuple(det), tuple(xis), phi, poly, gens)


@dataclass(frozen=True)
class LieInvariantCoordinates:
    invariants: tuple[sp.Expr, ...]
    transverse: tuple[sp.Expr, ...]
    jacobian: sp.Expr
    validity_conditions: tuple[sp.Expr, ...]


def invariants_from_point_generator(indep_vars, dep_symbol, xi_coeffs, phi=0):
    """Construct local invariants of one point-symmetry generator via Frobenius coordinates."""
    vars_all = tuple(indep_vars) + (dep_symbol,)
    coeffs = tuple(map(sp.sympify, xi_coeffs)) + (sp.sympify(phi),)
    if len(coeffs) != len(vars_all):
        raise ValueError(
            "generator coefficient count does not match independent variables + dependent symbol"
        )
    dist = DistributionKD(vars_all, (VectorFieldKD(vars_all, coeffs),))
    chart = local_frobenius_chart(dist)
    return LieInvariantCoordinates(
        tuple(chart.invariants),
        tuple(chart.transverse),
        chart.jacobian,
        tuple(chart.validity_conditions),
    )


__all__ = [
    "LiePointSymmetryAnalysis",
    "analyze_lie_point_symmetries",
    "LieInvariantCoordinates",
    "invariants_from_point_generator",
]
