from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from .frobenius import local_frobenius_chart
from .geometry import DistributionKD, VectorFieldKD
from .invariantjet import InvariantJetBundle, build_invariant_jet_bundle_from_chart


@dataclass
class DifferentialInvariantResult:
    base_invariants: tuple[sp.Expr, ...]
    dependent_invariant: sp.Expr
    first_order_invariants: tuple[sp.Expr, ...]
    method: str


def _decompose_affine_u_action(Phi, xs, u):
    Phi = sp.expand(Phi)
    a = sp.expand(sp.diff(Phi, u))
    if any(v in a.free_symbols for v in list(xs) + [u]):
        return None
    beta = sp.expand(Phi - a * u)
    if any(v in beta.free_symbols for v in list(xs) + [u]):
        return None
    return a, beta


def _dependent_invariant(u, s_expr, a, beta):
    a = sp.sympify(a)
    beta = sp.sympify(beta)
    if sp.simplify(a) != 0:
        return sp.expand(sp.exp(-a * s_expr) * (u + beta / a))
    return sp.expand(u - beta * s_expr)


def _first_order_invariants_from_coordinates(vars, invariants, transverse_single, I0):
    coords = tuple(invariants) + (transverse_single,)
    J = sp.Matrix([[sp.diff(c, v) for v in vars] for c in coords])
    try:
        Jinv = sp.simplify(J.inv())
    except Exception:
        Jinv = J.inv()

    gradI = sp.Matrix([sp.diff(I0, v) for v in vars])
    # ∂/∂z_j uses column j of J^{-1}
    out = []
    for j in range(len(invariants)):
        coeffs = Jinv[:, j]
        expr = sp.expand(sum(coeffs[i, 0] * gradI[i, 0] for i in range(len(vars))))
        out.append(expr)
    return tuple(out)


def first_order_differential_invariants_scalar(
    field: VectorFieldKD, Phi: sp.Expr, u: sp.Symbol
) -> DifferentialInvariantResult:
    """
    Compute a restricted set of scalar first-order differential invariants for one
    projectable affine generator:

        X = sum_i Xi(x) d/dx_i + Phi(x,u) d/du,

    where Phi = a*u + beta with constant a,beta and the independent-variable part
    is handled by the restricted local Frobenius engine.

    Returns
    -------
    DifferentialInvariantResult
        base invariants z^alpha,
        normalized dependent invariant I0,
        first-order differential invariants D_{z^alpha}(I0).
    """
    xs = field.vars
    dec = _decompose_affine_u_action(Phi, xs, u)
    if dec is None:
        raise ValueError("Phi must be affine in u with constant coefficients.")
    a, beta = dec

    dist = DistributionKD(xs, (VectorFieldKD(xs, field.coeffs),))
    chart = local_frobenius_chart(dist)
    if len(chart.transverse) != 1:
        raise ValueError("Single-generator characteristic chart expected.")

    s_expr = chart.transverse[0]
    invariants = chart.invariants
    I0 = _dependent_invariant(u, s_expr, a, beta)
    first = _first_order_invariants_from_coordinates(xs, invariants, s_expr, I0)

    return DifferentialInvariantResult(
        base_invariants=invariants,
        dependent_invariant=I0,
        first_order_invariants=first,
        method="restricted_local_frobenius_first_order",
    )


@dataclass
class InvariantDifferentiationOperators:
    invariant_variables: tuple[sp.Expr, ...]
    transverse_variables: tuple[sp.Expr, ...]
    operators: tuple[VectorFieldKD, ...]
    method: str

    def apply(self, index: int, expr: sp.Expr) -> sp.Expr:
        return self.operators[index].apply(expr)


@dataclass
class HigherDifferentialInvariantResult:
    base_invariants: tuple[sp.Expr, ...]
    dependent_invariant: sp.Expr
    first_order_invariants: tuple[sp.Expr, ...]
    second_order_invariants: tuple[tuple[sp.Expr, ...], ...]
    operators: InvariantDifferentiationOperators
    method: str


def invariant_differentiation_operators(
    distribution: DistributionKD,
) -> InvariantDifferentiationOperators:
    """
    Construct invariant differentiation operators D_{z^alpha} from a characteristic chart.
    """
    chart = local_frobenius_chart(distribution)
    xs = distribution.vars
    invariants = chart.invariants
    transverse = chart.transverse
    if len(invariants) == 0:
        return InvariantDifferentiationOperators(
            tuple(), transverse, tuple(), chart.method
        )
    coords = tuple(invariants) + tuple(transverse)
    J = sp.Matrix([[sp.diff(c, v) for v in xs] for c in coords])
    try:
        Jinv = sp.simplify(J.inv())
    except Exception:
        Jinv = J.inv()
    ops = []
    for j in range(len(invariants)):
        coeffs = tuple(sp.expand(Jinv[i, j]) for i in range(len(xs)))
        ops.append(VectorFieldKD(xs, coeffs))
    return InvariantDifferentiationOperators(
        tuple(invariants), tuple(transverse), tuple(ops), chart.method
    )


def _u_action_compatibility(a_list, beta_list):
    r = len(a_list)
    for i in range(r):
        for j in range(i + 1, r):
            if sp.simplify(beta_list[i] * a_list[j] - beta_list[j] * a_list[i]) != 0:
                return False
    return True


def _dependent_invariant_multi(u, s_exprs, a_list, beta_list):
    a_list = [sp.sympify(v) for v in a_list]
    beta_list = [sp.sympify(v) for v in beta_list]
    s_exprs = tuple(map(sp.expand, s_exprs))
    if not _u_action_compatibility(a_list, beta_list):
        raise ValueError("Dependent affine actions are not compatible.")
    if all(sp.simplify(a) == 0 for a in a_list):
        return sp.expand(
            u - sum(beta_list[j] * s_exprs[j] for j in range(len(s_exprs)))
        )
    kappa = None
    for a, beta in zip(a_list, beta_list):
        if sp.simplify(a) != 0:
            cand = sp.simplify(beta / a)
            if kappa is None:
                kappa = cand
            elif sp.simplify(cand - kappa) != 0:
                raise ValueError(
                    "Dependent affine actions are not compatible for common scalar normalization."
                )
        elif sp.simplify(beta) != 0:
            raise ValueError(
                "Dependent affine actions are not compatible for common scalar normalization."
            )
    phase = sp.expand(sum(a_list[j] * s_exprs[j] for j in range(len(s_exprs))))
    return sp.expand(sp.exp(-phase) * (u + kappa))


def second_order_differential_invariants_scalar(
    field: VectorFieldKD, Phi: sp.Expr, u: sp.Symbol
) -> HigherDifferentialInvariantResult:
    """Scalar differential invariants for one projectable affine generator.

    Returns first- and second-order invariant derivatives of the normalized dependent invariant.
    """
    dist = DistributionKD(field.vars, (field,))
    ops = invariant_differentiation_operators(dist)
    dec = _decompose_affine_u_action(Phi, field.vars, u)
    if dec is None:
        raise ValueError("Phi must be affine in u with constant coefficients.")
    a, beta = dec
    if len(ops.transverse_variables) != 1:
        raise ValueError("Single-generator characteristic chart expected.")
    I0 = _dependent_invariant(u, ops.transverse_variables[0], a, beta)
    first = tuple(op.apply(I0) for op in ops.operators)
    second = tuple(
        tuple(op_i.apply(first[j]) for j in range(len(first))) for op_i in ops.operators
    )
    return HigherDifferentialInvariantResult(
        base_invariants=ops.invariant_variables,
        dependent_invariant=I0,
        first_order_invariants=first,
        second_order_invariants=second,
        operators=ops,
        method=f"{ops.method}_second_order",
    )


def commuting_distribution_differential_invariants_scalar(
    distribution: DistributionKD,
    u: sp.Symbol,
    a_list: Sequence[sp.Expr] | None = None,
    beta_list: Sequence[sp.Expr] | None = None,
) -> HigherDifferentialInvariantResult:
    """
    Differential invariants for a commuting distribution with compatible scalar affine u-actions.

    The distribution acts only on the independent variables. The dependent action is provided
    by the scalar coefficients a_m, beta_m for each generator.
    """
    if not distribution.is_commuting():
        raise ValueError(
            "Distribution must commute for this restricted multi-generator differential invariant engine."
        )
    r = distribution.size
    if a_list is None:
        a_list = [0] * r
    if beta_list is None:
        beta_list = [0] * r
    if len(a_list) != r or len(beta_list) != r:
        raise ValueError("a_list and beta_list must have one entry per generator.")

    chart = local_frobenius_chart(distribution)
    coords = tuple(chart.invariants) + tuple(chart.transverse)
    xs = distribution.vars
    if len(chart.invariants) == 0:
        ops = InvariantDifferentiationOperators(
            tuple(), chart.transverse, tuple(), chart.method
        )
        I0 = _dependent_invariant_multi(u, chart.transverse, a_list, beta_list)
        return HigherDifferentialInvariantResult(
            tuple(), I0, tuple(), tuple(), ops, f"{chart.method}_multi"
        )

    J = sp.Matrix([[sp.diff(c, v) for v in xs] for c in coords])
    try:
        Jinv = sp.simplify(J.inv())
    except Exception:
        Jinv = J.inv()
    diff_ops = []
    for j in range(len(chart.invariants)):
        coeffs = tuple(sp.expand(Jinv[i, j]) for i in range(len(xs)))
        diff_ops.append(VectorFieldKD(xs, coeffs))
    ops = InvariantDifferentiationOperators(
        tuple(chart.invariants), tuple(chart.transverse), tuple(diff_ops), chart.method
    )

    I0 = _dependent_invariant_multi(u, chart.transverse, a_list, beta_list)
    first = tuple(op.apply(I0) for op in ops.operators)
    second = tuple(
        tuple(op_i.apply(first[j]) for j in range(len(first))) for op_i in ops.operators
    )
    return HigherDifferentialInvariantResult(
        base_invariants=ops.invariant_variables,
        dependent_invariant=I0,
        first_order_invariants=first,
        second_order_invariants=second,
        operators=ops,
        method=f"{chart.method}_commuting_distribution",
    )


# ------------------ Reduced jet machinery in invariant coordinates ------------------


def invariant_jet_bundle(
    distribution: DistributionKD, chart=None, dep_name: str = "w", max_order: int = 2
) -> InvariantJetBundle:
    """Build reduced jet machinery in invariant coordinates from a Frobenius chart.

    This is the main invariant-jet entry point for invariant differentiation / reduced jets.
    """
    return build_invariant_jet_bundle_from_chart(
        distribution, chart=chart, dep_name=dep_name, max_order=max_order
    )


@dataclass
class DifferentialInvariantTowerResult:
    base_invariants: tuple[sp.Expr, ...]
    dependent_invariant: sp.Expr
    operators: InvariantDifferentiationOperators
    invariants_by_order: dict[int, tuple[sp.Expr, ...]]
    method: str


def _flatten_unique(exprs):
    out = []
    seen = set()
    for expr in exprs:
        sig = sp.srepr(sp.expand(expr))
        if sig not in seen:
            seen.add(sig)
            out.append(sp.expand(expr))
    return tuple(out)


def _next_order_invariant_derivatives(
    operators: InvariantDifferentiationOperators, prev: tuple[sp.Expr, ...]
) -> tuple[sp.Expr, ...]:
    out = []
    for expr in prev:
        for op in operators.operators:
            out.append(sp.expand(op.apply(expr)))
    return _flatten_unique(out)


def differential_invariants_scalar_up_to_order(
    field: VectorFieldKD, Phi: sp.Expr, u: sp.Symbol, max_order: int = 3
) -> DifferentialInvariantTowerResult:
    if max_order < 0:
        raise ValueError("max_order must be nonnegative.")
    dist = DistributionKD(field.vars, (field,))
    ops = invariant_differentiation_operators(dist)
    dec = _decompose_affine_u_action(Phi, field.vars, u)
    if dec is None:
        raise ValueError("Phi must be affine in u with constant coefficients.")
    a, beta = dec
    if len(ops.transverse_variables) != 1:
        raise ValueError("Single-generator characteristic chart expected.")
    I0 = _dependent_invariant(u, ops.transverse_variables[0], a, beta)
    invariants_by_order = {0: (I0,)}
    current = (I0,)
    for order in range(1, max_order + 1):
        current = _next_order_invariant_derivatives(ops, current)
        invariants_by_order[order] = current
    return DifferentialInvariantTowerResult(
        base_invariants=ops.invariant_variables,
        dependent_invariant=I0,
        operators=ops,
        invariants_by_order=invariants_by_order,
        method=f"{ops.method}_tower_order_{max_order}",
    )


def differential_invariants_commuting_distribution_scalar_up_to_order(
    distribution: DistributionKD,
    u: sp.Symbol,
    a_list=None,
    beta_list=None,
    max_order: int = 3,
) -> DifferentialInvariantTowerResult:
    if not distribution.is_commuting():
        raise ValueError(
            "Distribution must commute for this restricted multi-generator differential invariant engine."
        )
    r = distribution.size
    if a_list is None:
        a_list = [0] * r
    if beta_list is None:
        beta_list = [0] * r
    if len(a_list) != r or len(beta_list) != r:
        raise ValueError("a_list and beta_list must match the number of generators.")
    ops = invariant_differentiation_operators(distribution)
    I0 = _dependent_invariant_multi(u, ops.transverse_variables, a_list, beta_list)
    invariants_by_order = {0: (I0,)}
    current = (I0,)
    for order in range(1, max_order + 1):
        current = _next_order_invariant_derivatives(ops, current)
        invariants_by_order[order] = current
    return DifferentialInvariantTowerResult(
        base_invariants=ops.invariant_variables,
        dependent_invariant=I0,
        operators=ops,
        invariants_by_order=invariants_by_order,
        method=f"{ops.method}_tower_order_{max_order}",
    )
