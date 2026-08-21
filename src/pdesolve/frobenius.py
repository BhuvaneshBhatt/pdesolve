from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp

from .canonical import canonicalize_coordinate_chart
from .charts import ChartAtlasKD, build_chart_atlas
from .coordinates import (
    _common_affine_coordinates_distribution,
    _common_affine_data,
    pdesolve,
)
from .diagnostics import explain_distribution, failure, local_chart_conditions_from_coords
from .geometry import (
    CharacteristicCoordinatesResult,
    DistributionKD,
    VectorFieldKD,
    distribution_closure,
)
from .symbolic_algebra_helpers import expr_complexity


@dataclass
class FrobeniusEngineDiagnostics:
    rank: int | None
    commuting: bool
    closed: bool
    affine: bool
    chosen_method: str | None
    structure_data: tuple | None
    validity_conditions: tuple[sp.Expr, ...]


@dataclass
class FrobeniusChartResult:
    invariants: tuple[sp.Expr, ...]
    transverse: tuple[sp.Expr, ...]
    jacobian: sp.Expr
    method: str
    validity_conditions: tuple[sp.Expr, ...]
    transformed_fields: tuple[VectorFieldKD, ...]
    closure_data: tuple | None


def _transform_fields_to_coordinates(
    distribution: DistributionKD, coords: Sequence[sp.Expr]
) -> tuple[VectorFieldKD, ...]:
    q = tuple(sp.Symbol(f"q{i + 1}", real=True) for i in range(len(coords)))
    transformed = []
    for X in distribution.fields:
        coeffs = [sp.expand(X.apply(expr)) for expr in coords]
        transformed.append(VectorFieldKD(q, tuple(coeffs)))
    return tuple(transformed)


def _triangular_score(fields: Sequence[VectorFieldKD]) -> int:
    score = 0
    for i, X in enumerate(fields):
        for j, c in enumerate(X.coeffs):
            if j < i and sp.simplify(c) != 0:
                score += 5
            elif sp.simplify(c) != 0:
                score += expr_complexity(c)
    return score


def _adapt_field_order(transformed_fields: Sequence[VectorFieldKD]) -> tuple[VectorFieldKD, ...]:
    # Heuristic: order fields by first nonzero coordinate index, then sparsity.
    def keyfun(X: VectorFieldKD):
        first = next((i for i, c in enumerate(X.coeffs) if sp.simplify(c) != 0), len(X.coeffs))
        nz = sum(1 for c in X.coeffs if sp.simplify(c) != 0)
        cx = sum(expr_complexity(c) for c in X.coeffs)
        return (first, nz, cx)

    return tuple(sorted(transformed_fields, key=keyfun))


def local_frobenius_chart(distribution: DistributionKD) -> FrobeniusChartResult:
    """Restricted local Frobenius/characteristic engine with ranked chart selection."""
    closure = distribution_closure(distribution)
    atlas = restricted_local_frobenius_atlas(distribution)
    best = atlas.best()
    if best is None:
        raise NotImplementedError(
            "Restricted local Frobenius engine could not construct a valid local chart for this distribution."
        )
    coords = best.chart.invariants + best.chart.transverse
    adapted = adapted_basis_in_chart(distribution, coords)
    transformed = adapted.adapted_fields
    method = best.chart.method
    if (
        closure.closed
        and adapted.triangular_score == 0
        and distribution.size > 0
        and method not in ("involutive_affine_full_rank_identity_chart",)
    ):
        method = method + "_rectified" if "rectified" not in method else method
    return FrobeniusChartResult(
        invariants=best.chart.invariants,
        transverse=best.chart.transverse,
        jacobian=best.chart.jacobian,
        method=method,
        validity_conditions=best.local_conditions,
        transformed_fields=transformed,
        closure_data=closure.structure_matrix_data,
    )


def local_frobenius_chart_with_diagnostics(distribution: DistributionKD):
    diag = distribution.diagnostics()
    closure = distribution_closure(distribution)
    chosen = None
    validity = tuple()
    result = None
    try:
        result = local_frobenius_chart(distribution)
        chosen = result.method
        validity = result.validity_conditions
    except Exception:
        result = None
    return result, FrobeniusEngineDiagnostics(
        rank=diag.rank,
        commuting=diag.commuting,
        closed=closure.closed,
        affine=diag.affine,
        chosen_method=chosen,
        structure_data=closure.structure_matrix_data,
        validity_conditions=validity,
    )


def _chart_candidate_from_result(
    distribution: DistributionKD, result: CharacteristicCoordinatesResult
) -> CharacteristicCoordinatesResult:
    result = canonicalize_coordinate_chart(result, distribution.vars)
    validity = tuple(result.validity_conditions) + local_chart_conditions_from_coords(
        distribution.vars, result.invariants + result.transverse
    )
    validity = tuple(dict.fromkeys(sp.simplify(v) for v in validity))
    return CharacteristicCoordinatesResult(
        result.invariants, result.transverse, result.jacobian, result.method, validity
    )


def restricted_local_frobenius_atlas(distribution: DistributionKD) -> ChartAtlasKD:
    """Build a small ranked atlas of candidate local charts for the restricted engine.

    This improves chart management by keeping alternative local charts and ranking
    them by simplicity/validity conditions rather than committing immediately to one.
    """
    diag = distribution.diagnostics()
    closure = distribution_closure(distribution)
    candidates = []

    try:
        cand = pdesolve(distribution)
        candidates.append(_chart_candidate_from_result(distribution, cand))
    except Exception:
        pass

    if diag.affine and closure.closed:
        if diag.rank == distribution.dimension:
            candidates.append(
                _chart_candidate_from_result(
                    distribution,
                    CharacteristicCoordinatesResult(
                        tuple(),
                        distribution.vars,
                        sp.Integer(1),
                        "involutive_affine_full_rank_identity_chart",
                        tuple(),
                    ),
                )
            )

        data = _common_affine_data(distribution)
        if data is not None:
            Ms, bs = data
            coords = _common_affine_coordinates_distribution(
                Ms, bs, list(distribution.vars), distribution.size
            )
            if coords is not None:
                coords = canonicalize_coordinate_chart(coords, distribution.vars)
                candidates.append(
                    _chart_candidate_from_result(
                        distribution,
                        CharacteristicCoordinatesResult(
                            coords.invariants,
                            coords.transverse,
                            coords.jacobian,
                            "involutive_affine_constant_derivative_coords",
                            coords.validity_conditions,
                        ),
                    )
                )

    # adapted chart candidate from identity coordinates for involutive cases
    if closure.closed and distribution.size > 0:
        coords = CharacteristicCoordinatesResult(
            tuple(), distribution.vars, sp.Integer(1), "identity_chart_candidate", tuple()
        )
        candidates.append(_chart_candidate_from_result(distribution, coords))

    if not candidates:
        return build_chart_atlas(distribution.vars, tuple())

    atlas = build_chart_atlas(distribution.vars, candidates)
    return atlas


def local_frobenius_chart_explain(distribution: DistributionKD) -> dict:
    """Explain, in a structured form, why the restricted engine succeeded or failed."""
    report = explain_distribution(distribution)
    atlas = restricted_local_frobenius_atlas(distribution)
    if atlas.charts:
        best = atlas.best()
        return {
            "success": True,
            "distribution": report,
            "best_method": best.source,
            "chart_count": len(atlas.charts),
            "local_conditions": best.local_conditions,
            "score": best.score,
        }
    reasons = list(report.reasons)
    if not report.closure_closed:
        reasons.append("distribution_not_involutive")
    return {
        "success": False,
        "distribution": report,
        "failures": (
            failure("frobenius", "no_valid_chart_found", *reasons, conditions=report.conditions),
        ),
    }


@dataclass
class AdaptedBasisResult:
    original_fields: tuple[VectorFieldKD, ...]
    adapted_fields: tuple[VectorFieldKD, ...]
    transformation_matrix: sp.Matrix
    coordinate_labels: tuple[sp.Symbol, ...]
    triangular_score: int


def adapted_basis_in_chart(
    distribution: DistributionKD, coords: Sequence[sp.Expr]
) -> AdaptedBasisResult:
    """
    Build a more systematic adapted basis in the provided coordinate chart.

    Strategy:
      1. transform the fields to the new coordinates;
      2. greedily choose pivot coordinates;
      3. perform symbolic Gaussian elimination on the field coefficient matrix
         to obtain a basis that is as triangular as possible.

    This is still local and heuristic, but more systematic than simple sorting.
    """
    transformed = list(_transform_fields_to_coordinates(distribution, coords))
    q = (
        transformed[0].vars
        if transformed
        else tuple(sp.Symbol(f"q{i + 1}") for i in range(len(coords)))
    )
    r = len(transformed)
    n = len(q)
    if r == 0:
        return AdaptedBasisResult(tuple(), tuple(), sp.eye(0), q, 0)

    A = sp.Matrix([[sp.expand(X.coeffs[j]) for j in range(n)] for X in transformed])
    T = sp.eye(r)

    pivot_cols = []
    remaining_cols = list(range(n))
    for i in range(r):
        # Choose a pivot column where some row i..r-1 has nonzero entry, preferring low complexity.
        best = None
        for col in remaining_cols:
            for row in range(i, r):
                val = sp.expand(A[row, col])
                if sp.simplify(val) != 0:
                    cand = (expr_complexity(val), col, row)
                    if best is None or cand < best:
                        best = cand
        if best is None:
            break
        _, col, row = best
        if row != i:
            A.row_swap(i, row)
            T.row_swap(i, row)
        pivot = sp.expand(A[i, col])
        # Eliminate other rows in this pivot column.
        for row2 in range(r):
            if row2 == i:
                continue
            val = sp.expand(A[row2, col])
            if sp.simplify(val) == 0:
                continue
            factor = sp.simplify(val / pivot)
            A.row_op(row2, lambda v, j, factor=factor, row=i: sp.expand(v - factor * A[row, j]))
            T.row_op(row2, lambda v, j, factor=factor, row=i: sp.expand(v - factor * T[row, j]))
        pivot_cols.append(col)
        remaining_cols.remove(col)

    adapted_fields = []
    for i in range(r):
        coeffs = tuple(sp.expand(A[i, j]) for j in range(n))
        adapted_fields.append(VectorFieldKD(q, coeffs))

    return AdaptedBasisResult(
        original_fields=tuple(transformed),
        adapted_fields=tuple(adapted_fields),
        transformation_matrix=sp.Matrix(T),
        coordinate_labels=q,
        triangular_score=_triangular_score(adapted_fields),
    )


def local_chart_conditions(
    distribution: DistributionKD, coords: Sequence[sp.Expr]
) -> tuple[sp.Expr, ...]:
    """Heuristic local validity conditions for a coordinate chart."""
    conds = list(local_chart_conditions_from_coords(vars, tuple(coords)))
    diag = distribution.diagnostics()
    if diag.minor_columns is not None and len(diag.minor_columns) == distribution.size:
        A = distribution.coefficient_matrix()
        try:
            minor = sp.expand(A[:, diag.minor_columns].det())
            if sp.simplify(minor) != 0:
                conds.append(sp.Ne(minor, 0))
        except Exception:
            pass
    return tuple(dict.fromkeys(sp.simplify(c) for c in conds))
