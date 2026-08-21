from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .frobenius import local_frobenius_chart
from .geometry import CharacteristicCoordinatesResult, DistributionKD, VectorFieldKD
from .symbolic_algebra_helpers import enumerate_multiindices, multiindex_sum


@dataclass
class ReducedJetSpaceKD:
    invariant_symbols: tuple[sp.Symbol, ...]
    max_order: int
    dep_name: str = "w"

    def __post_init__(self):
        self.invariant_symbols = tuple(self.invariant_symbols)
        self.r = len(self.invariant_symbols)
        self._cache = {}
        for M in enumerate_multiindices(self.r, self.max_order):
            self._cache[M] = sp.Symbol(self._coord_name(M))

    def _coord_name(self, M):
        if multiindex_sum(M) == 0:
            return self.dep_name
        parts = [self.dep_name]
        for i, p in enumerate(M):
            if p == 0:
                continue
            name = str(self.invariant_symbols[i])
            parts.append(name if p == 1 else f"{name}{p}")
        return "_".join(parts)

    def coord(self, M):
        return self._cache[tuple(M)]

    @property
    def u(self):
        return self.coord((0,) * self.r if self.r else ())


@dataclass
class InvariantJetBundle:
    chart: CharacteristicCoordinatesResult
    reduced_jet: ReducedJetSpaceKD
    invariant_diff_ops: tuple[VectorFieldKD, ...]
    method: str

    def apply_invariant_derivative(self, index: int, expr: sp.Expr) -> sp.Expr:
        return self.invariant_diff_ops[index].apply(expr)

    def pullback_reduced_jet_symbol(self, M):
        """Formal pullback of the reduced jet symbol w_M.

        In invariant coordinates, D_{z^alpha} acts on reduced jet symbols by
        increasing the corresponding multi-index. Since the reduced field is a
        formal dependent variable, we expose this action symbolically rather than
        differentiating an ambient x-expression.
        """
        return self.reduced_jet.coord(tuple(M))


def build_invariant_jet_bundle_from_chart(
    distribution: DistributionKD,
    chart: CharacteristicCoordinatesResult | None = None,
    dep_name: str = "w",
    max_order: int = 2,
) -> InvariantJetBundle:
    if chart is None:
        chart = local_frobenius_chart(distribution)

    invariants = tuple(sp.Symbol(f"z{i + 1}", real=True) for i in range(len(chart.invariants)))
    reduced_jet = ReducedJetSpaceKD(invariants, max_order=max_order, dep_name=dep_name)

    # Differential operators D_{z^alpha} in x-coordinates from the Jacobian inverse.
    coords = chart.invariants + chart.transverse
    xs = distribution.vars
    if len(chart.invariants) == 0:
        ops = tuple()
    else:
        J = sp.Matrix([[sp.diff(c, v) for v in xs] for c in coords])
        try:
            Jinv = sp.simplify(J.inv())
        except Exception:
            Jinv = J.inv()
        ops = []
        for j in range(len(chart.invariants)):
            coeffs = tuple(sp.expand(Jinv[i, j]) for i in range(len(xs)))
            ops.append(VectorFieldKD(xs, coeffs))
        ops = tuple(ops)

    return InvariantJetBundle(
        chart=chart,
        reduced_jet=reduced_jet,
        invariant_diff_ops=ops,
        method=f"{chart.method}_invariant_jet",
    )
