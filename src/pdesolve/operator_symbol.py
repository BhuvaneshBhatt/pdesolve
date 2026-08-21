from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import factorial

import sympy as sp


@dataclass(frozen=True)
class SymbolLowestTerm:
    total_degree: int
    multiindex: tuple[int, ...]
    coefficient: sp.Expr
    monomial: sp.Expr


@dataclass(frozen=True)
class ShiftedOperatorSymbol:
    variables: tuple[sp.Symbol, ...]
    shift: tuple[sp.Expr, ...]
    shifted_expr: sp.Expr
    lowest_term: SymbolLowestTerm | None

    @property
    def resonance_multiplicity(self) -> int:
        return 0 if self.lowest_term is None else int(self.lowest_term.total_degree)

    @property
    def is_resonant(self) -> bool:
        return self.resonance_multiplicity > 0

    def lifted_polynomial_for_constant_forcing(
        self, indep_vars: Iterable[sp.Symbol]
    ) -> sp.Expr | None:
        if self.lowest_term is None:
            return None
        indep_vars = tuple(indep_vars)
        alpha = self.lowest_term.multiindex
        coeff = sp.expand(self.lowest_term.coefficient)
        if coeff == 0:
            return None
        denom = coeff
        for a in alpha:
            denom *= factorial(int(a))
        mono = sp.Integer(1)
        for var, power in zip(indep_vars, alpha, strict=True):
            mono *= var ** int(power)
        return sp.expand(mono / denom)


@dataclass(frozen=True)
class ConstantCoefficientSymbol:
    variables: tuple[sp.Symbol, ...]
    expr: sp.Expr

    def evaluate(self, vector: Iterable[sp.Expr]) -> sp.Expr:
        vector = tuple(vector)
        subs = {v: value for v, value in zip(self.variables, vector, strict=True)}
        return sp.expand(self.expr.subs(subs))

    def shift(self, vector: Iterable[sp.Expr]) -> ShiftedOperatorSymbol:
        vector = tuple(vector)
        zvars = sp.symbols(f"z0:{len(self.variables)}")
        subs = {
            var: sp.expand(shift + z)
            for var, shift, z in zip(self.variables, vector, zvars, strict=True)
        }
        shifted = sp.expand(self.expr.subs(subs))
        lowest = _lowest_nonzero_term(shifted, zvars)
        return ShiftedOperatorSymbol(tuple(zvars), vector, shifted, lowest)


def _lowest_nonzero_term(expr: sp.Expr, vars_: tuple[sp.Symbol, ...]) -> SymbolLowestTerm | None:
    expr = sp.expand(expr)
    if expr == 0:
        return None
    poly = sp.Poly(expr, *vars_, domain="EX")
    terms = []
    for monom, coeff in poly.terms():
        coeff = sp.expand(coeff)
        if coeff == 0:
            continue
        total_degree = int(sum(monom))
        mono_expr = sp.Integer(1)
        for var, power in zip(vars_, monom, strict=True):
            mono_expr *= var ** int(power)
        terms.append((total_degree, tuple(int(i) for i in monom), coeff, mono_expr))
    if not terms:
        return None
    total_degree, multiindex, coeff, mono_expr = min(terms, key=lambda item: (item[0], item[1]))
    return SymbolLowestTerm(total_degree, multiindex, coeff, mono_expr)


__all__ = [
    "SymbolLowestTerm",
    "ShiftedOperatorSymbol",
    "ConstantCoefficientSymbol",
]
