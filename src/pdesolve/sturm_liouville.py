from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class SturmLiouvilleProblem:
    variable: sp.Symbol
    function: sp.Expr
    p: sp.Expr
    q: sp.Expr
    weight: sp.Expr
    interval: tuple[sp.Expr, sp.Expr]
    left_boundary: str = "dirichlet"
    right_boundary: str = "dirichlet"

    def differential_equation(self, eigenvalue=None):
        lam = eigenvalue or sp.Symbol("lambda", real=True)
        x, X = self.variable, self.function
        return sp.Eq(-sp.diff(self.p * sp.diff(X, x), x) + self.q * X, lam * self.weight * X)


@dataclass(frozen=True)
class SturmLiouvilleSpectrum:
    problem: SturmLiouvilleProblem
    index: sp.Symbol
    eigenvalues: sp.Expr
    eigenfunctions: sp.Expr
    norm_squared: sp.Expr
    orthogonality_weight: sp.Expr
    includes_zero_mode: bool = False

    def coefficient(self, profile):
        x = self.problem.variable
        a, b = self.problem.interval
        return sp.simplify(
            sp.Integral(self.orthogonality_weight * profile * self.eigenfunctions, (x, a, b))
            / self.norm_squared
        )


def solve_regular_constant_sturm_liouville(
    problem: SturmLiouvilleProblem,
) -> SturmLiouvilleSpectrum:
    """Solve the common constant-coefficient regular SL problem on a finite interval.

    Supports homogeneous Dirichlet/Dirichlet and Neumann/Neumann boundaries.
    """
    x = problem.variable
    a, b = map(sp.sympify, problem.interval)
    L = sp.simplify(b - a)
    p, q, w = map(sp.sympify, (problem.p, problem.q, problem.weight))
    if any(v.has(x) for v in (p, q, w)):
        raise NotImplementedError(
            "first-class SL solver currently requires constant p, q, and weight"
        )
    n = sp.Symbol("n", integer=True, positive=True)
    left, right = problem.left_boundary.lower(), problem.right_boundary.lower()
    if (left, right) == ("dirichlet", "dirichlet"):
        k = sp.pi * n / L
        phi = sp.sin(k * (x - a))
        lam = sp.simplify((p * k**2 + q) / w)
        norm = sp.simplify(w * L / 2)
        return SturmLiouvilleSpectrum(problem, n, lam, phi, norm, w, False)
    if (left, right) == ("neumann", "neumann"):
        n0 = sp.Symbol("n", integer=True, nonnegative=True)
        k = sp.pi * n0 / L
        phi = sp.cos(k * (x - a))
        lam = sp.simplify((p * k**2 + q) / w)
        norm = sp.Piecewise((w * L, sp.Eq(n0, 0)), (w * L / 2, True))
        return SturmLiouvilleSpectrum(problem, n0, lam, phi, norm, w, True)
    raise NotImplementedError(
        "supported boundary pairs are homogeneous Dirichlet/Dirichlet and Neumann/Neumann"
    )


__all__ = [
    "SturmLiouvilleProblem",
    "SturmLiouvilleSpectrum",
    "solve_regular_constant_sturm_liouville",
]
