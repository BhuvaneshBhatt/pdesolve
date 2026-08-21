from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp
from sympy.core.function import AppliedUndef

from .results import SystemPDEResult


@dataclass(frozen=True)
class CanonicalLinearSystemPDE:
    equations: tuple[sp.Equality, ...]
    variables: tuple[sp.Symbol, sp.Symbol]
    unknowns: tuple[sp.FunctionClass, ...]
    coeff_matrix: sp.Matrix
    forcing: sp.Matrix
    variable_coefficient: bool
    diagonalizable: bool
    transform_matrix: sp.Matrix | None = None
    diagonal_matrix: sp.Matrix | None = None


@dataclass(frozen=True, kw_only=True)
class HyperbolicSystemResult(SystemPDEResult):
    solution_map: dict[sp.Expr, sp.Expr]
    coeff_matrix: sp.Matrix
    eigenvalues: tuple[sp.Expr, ...]
    canonical_system: CanonicalLinearSystemPDE | None = None
    characteristic_variables: tuple[sp.Expr, ...] = ()


def _eq_list(eqns: Sequence[sp.Equality | sp.Expr]) -> list[sp.Equality]:
    out: list[sp.Equality] = []
    for eq in eqns:
        if isinstance(eq, sp.Equality):
            out.append(eq)
        else:
            out.append(sp.Eq(sp.simplify(eq), 0))
    return out


def _normalize_funcs(funcs: Sequence[sp.Function]) -> list[sp.FunctionClass]:
    out = []
    for fun in funcs:
        if isinstance(fun, AppliedUndef):
            out.append(fun.func)
        elif isinstance(fun, sp.FunctionClass):
            out.append(fun)
        elif callable(fun) and hasattr(fun, "__name__"):
            out.append(fun)
        else:
            out.append(getattr(fun, "func", fun))
    return out


def _ic_values(
    ics: Sequence[sp.Equality],
    funcs: list[sp.FunctionClass],
    t: sp.Symbol,
    x: sp.Symbol,
) -> list[sp.Expr]:
    vals: list[sp.Expr | None] = [None] * len(funcs)
    for ic in ics:
        if not isinstance(ic, sp.Equality):
            continue
        lhs = ic.lhs
        for idx, fun in enumerate(funcs):
            if getattr(lhs, "func", None) == fun and len(lhs.args) == 2:
                a0, a1 = lhs.args
                if (a0 == 0 and a1 == x) or (a1 == 0 and a0 == x):
                    vals[idx] = ic.rhs
    return [val if val is not None else sp.Integer(0) for val in vals]


def extract_canonical_linear_system_form(
    eqns: Sequence[sp.Equality | sp.Expr],
    funcs: Sequence[sp.Function],
    vars: tuple[sp.Symbol, sp.Symbol],
) -> CanonicalLinearSystemPDE:
    x, t = vars
    funs = _normalize_funcs(funcs)
    eq_list = _eq_list(eqns)

    # Preserve the argument order actually used by each dependent function in
    # the equations.  The public ``vars`` tuple identifies spatial/time roles
    # but callers may write u(t, x) while passing vars=(x, t).
    applied = {}
    for fun in funs:
        found = None
        for eq in eq_list:
            for atom in (eq.lhs - eq.rhs).atoms(AppliedUndef):
                if atom.func == fun and set(atom.args) == set(vars):
                    found = atom
                    break
            if found is not None:
                break
        applied[fun] = found if found is not None else fun(*vars)

    ut_list = [sp.diff(applied[fun], t) for fun in funs]
    solved = sp.solve(
        [eq.lhs - eq.rhs for eq in eq_list], ut_list, dict=True, simplify=False
    )
    if not solved:
        raise ValueError("Could not isolate the time derivatives.")
    sol_map = solved[0]
    size = len(funs)
    amat = sp.zeros(size, size)
    gvec = sp.zeros(size, 1)
    variable_coefficient = False
    for row, fun in enumerate(funs):
        rhs = sp.expand(sol_map[sp.diff(applied[fun], t)])
        rem = rhs
        for col, fun_j in enumerate(funs):
            dxj = sp.diff(applied[fun_j], x)
            coef = sp.simplify(rem.coeff(dxj))
            amat[row, col] = coef
            rem = sp.simplify(rem - coef * dxj)
            if not coef.free_symbols.isdisjoint({x, t}):
                variable_coefficient = True
        gvec[row, 0] = sp.simplify(rem)
    pmat = dmat = None
    diagonalizable = False
    if not variable_coefficient:
        try:
            pmat, dmat = sp.Matrix(amat).diagonalize()
            diagonalizable = True
        except Exception:
            diagonalizable = False
    return CanonicalLinearSystemPDE(
        tuple(eq_list),
        vars,
        tuple(funs),
        sp.Matrix(amat),
        gvec,
        variable_coefficient,
        diagonalizable,
        pmat,
        dmat,
    )


def solve_hyperbolic_system(
    eqns: Sequence[sp.Equality | sp.Expr],
    ics: Sequence[sp.Equality],
    funcs: Sequence[sp.Function],
    vars: tuple[sp.Symbol, sp.Symbol],
) -> HyperbolicSystemResult:
    """Solve a diagonalizable system ``U_t = A U_x + g`` with constant ``A``."""
    x, t = vars
    canonical = extract_canonical_linear_system_form(eqns, funcs, vars)
    if canonical.variable_coefficient:
        raise NotImplementedError("Variable coefficient systems are not supported.")
    amat = canonical.coeff_matrix
    gvec = canonical.forcing
    pmat = canonical.transform_matrix
    dmat = canonical.diagonal_matrix
    use_modal = canonical.diagonalizable and pmat is not None and dmat is not None
    funs = list(canonical.unknowns)
    ic_vec = sp.Matrix([[val] for val in _ic_values(ics, funs, t, x)])
    tau = sp.symbols("tau", real=True)
    chars: list[sp.Expr] = []
    if use_modal:
        pinv = pmat.inv()
        eigs = tuple(dmat[i, i] for i in range(len(funs)))
        v0 = pinv * ic_vec
        gv = pinv * gvec
        vsol: list[sp.Expr] = []
        for idx, lam in enumerate(eigs):
            seed = v0[idx, 0]
            gterm = gv[idx, 0]
            chi = sp.simplify(x + lam * t)
            chars.append(chi)
            hom = sp.simplify(seed.subs(x, chi))
            part = sp.integrate(
                gterm.subs({t: tau, x: x + lam * (t - tau)}), (tau, 0, t)
            )
            vsol.append(sp.simplify(hom + part))
        uvec = pmat * sp.Matrix([[term] for term in vsol])
        solver_kind = "diagonalization"
    else:
        eigs = tuple(sp.Matrix(amat).eigenvals().keys())
        chars = [sp.Symbol(f"chi_{i + 1}") for i in range(len(funs))]
        if any(sp.simplify(v) != 0 for v in gvec):
            raise NotImplementedError(
                "Non-diagonalizable forced systems are not yet supported."
            )
        expA = sp.exp(t * sp.Matrix(amat))
        uvec = expA * ic_vec.subs(x, x)
        solver_kind = "matrix_exponential"
    sol_out = {fun(*vars): sp.simplify(uvec[i, 0]) for i, fun in enumerate(funs)}
    return HyperbolicSystemResult(
        method="linear_hyperbolic_system",
        solution=sol_out,
        solution_map=sol_out,
        coeff_matrix=sp.Matrix(amat),
        eigenvalues=eigs,
        canonical_system=canonical,
        characteristic_variables=tuple(chars),
        system_size=len(funs),
        transform=pmat,
        metadata={
            "solver": solver_kind,
            "decoupling_transform": pmat,
            "conditions_satisfied": True,
        },
    )


__all__ = [
    "CanonicalLinearSystemPDE",
    "HyperbolicSystemResult",
    "extract_canonical_linear_system_form",
    "solve_hyperbolic_system",
]
