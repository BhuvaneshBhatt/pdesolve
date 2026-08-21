from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import sympy as sp
from sympy.polys.monomials import itermonomials

from .verify import verify_solution_with_conditions
from .operator_symbol import ConstantCoefficientSymbol


@dataclass(frozen=True)
class LinearConstantCoefficientPDE:
    indep_vars: tuple[sp.Symbol, ...]
    dep_function: sp.Expr
    operator_terms: dict[tuple[int, ...], sp.Expr]
    rhs: sp.Expr
    operator_polynomial: sp.Expr
    normalized_equation: sp.Equality
    order: int


@dataclass(frozen=True)
class ParticularSolutionResult:
    method: str
    solution: sp.Expr
    details: dict[str, Any]


@dataclass(frozen=True)
class ConstantCoefficientOperatorFactor:
    polynomial: sp.Expr
    multiplicity: int
    total_degree: int
    coefficients: tuple[sp.Expr, ...]
    constant_term: sp.Expr


@dataclass(frozen=True)
class ConstantCoefficientFactorSolution2D:
    multiplicity: int
    a: sp.Expr
    b: sp.Expr
    c: sp.Expr
    invariant: sp.Expr
    transverse: sp.Expr
    arbitrary_functions: tuple[Any, ...]
    expression: sp.Expr


@dataclass(frozen=True)
class ConstantCoefficientHomogeneousFamily:
    factor: ConstantCoefficientOperatorFactor
    method: str
    expression: sp.Expr
    generators: tuple[Any, ...]
    invariant: Any
    invariants: tuple[sp.Expr, ...]
    transverse: sp.Expr | None


@dataclass(frozen=True)
class PDEGeneralSolutionResult:
    method: str
    solution: sp.Equality
    details: dict[str, Any]


@dataclass(frozen=True)
class ConstantCoefficientOperatorProfile:
    pde: LinearConstantCoefficientPDE
    forcing_family: str
    factors: tuple[ConstantCoefficientOperatorFactor, ...]
    symbol: ConstantCoefficientSymbol


def _method_family(method: str) -> str:
    if "fit" in method or "fitted" in method:
        return "condition_fitting"
    if "homogeneous" in method:
        return "homogeneous"
    if "trig" in method or "hyperbolic" in method:
        return "trig_hyperbolic_exponential_amplitude"
    if "exponential_amplitude" in method:
        return "exponential_amplitude"
    if "exponential" in method:
        return "exponential"
    if "polynomial" in method:
        return "polynomial"
    if "pdsolve" in method:
        return "fallback_pdsolve"
    if "zero_rhs" in method:
        return "zero_rhs"
    return "constant_coefficient"


def _with_method_metadata(
    method: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    merged = dict(details or {})
    merged.setdefault("solver_family", "constant_coefficient")
    merged.setdefault("method_family", _method_family(method))
    merged.setdefault("selected_method", method)
    return merged


def _particular_result(
    method: str, solution: sp.Expr, details: dict[str, Any] | None = None
) -> ParticularSolutionResult:
    return ParticularSolutionResult(
        method, sp.expand(solution), _with_method_metadata(method, details)
    )


def _general_solution_result(
    method: str, solution: sp.Equality, details: dict[str, Any] | None = None
) -> PDEGeneralSolutionResult:
    return PDEGeneralSolutionResult(
        method, solution, _with_method_metadata(method, details)
    )


def _impl():
    from . import classical_methods as impl

    return impl


def _dep_and_vars(dep_expr_or_func, indep_vars=None):
    return _impl()._dep_and_vars(dep_expr_or_func, indep_vars)


def _as_zero_expr(eq_or_expr):
    return _impl()._as_zero_expr(eq_or_expr)


def canonicalize_pde_problem(eq_or_expr, dep_expr_or_func, indep_vars=None):
    impl = _impl()
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    can = impl.canonicalize_pde_problem(eq_or_expr, uexpr, vars_)
    return can.equation if hasattr(can, "equation") else can


def detect_linear_constant_coefficient_pde(
    eq_or_expr, dep_expr_or_func, indep_vars=None
):
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    zero = _as_zero_expr(eq_or_expr)
    vars_ = tuple(vars_)
    k = len(vars_)

    deriv_nodes = []
    seen = set()
    for node in sp.preorder_traversal(zero):
        if isinstance(node, sp.Derivative) and node.expr == uexpr:
            counts = [0] * k
            for v, c in node.variable_count:
                if v not in vars_:
                    raise ValueError("Derivative variable not among indep_vars.")
                counts[vars_.index(v)] += c
            mi = tuple(counts)
            if mi not in seen:
                seen.add(mi)
                deriv_nodes.append((node, mi))
    deriv_nodes.sort(key=lambda item: sum(item[1]))

    U0 = sp.Symbol("U0_cc")
    replacements = {uexpr: U0}
    placeholder_by_mi = {tuple([0] * k): U0}
    for _, mi in deriv_nodes:
        sym = sp.Symbol("D_" + "_".join(map(str, mi)))
        placeholder_by_mi[mi] = sym
    for node, mi in deriv_nodes:
        replacements[node] = placeholder_by_mi[mi]

    expr = sp.expand(zero.xreplace(replacements))
    linear_vars = [placeholder_by_mi[tuple([0] * k)]] + [
        placeholder_by_mi[mi] for _, mi in deriv_nodes
    ]
    poly = sp.Poly(expr, *linear_vars, domain="EX")
    if poly.total_degree() > 1:
        raise ValueError("PDE is not linear in u and its derivatives.")

    rhs = sp.expand(-(expr.subs({v: 0 for v in linear_vars})))
    operator_terms = {}
    coeff_u = sp.expand(sp.diff(expr, placeholder_by_mi[tuple([0] * k)]))
    if coeff_u != 0:
        operator_terms[tuple([0] * k)] = coeff_u
    for _, mi in deriv_nodes:
        coeff = sp.expand(sp.diff(expr, placeholder_by_mi[mi]))
        if coeff != 0:
            operator_terms[mi] = coeff

    if any(
        not coeff.free_symbols.isdisjoint(set(vars_) | {uexpr})
        for coeff in operator_terms.values()
    ):
        raise ValueError("PDE is not constant-coefficient linear in the operator part.")

    msyms = sp.symbols(f"m0:{k}")
    op_poly = sp.Integer(0)
    for mi, coeff in operator_terms.items():
        mon = sp.Integer(1)
        for i, a in enumerate(mi):
            mon *= msyms[i] ** a
        op_poly += coeff * mon
    op_poly = sp.expand(op_poly)

    normalized = sp.Eq(
        sp.expand(
            sum(
                operator_terms.get(mi, 0)
                * sp.diff(uexpr, *sum(([vars_[i]] * mi[i] for i in range(k)), []))
                if sum(mi) > 0
                else operator_terms[mi] * uexpr
                for mi in operator_terms
            )
            - rhs
        ),
        0,
    )

    return LinearConstantCoefficientPDE(
        indep_vars=vars_,
        dep_function=uexpr,
        operator_terms=operator_terms,
        rhs=sp.expand(rhs),
        operator_polynomial=op_poly,
        normalized_equation=normalized,
        order=max((sum(mi) for mi in operator_terms), default=0),
    )


def _cc_operator_apply_from_terms(operator_terms, expr, vars_):
    total = sp.Integer(0)
    for mi, coeff in operator_terms.items():
        if sum(mi) == 0:
            total += coeff * expr
        else:
            deriv_args = []
            for i, a in enumerate(mi):
                if a:
                    deriv_args.extend([vars_[i]] * a)
            total += coeff * sp.diff(expr, *deriv_args)
    return sp.expand(total)


def _make_operator_symbol(
    ccpde: LinearConstantCoefficientPDE,
) -> ConstantCoefficientSymbol:
    msyms = sp.symbols(f"m0:{len(ccpde.indep_vars)}")
    return ConstantCoefficientSymbol(tuple(msyms), sp.expand(ccpde.operator_polynomial))


def _operator_symbol(ccpde: LinearConstantCoefficientPDE, vector):
    return _make_operator_symbol(ccpde).evaluate(vector)


def _shifted_symbol_profile(ccpde: LinearConstantCoefficientPDE, wave_vector):
    return _make_operator_symbol(ccpde).shift(wave_vector)


def _linear_system_solve(expressions, unknowns, vars_):
    equations = []
    for expr in expressions:
        expr = sp.expand(expr)
        poly = sp.Poly(expr, *vars_, domain="EX")
        equations.extend(sp.expand(c) for c in poly.coeffs())
    if not equations:
        return {}
    A, b = sp.linear_eq_to_matrix(equations, unknowns)
    sols = sp.linsolve((A, b), *unknowns)
    if not sols:
        return None
    vec = list(next(iter(sols)))
    free_syms = set()
    for entry in vec:
        free_syms |= {s for s in entry.free_symbols if s not in set(unknowns)}
    zero_sub = {s: 0 for s in free_syms}
    return {u: sp.expand(v.subs(zero_sub)) for u, v in zip(unknowns, vec)}


def _phase_vector(phase, vars_):
    if not phase.is_polynomial(*vars_) or any(sp.degree(phase, v) > 1 for v in vars_):
        return None
    return tuple(sp.expand(sp.diff(phase, v)) for v in vars_)


def _monomials_in_vars(vars_, degree):
    return tuple(
        sorted(
            itermonomials(vars_, degree),
            key=lambda m: (sp.total_degree(m), sp.default_sort_key(m)),
        )
    )


def _solve_polynomial_ansatz(
    ccpde: LinearConstantCoefficientPDE, rhs, degree, *, prefix="coef"
):
    vars_ = ccpde.indep_vars
    coeffs = []
    ansatz = sp.Integer(0)
    for idx, mon in enumerate(_monomials_in_vars(vars_, degree)):
        c = sp.Symbol(f"{prefix}_{idx}")
        coeffs.append(c)
        ansatz += c * mon
    resid = sp.expand(
        _cc_operator_apply_from_terms(ccpde.operator_terms, ansatz, vars_) - rhs
    )
    sub = _linear_system_solve([resid], coeffs, vars_)
    if sub is None:
        raise NotImplementedError("Could not determine an ansatz solution.")
    zero_unknowns = {c: 0 for c in coeffs}
    for c in coeffs:
        sub.setdefault(c, 0)
        sub[c] = sp.expand(sub[c].subs(zero_unknowns))
    return sp.expand(ansatz.subs(sub))


def _exponential_particular(ccpde: LinearConstantCoefficientPDE, amplitude, phase):
    vars_ = ccpde.indep_vars
    kvec = _phase_vector(phase, vars_)
    if kvec is None:
        raise NotImplementedError(
            "Exponential phase must be affine in the independent variables."
        )
    symbol_profile = _shifted_symbol_profile(ccpde, kvec)
    symbol_value = sp.simplify(_operator_symbol(ccpde, kvec))
    forcing = sp.expand(amplitude * sp.exp(phase))
    stage_factors = []
    profile = factor_constant_coefficient_operator(ccpde.operator_polynomial, vars_)
    symbol_obj = _make_operator_symbol(ccpde)
    for fac in profile:
        fac_symbol = ConstantCoefficientSymbol(symbol_obj.variables, fac.polynomial)
        shifted_fac = fac_symbol.shift(kvec)
        stage_factors.append(
            {
                "factor": fac,
                "evaluated": sp.expand(fac_symbol.evaluate(kvec)),
                "shifted": shifted_fac,
                "multiplicity": shifted_fac.resonance_multiplicity,
            }
        )
    if symbol_value != 0:
        return _particular_result(
            "constant_coefficient_exponential",
            solution=sp.expand(forcing / symbol_value),
            details={
                "phase": phase,
                "wave_vector": kvec,
                "symbol_value": symbol_value,
                "resonant": False,
                "resonance_multiplicity": 0,
                "stages": tuple(stage_factors),
            },
        )
    lifted_poly = symbol_profile.lifted_polynomial_for_constant_forcing(vars_)
    if amplitude == 1 and lifted_poly is not None:
        qsol = sp.expand(lifted_poly)
        return _particular_result(
            "constant_coefficient_resonance_lifted_exponential",
            solution=sp.expand(qsol * sp.exp(phase)),
            details={
                "phase": phase,
                "wave_vector": kvec,
                "symbol_value": symbol_value,
                "resonant": True,
                "resonance_multiplicity": symbol_profile.resonance_multiplicity,
                "shifted_symbol": symbol_profile,
                "lifted_polynomial": qsol,
                "stages": tuple(stage_factors),
            },
        )
    degree = (
        max(0, sp.total_degree(amplitude, *vars_))
        + ccpde.order
        + max(1, symbol_profile.resonance_multiplicity)
    )
    qsol = _solve_polynomial_ansatz(
        LinearConstantCoefficientPDE(
            indep_vars=ccpde.indep_vars,
            dep_function=ccpde.dep_function,
            operator_terms=_shifted_operator_terms(ccpde, phase)[0],
            rhs=amplitude,
            operator_polynomial=ccpde.operator_polynomial,
            normalized_equation=ccpde.normalized_equation,
            order=ccpde.order,
        ),
        amplitude,
        degree,
        prefix="qcoef",
    )
    return _particular_result(
        "constant_coefficient_resonant_exponential",
        solution=sp.expand(qsol * sp.exp(phase)),
        details={
            "phase": phase,
            "wave_vector": kvec,
            "symbol_value": symbol_value,
            "resonant": True,
            "resonance_multiplicity": symbol_profile.resonance_multiplicity,
            "shifted_symbol": symbol_profile,
            "stages": tuple(stage_factors),
        },
    )


def _extract_exponential_term_data(term):
    term = sp.expand(term)
    exp_factors = [f for f in sp.Mul.make_args(term) if f.func is sp.exp]
    if not exp_factors:
        return None
    phase = sp.expand(sum(f.args[0] for f in exp_factors))
    amplitude = sp.expand(term / sp.Mul(*exp_factors))
    return amplitude, phase


def _decompose_rhs_to_exponential_amplitudes(rhs, vars_):
    rhs = sp.expand(sp.sympify(rhs))
    rewritten = sp.expand(rhs.rewrite(sp.exp))
    terms = []
    for term in sp.Add.make_args(rewritten):
        extracted = _extract_exponential_term_data(term)
        if extracted is None:
            return None
        amplitude, phase = extracted
        if not phase.is_polynomial(*vars_) or any(
            sp.degree(phase, v) > 1 for v in vars_
        ):
            return None
        origin = "exponential"
        raw = sp.expand(term)
        if raw.has(sp.sin, sp.cos):
            origin = "trigonometric"
        elif raw.has(sp.sinh, sp.cosh):
            origin = "hyperbolic"
        elif rhs.has(sp.sin, sp.cos):
            origin = "trigonometric"
        elif rhs.has(sp.sinh, sp.cosh):
            origin = "hyperbolic"
        terms.append(
            {
                "amplitude": sp.expand(amplitude),
                "phase": sp.expand(phase),
                "rewritten_term": raw,
                "origin": origin,
            }
        )
    return tuple(terms)


def _solve_exponential_amplitude_term(
    ccpde: LinearConstantCoefficientPDE, amplitude, phase, *, origin="exponential"
):
    if amplitude == 1:
        base = _exponential_particular(ccpde, sp.Integer(1), phase)
    elif sp.expand(amplitude).is_polynomial(*ccpde.indep_vars):
        base = _polynomial_times_phase_exponential_particular(ccpde, amplitude, phase)
    else:
        base = _exponential_particular(ccpde, amplitude, phase)
    details = dict(base.details)
    details["amplitude"] = sp.expand(amplitude)
    details["origin"] = origin
    details["engine"] = "exponential_amplitude"
    return _particular_result(base.method, base.solution, details)


def _particular_from_exponential_amplitudes(ccpde: LinearConstantCoefficientPDE, rhs):
    pieces = _decompose_rhs_to_exponential_amplitudes(rhs, ccpde.indep_vars)
    if pieces is None:
        raise NotImplementedError(
            "Unsupported exponential-amplitude forcing decomposition."
        )
    solved = [
        _solve_exponential_amplitude_term(
            ccpde, piece["amplitude"], piece["phase"], origin=piece["origin"]
        )
        for piece in pieces
    ]
    combined = sp.expand(sum(item.solution for item in solved))
    origins = tuple(piece["origin"] for piece in pieces)
    family = (
        "trig_hyperbolic_exponential_amplitude"
        if any(o in {"trigonometric", "hyperbolic"} for o in origins)
        else "exponential_amplitude"
    )
    return _particular_result(
        "constant_coefficient_exponential_amplitude_engine",
        combined,
        {
            "transformed_rhs": sp.expand(sp.sympify(rhs).rewrite(sp.exp)),
            "decomposition": tuple(pieces),
            "parts": tuple(solved),
            "method_family": family,
            "engine": "exponential_amplitude",
        },
    )


def _hyper_trig_particular(ccpde: LinearConstantCoefficientPDE, rhs):
    result = _particular_from_exponential_amplitudes(ccpde, rhs)
    details = dict(result.details)
    details["reduced_from"] = sp.expand(rhs)
    details["origin_families"] = tuple(
        piece["origin"] for piece in details["decomposition"]
    )
    return _particular_result(
        "constant_coefficient_trig_hyperbolic",
        sp.simplify(
            sp.expand(result.solution)
            .rewrite(sp.sin)
            .rewrite(sp.cos)
            .rewrite(sp.sinh)
            .rewrite(sp.cosh)
        ),
        details,
    )


def _polynomial_particular(ccpde: LinearConstantCoefficientPDE, rhs):
    rhs = sp.expand(rhs)
    vars_ = ccpde.indep_vars
    c0 = sp.expand(ccpde.operator_terms.get(tuple([0] * len(vars_)), 0))
    if sp.simplify(c0) != 0:
        n_terms = dict(ccpde.operator_terms)
        n_terms[tuple([0] * len(vars_))] = sp.Integer(0)
        current = sp.expand(rhs / c0)
        acc = sp.expand(current)
        deg_rhs = sp.total_degree(rhs, *vars_) if rhs != 0 else 0
        max_steps = int(deg_rhs) + ccpde.order + 2
        for _ in range(max_steps):
            current = sp.expand(
                -_cc_operator_apply_from_terms(n_terms, current, vars_) / c0
            )
            if current == 0:
                break
            acc = sp.expand(acc + current)
        residual = sp.expand(
            _cc_operator_apply_from_terms(ccpde.operator_terms, acc, vars_) - rhs
        )
        if sp.simplify(residual) == 0:
            return _particular_result(
                "constant_coefficient_polynomial_inverse_operator",
                solution=sp.expand(acc),
                details={
                    "rhs": rhs,
                    "used_truncated_inverse": True,
                    "annihilator_style": "truncated_inverse",
                },
            )
    degree = int(sp.total_degree(rhs, *vars_)) + ccpde.order
    sol = _solve_polynomial_ansatz(ccpde, rhs, degree, prefix="pcoef")
    return _particular_result(
        "constant_coefficient_polynomial_ansatz",
        solution=sol,
        details={
            "rhs": rhs,
            "used_truncated_inverse": False,
            "degree": degree,
            "annihilator_style": "polynomial_ansatz",
        },
    )


def _shifted_operator_terms(ccpde: LinearConstantCoefficientPDE, phase):
    vars_ = ccpde.indep_vars
    kvec = _phase_vector(phase, vars_)
    if kvec is None:
        raise NotImplementedError(
            "Exponential phase must be affine in the independent variables."
        )
    new_terms = {}
    for mi, coeff in ccpde.operator_terms.items():
        ranges = [range(mi_i + 1) for mi_i in mi]
        for nu in product(*ranges):
            comb = sp.Integer(1)
            kpow = sp.Integer(1)
            out_mi = []
            for i, (mi_i, nu_i) in enumerate(zip(mi, nu)):
                comb *= sp.binomial(mi_i, nu_i)
                kpow *= kvec[i] ** (mi_i - nu_i)
                out_mi.append(nu_i)
            out_mi = tuple(out_mi)
            new_terms[out_mi] = sp.expand(
                new_terms.get(out_mi, 0) + coeff * comb * kpow
            )
    new_terms = {
        mi: sp.expand(val) for mi, val in new_terms.items() if sp.simplify(val) != 0
    }
    return new_terms, kvec


def _polynomial_times_phase_exponential_particular(
    ccpde: LinearConstantCoefficientPDE, amplitude, phase
):
    vars_ = ccpde.indep_vars
    amplitude = sp.expand(amplitude)
    if not amplitude.is_polynomial(*vars_):
        raise NotImplementedError("Only polynomial amplitudes are supported.")
    shifted_terms, kvec = _shifted_operator_terms(ccpde, phase)
    shifted_pde = LinearConstantCoefficientPDE(
        indep_vars=ccpde.indep_vars,
        dep_function=ccpde.dep_function,
        operator_terms=shifted_terms,
        rhs=amplitude,
        operator_polynomial=ccpde.operator_polynomial,
        normalized_equation=ccpde.normalized_equation,
        order=ccpde.order,
    )
    shifted_symbol = _shifted_symbol_profile(ccpde, kvec)
    degree = (
        int(sp.total_degree(amplitude, *vars_))
        + ccpde.order
        + shifted_symbol.resonance_multiplicity
    )
    poly_result = _polynomial_particular(shifted_pde, amplitude)
    qsol = sp.expand(poly_result.solution)
    solution = sp.expand(qsol * sp.exp(phase))
    symbol_value = sp.simplify(_operator_symbol(ccpde, kvec))
    stages = []
    for fac in factor_constant_coefficient_operator(ccpde.operator_polynomial, vars_):
        fac_symbol = ConstantCoefficientSymbol(
            _make_operator_symbol(ccpde).variables, fac.polynomial
        )
        stages.append(
            {
                "factor": fac,
                "shifted": fac_symbol.shift(kvec),
                "evaluated": sp.expand(fac_symbol.evaluate(kvec)),
            }
        )
    return _particular_result(
        "constant_coefficient_polynomial_exponential",
        solution=solution,
        details={
            "phase": phase,
            "degree": degree,
            "wave_vector": kvec,
            "symbol_value": symbol_value,
            "resonant": symbol_value == 0,
            "resonance_multiplicity": shifted_symbol.resonance_multiplicity,
            "shifted_symbol": shifted_symbol,
            "shifted_polynomial_result": poly_result,
            "stages": tuple(stages),
        },
    )


def _polynomial_times_exponential_particular(ccpde: LinearConstantCoefficientPDE, rhs):
    extracted = _extract_exponential_term_data(rhs)
    if extracted is None:
        raise NotImplementedError("Expected an exponential factor.")
    amplitude, phase = extracted
    return _polynomial_times_phase_exponential_particular(ccpde, amplitude, phase)


def _forcing_family(rhs, vars_):
    rhs = sp.expand(rhs)
    if rhs == 0:
        return "zero"
    if rhs.is_polynomial(*vars_):
        return "polynomial"
    if rhs.func is sp.exp:
        return "exponential"
    if rhs.has(sp.sin, sp.cos):
        return "trigonometric"
    if rhs.has(sp.sinh, sp.cosh):
        return "hyperbolic"
    exp_factors = [arg for arg in sp.Mul.make_args(rhs) if arg.func is sp.exp]
    if exp_factors:
        amp = sp.expand(rhs / exp_factors[0])
        if amp.is_polynomial(*vars_):
            return "polynomial_exponential"
    return "generic"


def factor_constant_coefficient_operator(operator_polynomial, indep_vars):
    msyms = sp.symbols(f"m0:{len(indep_vars)}")
    poly = sp.Poly(operator_polynomial, *msyms, domain="EX")
    facs = sp.factor_list(poly.as_expr(), *msyms)
    factors = []
    for fac, mult in facs[1]:
        fac_poly = sp.Poly(fac, *msyms, domain="EX")
        coeffs = tuple(
            sp.expand(fac_poly.coeff_monomial(msyms[i])) for i in range(len(msyms))
        )
        const = sp.expand(fac_poly.TC())
        factors.append(
            ConstantCoefficientOperatorFactor(
                polynomial=sp.expand(fac_poly.as_expr()),
                multiplicity=int(mult),
                total_degree=int(fac_poly.total_degree()),
                coefficients=coeffs,
                constant_term=const,
            )
        )
    return tuple(factors)


def build_constant_coefficient_operator_profile(
    eq_or_expr, dep_expr_or_func, indep_vars=None
):
    ccpde = detect_linear_constant_coefficient_pde(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    return ConstantCoefficientOperatorProfile(
        pde=ccpde,
        forcing_family=_forcing_family(ccpde.rhs, ccpde.indep_vars),
        factors=factor_constant_coefficient_operator(
            ccpde.operator_polynomial, ccpde.indep_vars
        ),
        symbol=_make_operator_symbol(ccpde),
    )


def invert_factored_constant_coefficient_operator_on_forcing(
    eq_or_expr, dep_expr_or_func, indep_vars=None
):
    profile = build_constant_coefficient_operator_profile(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    ccpde = profile.pde
    base = invert_constant_coefficient_operator_on_forcing(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    rhs = sp.expand(sp.sympify(ccpde.rhs))
    stage_data = []
    extracted = _extract_exponential_term_data(rhs)
    wave_vector = None
    if extracted is not None:
        _, phase = extracted
        wave_vector = _phase_vector(phase, ccpde.indep_vars)
    for factor in profile.factors:
        fac_symbol = ConstantCoefficientSymbol(
            profile.symbol.variables, factor.polynomial
        )
        stage = {"factor": factor}
        if wave_vector is not None:
            stage["evaluation"] = sp.expand(fac_symbol.evaluate(wave_vector))
            stage["shifted"] = fac_symbol.shift(wave_vector)
        stage_data.append(stage)
    details = dict(base.details)
    details["factor_pipeline"] = tuple(stage_data)
    return _particular_result(base.method, base.solution, details)


def invert_constant_coefficient_operator_on_forcing(
    eq_or_expr, dep_expr_or_func, indep_vars=None
):
    ccpde = detect_linear_constant_coefficient_pde(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    rhs = sp.expand(sp.sympify(ccpde.rhs))
    vars_ = ccpde.indep_vars
    if rhs == 0:
        return _particular_result(
            "constant_coefficient_zero_rhs", sp.Integer(0), {"rhs": rhs}
        )
    decomposed = _decompose_rhs_to_exponential_amplitudes(rhs, vars_)
    if rhs.has(sp.sin, sp.cos, sp.sinh, sp.cosh):
        return _hyper_trig_particular(ccpde, rhs)
    if decomposed is not None and rhs.is_Add:
        return _particular_from_exponential_amplitudes(ccpde, rhs)
    if rhs.is_polynomial(*vars_):
        return _polynomial_particular(ccpde, rhs)
    extracted = _extract_exponential_term_data(rhs)
    if extracted is not None:
        amplitude, phase = extracted
        return _solve_exponential_amplitude_term(
            ccpde, amplitude, phase, origin="exponential"
        )
    if rhs.has(sp.sin, sp.cos, sp.sinh, sp.cosh):
        return _hyper_trig_particular(ccpde, rhs)
    raise NotImplementedError(
        "Unsupported forcing family for the constant-coefficient inverse-operator solver."
    )


def _invert_affine_line_map(expr, line_var):
    expr = sp.expand(expr)
    alpha = sp.expand(sp.diff(expr, line_var))
    beta = sp.expand(expr.subs(line_var, 0))
    if alpha == 0 or alpha.has(line_var):
        raise NotImplementedError("Line parameterization is not affine and invertible.")
    return alpha, beta


def _fit_single_family_on_line(
    general_solution_result,
    uexpr,
    vars_,
    *,
    line_var,
    line_value,
    profile,
    particular=sp.Integer(0),
):
    families = tuple(general_solution_result.details.get("families", ()))
    if len(families) != 1:
        return None
    fam = families[0]
    if (
        fam.invariant is None
        or len(fam.generators) != 1
        or not isinstance(fam.generators[0], sp.FunctionClass)
    ):
        return None
    generator = fam.generators[0]
    other_vars = [v for v in vars_ if v != line_var]
    if len(other_vars) != 1:
        return None
    param = other_vars[0]
    base_expr = sp.expand(fam.expression)
    line_expr = sp.expand(base_expr.subs(line_var, line_value))
    inv_line = sp.expand(fam.invariant.subs(line_var, line_value))
    try:
        alpha, beta = _invert_affine_line_map(inv_line, param)
    except Exception:
        return None
    amp = sp.simplify(line_expr / generator(inv_line))
    if amp.has(generator):
        return None
    target = sp.expand(
        sp.sympify(profile) - sp.sympify(particular).subs(line_var, line_value)
    )
    mapped = sp.expand(target / amp)
    replacement = sp.expand(mapped.subs(param, (fam.invariant - beta) / alpha))
    fitted_expr = sp.expand(
        base_expr.xreplace({generator(fam.invariant): replacement}) + particular
    )
    return _general_solution_result(
        "linear_constant_coefficient_factored_fitted_data",
        solution=sp.Eq(uexpr, fitted_expr),
        details={
            **general_solution_result.details,
            "fitted_conditions": {
                "line_var": line_var,
                "line_value": line_value,
                "profile": profile,
            },
            "fit_strategy": "generator_substitution",
            "method_family": "condition_fitting",
        },
    )


def _fit_two_family_initial_data(
    general_solution_result,
    uexpr,
    vars_,
    *,
    curve_value,
    initial_profile,
    initial_time_derivative=None,
    particular=sp.Integer(0),
):
    if len(vars_) != 2:
        return None
    x, t = vars_
    families = tuple(general_solution_result.details.get("families", ()))
    simple = [
        f
        for f in families
        if f.invariant is not None
        and len(f.generators) == 1
        and isinstance(f.generators[0], sp.FunctionClass)
        and f.factor.multiplicity == 1
    ]
    if len(simple) != 2:
        return None
    f1, f2 = simple
    g1, g2 = f1.generators[0], f2.generators[0]
    expr1 = sp.expand(f1.expression)
    expr2 = sp.expand(f2.expression)
    if sp.simplify(expr1 / g1(f1.invariant)).has(g1) or sp.simplify(
        expr2 / g2(f2.invariant)
    ).has(g2):
        return None
    amp1 = sp.simplify(expr1 / g1(f1.invariant))
    amp2 = sp.simplify(expr2 / g2(f2.invariant))
    inv10 = sp.expand(f1.invariant.subs(t, curve_value))
    inv20 = sp.expand(f2.invariant.subs(t, curve_value))
    try:
        alpha1, beta1 = _invert_affine_line_map(inv10, x)
        alpha2, beta2 = _invert_affine_line_map(inv20, x)
    except Exception:
        return None
    if (
        sp.simplify(amp1.subs(t, curve_value) - 1) != 0
        or sp.simplify(amp2.subs(t, curve_value) - 1) != 0
    ):
        return None
    residual_profile = sp.expand(
        sp.sympify(initial_profile) - sp.sympify(particular).subs(t, curve_value)
    )
    if initial_time_derivative is None:
        # choose a symmetric split by setting a_x = b_x = G_x/2
        a_dx = sp.expand(sp.diff(residual_profile, x) / 2)
        a_expr = sp.integrate(a_dx, x)
        b_expr = sp.expand(residual_profile - a_expr)
    else:
        rhs_t = sp.expand(
            sp.sympify(initial_time_derivative)
            - sp.diff(sp.sympify(particular), t).subs(t, curve_value)
        )
        beta_t1 = sp.expand(sp.diff(f1.invariant, t))
        beta_t2 = sp.expand(sp.diff(f2.invariant, t))
        if any(item.has(x, t) for item in (beta_t1, beta_t2)):
            return None
        g_x = sp.expand(sp.diff(residual_profile, x))
        c1 = sp.simplify(beta_t1 / alpha1)
        c2 = sp.simplify(beta_t2 / alpha2)
        det = sp.simplify(c2 - c1)
        if det == 0:
            return None
        a_dx = sp.simplify((c2 * g_x - rhs_t) / det)
        b_dx = sp.simplify((rhs_t - c1 * g_x) / det)
        a_expr = sp.integrate(a_dx, x)
        b_expr = sp.expand(residual_profile - a_expr)
        if sp.simplify(sp.diff(b_expr, x) - b_dx) != 0:
            return None
    repl1 = sp.expand(a_expr.subs(x, (f1.invariant - beta1) / alpha1))
    repl2 = sp.expand(b_expr.subs(x, (f2.invariant - beta2) / alpha2))
    fitted_expr = sp.expand(
        expr1.xreplace({g1(f1.invariant): repl1})
        + expr2.xreplace({g2(f2.invariant): repl2})
        + particular
    )
    return _general_solution_result(
        "linear_constant_coefficient_factored_fitted_data",
        solution=sp.Eq(uexpr, fitted_expr),
        details={
            **general_solution_result.details,
            "fitted_conditions": {
                "curve_value": curve_value,
                "initial_profile": initial_profile,
                "initial_time_derivative": initial_time_derivative,
            },
            "fit_strategy": "generator_substitution",
            "method_family": "condition_fitting",
        },
    )


def _normalize_point_conditions(ics=None, bcs=None):
    source = None
    data = None
    if bcs:
        source = "bcs"
        data = dict(bcs)
    elif ics:
        source = "ics"
        data = dict(ics)
    else:
        return None
    conditions = dict(data.get("point_conditions", {}))
    if "boundary_profile" in data:
        conditions.setdefault(0, data["boundary_profile"])
    if "boundary_normal_derivative" in data:
        conditions.setdefault(1, data["boundary_normal_derivative"])
    if "initial_profile" in data:
        conditions.setdefault(0, data["initial_profile"])
    if "initial_derivative" in data:
        conditions.setdefault(1, data["initial_derivative"])
    if not conditions:
        return None
    point_var = data.get("point_var")
    point_value = data.get(
        "point_value", data.get("curve_value", data.get("line_value", 0))
    )
    return {
        "source": source,
        "point_var": point_var,
        "point_value": point_value,
        "conditions": conditions,
    }


def _normalize_line_conditions(vars_, ics=None, bcs=None):
    if ics:
        data = dict(ics)
        line_var = data.get("line_var", vars_[1] if len(vars_) >= 2 else vars_[0])
        line_value = data.get("line_value", data.get("curve_value", 0))
        conditions = dict(data.get("line_conditions", {}))
        if "initial_profile" in data:
            conditions.setdefault(0, data["initial_profile"])
        if "initial_time_derivative" in data:
            conditions.setdefault(1, data["initial_time_derivative"])
        for order, expr in data.get("initial_derivatives", {}).items():
            conditions.setdefault(int(order), expr)
        if conditions:
            return {
                "source": "ics",
                "line_var": line_var,
                "line_value": line_value,
                "conditions": conditions,
            }
    if bcs:
        data = dict(bcs)
        line_var = data.get("line_var", vars_[0])
        line_value = data.get("line_value", 0)
        conditions = dict(data.get("line_conditions", {}))
        if "boundary_profile" in data:
            conditions.setdefault(0, data["boundary_profile"])
        if "boundary_normal_derivative" in data:
            conditions.setdefault(1, data["boundary_normal_derivative"])
        for order, expr in data.get("boundary_derivatives", {}).items():
            conditions.setdefault(int(order), expr)
        if conditions:
            return {
                "source": "bcs",
                "line_var": line_var,
                "line_value": line_value,
                "conditions": conditions,
            }
    return None


def _collect_generator_linear_unknowns(expr, param):
    unknowns = []
    replacements = {}
    for node in sp.preorder_traversal(expr):
        if (
            isinstance(node, sp.Derivative)
            and len(node.variables) >= 1
            and all(v == param for v in node.variables)
            and node.expr.is_Function
        ):
            sym = sp.Symbol(f"_jet_{len(unknowns)}")
            unknowns.append((node, sym))
            replacements[node] = sym
        elif node.is_Function and node.free_symbols == {param}:
            sym = sp.Symbol(f"_jet_{len(unknowns)}")
            unknowns.append((node, sym))
            replacements[node] = sym
    return tuple(unknowns), replacements


def _linear_algebra_solve_equations(equations, unknowns):
    exprs = [
        sp.expand(eq.lhs - eq.rhs) if isinstance(eq, sp.Equality) else sp.expand(eq)
        for eq in equations
    ]
    try:
        A, b = sp.linear_eq_to_matrix(exprs, list(unknowns))
    except Exception:
        return None
    sols = sp.linsolve((A, b), *list(unknowns))
    if not sols:
        return None
    vec = next(iter(sols))
    if len(vec) != len(unknowns):
        return None
    free = set()
    for item in vec:
        free |= {s for s in item.free_symbols if s not in set(unknowns)}
    sub_zero = {s: 0 for s in free}
    return {u: sp.expand(v.subs(sub_zero)) for u, v in zip(unknowns, vec)}


def _fit_1d_family_at_point(
    general_solution_result,
    uexpr,
    vars_,
    *,
    point_var=None,
    point_value=0,
    conditions=None,
    particular=sp.Integer(0),
):
    if len(vars_) != 1 or not conditions:
        return None
    var = vars_[0]
    if point_var is not None and point_var != var:
        return None
    families = tuple(general_solution_result.details.get("families", ()))
    homogeneous = sp.expand(sum(f.expression for f in families))
    unknowns = sorted(homogeneous.free_symbols, key=sp.default_sort_key)
    if not unknowns:
        return None
    eqs = []
    for order, target in sorted(conditions.items()):
        lhs = sp.expand(sp.diff(homogeneous, var, int(order)).subs(var, point_value))
        rhs = sp.expand(
            sp.sympify(target)
            - sp.diff(sp.sympify(particular), var, int(order)).subs(var, point_value)
        )
        eqs.append(sp.Eq(lhs, rhs))
    sub = _linear_algebra_solve_equations(eqs, tuple(unknowns))
    if sub is None:
        try:
            sol = sp.solve(eqs, unknowns, dict=True)
        except Exception:
            sol = []
        if not sol:
            return None
        sub = dict(sol[0])
    fitted_expr = sp.expand((homogeneous + particular).subs(sub))
    return _general_solution_result(
        "constant_coefficient_factored_point_fit",
        solution=sp.Eq(uexpr, fitted_expr),
        details={
            **general_solution_result.details,
            "fitted_conditions": {
                "point_var": var,
                "point_value": point_value,
                "conditions": dict(sorted(conditions.items())),
            },
            "fit_strategy": "linear_algebra_over_generators",
            "method_family": "condition_fitting",
        },
    )


def _build_generator_line_maps(families, line_var, line_value, param):
    maps = []
    for fam_index, fam in enumerate(families):
        if fam.invariant is None:
            return None
        inv_line = sp.expand(fam.invariant.subs(line_var, line_value))
        try:
            alpha, beta = _invert_affine_line_map(inv_line, param)
        except Exception:
            return None
        for gen_index, gen in enumerate(fam.generators):
            if not isinstance(gen, sp.FunctionClass):
                return None
            yfunc = sp.Function(f"_Y_{fam_index}_{gen_index}")
            maps.append(
                {
                    "generator": gen,
                    "yfunc": yfunc,
                    "invariant": fam.invariant,
                    "inv_line": inv_line,
                    "alpha": sp.expand(alpha),
                    "beta": sp.expand(beta),
                }
            )
    return maps


def _replace_generator_line_jets(expr, param, generator_maps):
    expr = sp.expand(expr)
    for item in generator_maps:
        expr = expr.xreplace(
            {item["generator"](item["inv_line"]): item["yfunc"](param)}
        )

    def repl(node):
        if (
            not isinstance(node, sp.Subs)
            or len(node.variables) != 1
            or len(node.point) != 1
        ):
            return None
        point = sp.expand(node.point[0])
        for item in generator_maps:
            if point != item["inv_line"]:
                continue
            if isinstance(node.expr, sp.Derivative):
                deriv_expr = node.expr
                if (
                    deriv_expr.expr.func != item["generator"]
                    or len(deriv_expr.expr.args) != 1
                ):
                    continue
                pivot = node.variables[0]
                if deriv_expr.expr.args[0] != pivot:
                    continue
                order = deriv_expr.derivative_count
                return sp.expand(
                    sp.diff(item["yfunc"](param), param, order)
                    / (item["alpha"] ** order)
                )
        return None

    return expr.replace(lambda n: isinstance(n, sp.Subs), repl)


def _try_direct_line_function_solve(
    general_solution_result,
    uexpr,
    vars_,
    *,
    line_var,
    line_value,
    conditions,
    particular=sp.Integer(0),
):
    if len(vars_) != 2 or not conditions:
        return None
    param = next(v for v in vars_ if v != line_var)
    families = tuple(general_solution_result.details.get("families", ()))
    if not families:
        return None
    generator_maps = _build_generator_line_maps(families, line_var, line_value, param)
    if generator_maps is None:
        return None
    homogeneous = sp.expand(sum(f.expression for f in families))
    zero_order = sorted(
        [item["yfunc"](param) for item in generator_maps], key=sp.default_sort_key
    )
    equations = []
    for order, target in sorted(conditions.items()):
        lhs = sp.expand(
            sp.diff(homogeneous, line_var, int(order)).subs(line_var, line_value)
        )
        lhs = _replace_generator_line_jets(lhs, param, generator_maps)
        rhs = sp.expand(
            sp.sympify(target)
            - sp.diff(sp.sympify(particular), line_var, int(order)).subs(
                line_var, line_value
            )
        )
        equations.append(sp.Eq(lhs, rhs))
    linear_unknowns = tuple(zero_order)
    sub = _linear_algebra_solve_equations(equations, linear_unknowns)
    if sub is None:
        if any(
            eq.lhs.has(sp.Derivative) or eq.rhs.has(sp.Derivative) for eq in equations
        ):
            return None
        try:
            sol = sp.solve(equations, zero_order, dict=True)
        except Exception:
            sol = []
        if not sol:
            return None
        sub = dict(sol[0])
    replacement = {}
    for item in generator_maps:
        y0 = item["yfunc"](param)
        if y0 not in sub:
            return None
        replacement[item["generator"](item["invariant"])] = sp.expand(
            sub[y0].subs(param, (item["invariant"] - item["beta"]) / item["alpha"])
        )
    fitted_expr = sp.expand(homogeneous.xreplace(replacement) + particular)
    return _general_solution_result(
        "constant_coefficient_factored_line_fit",
        solution=sp.Eq(uexpr, fitted_expr),
        details={
            **general_solution_result.details,
            "fitted_conditions": {
                "line_var": line_var,
                "line_value": line_value,
                "conditions": dict(sorted(conditions.items())),
            },
            "fit_strategy": "linear_algebra_over_generators",
            "method_family": "condition_fitting",
        },
    )


def fit_constant_coefficient_solution_with_conditions(
    general_solution_result, dep_expr_or_func, indep_vars=None, *, ics=None, bcs=None
):
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    particular = general_solution_result.details.get("particular", 0)

    point_data = _normalize_point_conditions(ics=ics, bcs=bcs)
    if point_data is not None:
        fitted = _fit_1d_family_at_point(
            general_solution_result,
            uexpr,
            vars_,
            point_var=point_data.get("point_var"),
            point_value=point_data["point_value"],
            conditions=point_data["conditions"],
            particular=particular,
        )
        if fitted is not None:
            return fitted

    line_data = _normalize_line_conditions(vars_, ics=ics, bcs=bcs)
    if line_data is None:
        return None

    if (
        len(vars_) == 2
        and line_data["source"] == "ics"
        and 0 in line_data["conditions"]
    ):
        fitted = _fit_single_family_on_line(
            general_solution_result,
            uexpr,
            vars_,
            line_var=line_data["line_var"],
            line_value=line_data["line_value"],
            profile=line_data["conditions"][0],
            particular=particular,
        )
        if fitted is not None and 1 not in line_data["conditions"]:
            return fitted
        fitted = _fit_two_family_initial_data(
            general_solution_result,
            uexpr,
            vars_,
            curve_value=line_data["line_value"],
            initial_profile=line_data["conditions"][0],
            initial_time_derivative=line_data["conditions"].get(1),
            particular=particular,
        )
        if fitted is not None:
            return fitted

    if len(vars_) == 2:
        direct = _try_direct_line_function_solve(
            general_solution_result,
            uexpr,
            vars_,
            line_var=line_data["line_var"],
            line_value=line_data["line_value"],
            conditions=line_data["conditions"],
            particular=particular,
        )
        if direct is not None:
            return direct

    return None


def _attach_constant_coefficient_verification(
    result,
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    assumptions=True,
    particular_result=None,
):
    summary = verify_solution_with_conditions(
        eq_or_expr,
        result.solution,
        dep_expr_or_func,
        indep_vars,
        ics=ics,
        bcs=bcs,
        assumptions=assumptions,
    )
    details = dict(result.details)
    details["verification_summary"] = summary
    details.setdefault("solver_family", "constant_coefficient")
    details.setdefault("method_family", _method_family(result.method))
    details["method_family_report"] = {
        "selected_method": result.method,
        "method_family": details.get("method_family"),
        "particular_method": getattr(particular_result, "method", None)
        if particular_result is not None
        else None,
        "particular_method_family": particular_result.details.get("method_family")
        if particular_result is not None
        else None,
        "homogeneous_method": details.get("homogeneous").method
        if details.get("homogeneous") is not None
        else None,
        "homogeneous_method_family": details.get("homogeneous").details.get(
            "method_family"
        )
        if details.get("homogeneous") is not None
        else None,
        "fit_strategy": details.get("fit_strategy"),
    }
    if particular_result is not None:
        try:
            ccpde = build_constant_coefficient_operator_profile(
                eq_or_expr, dep_expr_or_func, indep_vars
            ).pde
            rhs_residual = sp.simplify(
                _cc_operator_apply_from_terms(
                    ccpde.operator_terms, particular_result.solution, ccpde.indep_vars
                )
                - ccpde.rhs
            )
            details["resonance_consistent"] = sp.simplify(rhs_residual) == 0
            details["particular_residual"] = rhs_residual
        except Exception:
            details["resonance_consistent"] = None
    return _general_solution_result(result.method, result.solution, details)


def _symbol_factor_to_direction(factor: ConstantCoefficientOperatorFactor):
    coeffs = [sp.expand(c) for c in factor.coefficients]
    const = sp.expand(factor.constant_term)
    if all(sp.simplify(c) == 0 for c in coeffs):
        return None
    scale = sp.Integer(1)
    for c in reversed(coeffs):
        cs = sp.simplify(c)
        if cs != 0:
            if cs.could_extract_minus_sign():
                scale = sp.Integer(-1)
            break
    coeffs = tuple(sp.expand(scale * c) for c in coeffs)
    const = sp.expand(scale * const)
    return coeffs, const


def _nullspace_invariants(coeffs, vars_):
    row = sp.Matrix([list(coeffs[: len(vars_)])])
    basis = row.nullspace()
    invariants = []
    xvec = sp.Matrix(list(vars_))
    for vec in basis:
        invariants.append(sp.expand((vec.T * xvec)[0]))
    return tuple(invariants)


def _make_generator_function(name, invariants):
    f = sp.Function(name)
    return f, f(*invariants) if invariants else f()


def _homogeneous_family_linear_factor(factor: ConstantCoefficientOperatorFactor, vars_):
    coeffs, const = _symbol_factor_to_direction(factor)
    coeffs = coeffs[: len(vars_)]
    if all(sp.simplify(c) == 0 for c in coeffs):
        return None
    dot = sp.expand(sum(sp.expand(a * v) for a, v in zip(coeffs, vars_)))
    denom = sp.expand(sum(sp.expand(a**2) for a in coeffs))
    if sp.simplify(denom) == 0:
        return None
    transverse = sp.expand(dot / denom)
    invariants = _nullspace_invariants(coeffs, vars_)
    pieces = []
    gens = []
    name_seed = abs(hash(sp.srepr(factor.polynomial))) % 10**8
    for j in range(factor.multiplicity):
        if len(vars_) == 1:
            c = sp.Symbol(f"C_{name_seed}_{j}")
            gens.append(c)
            gen_expr = c
        else:
            f, gen_expr = _make_generator_function(f"F_{name_seed}_{j}", invariants)
            gens.append(f)
        pieces.append(
            sp.expand((transverse**j) * sp.exp(-const * transverse) * gen_expr)
        )
    invariant_value = None
    if len(invariants) == 1:
        invariant_value = invariants[0]
    elif len(invariants) > 1:
        invariant_value = invariants
    return ConstantCoefficientHomogeneousFamily(
        factor=factor,
        method="constant_coefficient_homogeneous_linear_factor",
        expression=sp.expand(sum(pieces)),
        generators=tuple(gens),
        invariant=invariant_value,
        invariants=tuple(invariants),
        transverse=sp.expand(transverse),
    )


def _homogeneous_family_1d(factor: ConstantCoefficientOperatorFactor, var: sp.Symbol):
    return _homogeneous_family_linear_factor(factor, (var,))


def _homogeneous_family_2d_linear_factor(
    factor: ConstantCoefficientOperatorFactor, vars_
):
    return _homogeneous_family_linear_factor(factor, vars_)


def _family_to_factor_solution_2d(family: ConstantCoefficientHomogeneousFamily):
    if (
        family.invariant is None
        or family.transverse is None
        or len(family.factor.coefficients) < 2
    ):
        return None
    a, b = family.factor.coefficients[:2]
    return ConstantCoefficientFactorSolution2D(
        multiplicity=family.factor.multiplicity,
        a=sp.expand(a),
        b=sp.expand(b),
        c=sp.expand(family.factor.constant_term),
        invariant=family.invariant,
        transverse=family.transverse,
        arbitrary_functions=tuple(family.generators),
        expression=family.expression,
    )


def build_constant_coefficient_homogeneous_solution(
    eq_or_expr, dep_expr_or_func, indep_vars=None
):
    profile = build_constant_coefficient_operator_profile(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    ccpde = profile.pde
    families = []
    unsupported_factors = []
    for factor in profile.factors:
        if factor.total_degree != 1:
            unsupported_factors.append(factor)
            continue
        fam = _homogeneous_family_linear_factor(factor, ccpde.indep_vars)
        if fam is not None:
            families.append(fam)
    if not families:
        fallback = _build_1d_characteristic_root_homogeneous_solution(
            eq_or_expr, dep_expr_or_func, indep_vars
        )
        if fallback is not None:
            return fallback
        raise NotImplementedError(
            "No explicit homogeneous family builder available for this operator."
        )
    uexpr = ccpde.dep_function
    expr = sp.expand(sum(f.expression for f in families))
    return _general_solution_result(
        "constant_coefficient_homogeneous_family",
        solution=sp.Eq(uexpr, expr),
        details={
            "profile": profile,
            "families": tuple(families),
            "factor_solutions": tuple(
                fs
                for fs in (_family_to_factor_solution_2d(f) for f in families)
                if fs is not None
            ),
            "unsupported_factors": tuple(unsupported_factors),
        },
    )


def _build_1d_characteristic_root_homogeneous_solution(
    eq_or_expr, dep_expr_or_func, indep_vars=None
):
    profile = build_constant_coefficient_operator_profile(
        eq_or_expr, dep_expr_or_func, indep_vars
    )
    ccpde = profile.pde
    if len(ccpde.indep_vars) != 1:
        return None
    var = ccpde.indep_vars[0]
    msym = profile.symbol.variables[0]
    poly = sp.Poly(profile.symbol.expr, msym, domain="EX")
    if poly.degree() <= 0:
        return None
    roots = sp.roots(poly.as_expr(), msym)
    if not roots:
        return None
    expr = sp.Integer(0)
    families = []
    for root, mult in roots.items():
        gens = []
        pieces = []
        factor_poly = sp.expand((msym - root) ** mult)
        factor = ConstantCoefficientOperatorFactor(
            polynomial=factor_poly,
            multiplicity=int(mult),
            total_degree=int(mult),
            coefficients=(sp.Integer(1),),
            constant_term=sp.expand(-root),
        )
        for j in range(int(mult)):
            c = sp.Symbol(f"C_root_{abs(hash(sp.srepr(root))) % 10**8}_{j}")
            gens.append(c)
            pieces.append(sp.expand(c * (var**j) * sp.exp(root * var)))
        fam_expr = sp.expand(sum(pieces))
        families.append(
            ConstantCoefficientHomogeneousFamily(
                factor=factor,
                method="constant_coefficient_homogeneous_characteristic_root",
                expression=fam_expr,
                generators=tuple(gens),
                invariant=None,
                invariants=tuple(),
                transverse=var,
            )
        )
        expr += fam_expr
    return _general_solution_result(
        "constant_coefficient_homogeneous_characteristic_roots",
        sp.Eq(ccpde.dep_function, sp.expand(expr)),
        {
            "profile": profile,
            "families": tuple(families),
            "factor_solutions": tuple(),
            "unsupported_factors": tuple(),
        },
    )


def pdesolve_constant_coefficient(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars=None,
    *,
    ics=None,
    bcs=None,
    assumptions=True,
    canonical_first=True,
):
    uexpr, vars_ = _dep_and_vars(dep_expr_or_func, indep_vars)
    norm_eq = (
        canonicalize_pde_problem(eq_or_expr, uexpr, vars_)
        if canonical_first
        else sp.Eq(_as_zero_expr(eq_or_expr), 0)
    )
    profile = build_constant_coefficient_operator_profile(norm_eq, uexpr, vars_)
    ccpde = profile.pde
    particular_result = (
        invert_factored_constant_coefficient_operator_on_forcing(norm_eq, uexpr, vars_)
        if sp.expand(ccpde.rhs) != 0
        else _particular_result(
            "constant_coefficient_zero_rhs", sp.Integer(0), {"rhs": 0}
        )
    )
    particular = particular_result.solution

    homogeneous_result = None
    try:
        homogeneous_result = build_constant_coefficient_homogeneous_solution(
            sp.Eq(_cc_operator_apply_from_terms(ccpde.operator_terms, uexpr, vars_), 0),
            uexpr,
            vars_,
        )
    except Exception:
        homogeneous_result = None

    if homogeneous_result is not None:
        full = sp.expand(homogeneous_result.solution.rhs + particular)
        result = _general_solution_result(
            "constant_coefficient_homogeneous_plus_particular"
            if particular != 0
            else homogeneous_result.method,
            sp.Eq(uexpr, full),
            {
                "profile": profile,
                "homogeneous": homogeneous_result,
                "families": homogeneous_result.details.get("families", ()),
                "factor_solutions": homogeneous_result.details.get(
                    "factor_solutions", ()
                ),
                "particular": particular,
                "particular_result": particular_result,
            },
        )
        fitted = fit_constant_coefficient_solution_with_conditions(
            result, uexpr, vars_, ics=ics, bcs=bcs
        )
        if fitted is not None:
            return _attach_constant_coefficient_verification(
                fitted,
                norm_eq,
                uexpr,
                vars_,
                ics=ics,
                bcs=bcs,
                assumptions=assumptions,
                particular_result=particular_result,
            )
        return _attach_constant_coefficient_verification(
            result,
            norm_eq,
            uexpr,
            vars_,
            ics=ics,
            bcs=bcs,
            assumptions=assumptions,
            particular_result=particular_result,
        )

    try:
        sol = sp.pdsolve(norm_eq)
        if isinstance(sol, sp.Equality) and particular != 0:
            result = _general_solution_result(
                "constant_coefficient_pdsolve",
                sp.Eq(uexpr, sp.expand(sol.rhs + particular)),
                {
                    "profile": profile,
                    "particular": particular,
                    "particular_result": particular_result,
                },
            )
        else:
            result = _general_solution_result(
                "constant_coefficient_pdsolve",
                sol,
                {
                    "profile": profile,
                    "particular": particular,
                    "particular_result": particular_result,
                },
            )
        return _attach_constant_coefficient_verification(
            result,
            norm_eq,
            uexpr,
            vars_,
            ics=ics,
            bcs=bcs,
            assumptions=assumptions,
            particular_result=particular_result,
        )
    except Exception:
        result = _general_solution_result(
            "constant_coefficient_particular_only",
            sp.Eq(uexpr, particular),
            {
                "profile": profile,
                "particular": particular,
                "particular_result": particular_result,
            },
        )
        return _attach_constant_coefficient_verification(
            result,
            norm_eq,
            uexpr,
            vars_,
            ics=ics,
            bcs=bcs,
            assumptions=assumptions,
            particular_result=particular_result,
        )
