from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from .domains import DomainGeometry
from .results import FundamentalSolutionResult, GreenFunctionResult
from .verify import verify_kernel_representation


@dataclass(frozen=True)
class AdvancedGreenPlan:
    method: str
    operator_family: str
    geometry_kind: str
    boundary_family: str
    source_point: tuple[sp.Expr, ...]
    metadata: dict[str, Any]


def _as_eq(eq_or_expr):
    return (
        eq_or_expr
        if isinstance(eq_or_expr, sp.Equality)
        else sp.Eq(sp.sympify(eq_or_expr), 0)
    )


def _dep_expr(dep_expr_or_func, vars_):
    dep = dep_expr_or_func
    if getattr(dep, "is_Function", False) and not getattr(dep, "args", None):
        return dep(*vars_)
    return dep


def _split_operator_and_source(eq: sp.Equality):
    zero = sp.expand(eq.lhs - eq.rhs)
    terms = list(sp.Add.make_args(zero))
    src = [t for t in terms if t.has(sp.DiracDelta)]
    if not src:
        return zero, None
    op = sp.expand(sp.Add(*[t for t in terms if not t.has(sp.DiracDelta)]))
    source = sp.expand(-sp.Add(*src))
    return op, source


def _dirac_locations(source, vars_):
    if source is None:
        return tuple(sp.Symbol(f"{v.name}_0", real=True) for v in vars_)
    locs = []
    deltas = list(source.atoms(sp.DiracDelta))
    for v in vars_:
        loc = None
        for d in deltas:
            arg = sp.expand(d.args[0])
            coeff = sp.diff(arg, v)
            if coeff != 0 and all(not coeff.has(w) for w in vars_):
                loc = sp.simplify(-arg.subs(v, 0) / coeff)
                break
        locs.append(loc if loc is not None else sp.Symbol(f"{v.name}_0", real=True))
    return tuple(locs)


def _const_coeff(expr, term):
    coeff = sp.simplify(sp.expand(expr).coeff(term))
    return coeff if coeff != 0 and not coeff.has(*term.free_symbols) else coeff


def _match_laplace_nd(op, dep, vars_):
    if len(vars_) < 2:
        return None
    second_terms = [sp.diff(dep, v, 2) for v in vars_]
    coeffs = [sp.simplify(sp.expand(op).coeff(t)) for t in second_terms]
    if any(c == 0 for c in coeffs):
        return None
    first = coeffs[0]
    if any(sp.simplify(c - first) != 0 for c in coeffs[1:]):
        return None
    rem = sp.simplify(sp.expand(op - first * sum(second_terms)))
    if rem != 0 or any(first.has(v) for v in vars_):
        return None
    return {
        "family": "laplace_nd",
        "scale": sp.simplify(first),
        "dimension": len(vars_),
    }


def _match_helmholtz_nd(op, dep, vars_):
    if len(vars_) < 2:
        return None
    second_terms = [sp.diff(dep, v, 2) for v in vars_]
    coeffs = [sp.simplify(sp.expand(op).coeff(t)) for t in second_terms]
    if any(c == 0 for c in coeffs):
        return None
    first = coeffs[0]
    if any(sp.simplify(c - first) != 0 for c in coeffs[1:]):
        return None
    mass = sp.simplify(sp.expand(op).coeff(dep))
    rem = sp.simplify(sp.expand(op - first * sum(second_terms) - mass * dep))
    if mass == 0 or rem != 0 or any(first.has(v) or mass.has(v) for v in vars_):
        return None
    return {
        "family": "helmholtz_nd",
        "scale": sp.simplify(first),
        "lambda": sp.simplify(mass / first),
        "dimension": len(vars_),
    }


def _match_heat_nd(op, dep, vars_):
    if len(vars_) < 2:
        return None
    time_candidates = [v for v in vars_ if str(v) == "t"] or (
        [vars_[-1]] if str(vars_[-1]) == "t" else []
    )
    for t in time_candidates:
        ut = sp.diff(dep, t)
        time_coeff = sp.simplify(sp.expand(op).coeff(ut))
        if time_coeff == 0:
            continue
        space = [v for v in vars_ if v != t]
        if not space:
            continue
        second_terms = [sp.diff(dep, v, 2) for v in space]
        coeffs = [sp.simplify(sp.expand(op).coeff(term)) for term in second_terms]
        if any(c == 0 for c in coeffs):
            continue
        first = coeffs[0]
        if any(sp.simplify(c - first) != 0 for c in coeffs[1:]):
            continue
        rem = sp.simplify(sp.expand(op - time_coeff * ut - first * sum(second_terms)))
        if rem == 0 and all(not time_coeff.has(v) and not first.has(v) for v in vars_):
            return {
                "family": "heat_nd",
                "time": t,
                "space": tuple(space),
                "a": sp.simplify(-first / time_coeff),
                "dimension": len(space),
            }
    return None


def _match_wave_nd(op, dep, vars_):
    if len(vars_) < 2:
        return None
    time_candidates = [v for v in vars_ if str(v) == "t"] or (
        [vars_[-1]] if str(vars_[-1]) == "t" else []
    )
    for t in time_candidates:
        utt = sp.diff(dep, t, 2)
        time_coeff = sp.simplify(sp.expand(op).coeff(utt))
        if time_coeff == 0:
            continue
        space = [v for v in vars_ if v != t]
        if len(space) not in {1, 2, 3}:
            continue
        second_terms = [sp.diff(dep, v, 2) for v in space]
        coeffs = [sp.simplify(sp.expand(op).coeff(term)) for term in second_terms]
        if any(c == 0 for c in coeffs):
            continue
        first = coeffs[0]
        if any(sp.simplify(c - first) != 0 for c in coeffs[1:]):
            continue
        rem = sp.simplify(sp.expand(op - time_coeff * utt - first * sum(second_terms)))
        if rem == 0 and all(not time_coeff.has(v) and not first.has(v) for v in vars_):
            return {
                "family": "wave_nd",
                "time": t,
                "space": tuple(space),
                "c": sp.sqrt(sp.simplify(-first / time_coeff)),
                "dimension": len(space),
            }
    return None


def _match_cr(op, dep, vars_):
    if len(vars_) != 2:
        return None
    x, y = vars_
    ux = sp.diff(dep, x)
    uy = sp.diff(dep, y)
    a = sp.simplify(sp.expand(op).coeff(ux))
    b = sp.simplify(sp.expand(op).coeff(uy))
    rem = sp.simplify(sp.expand(op - a * ux - b * uy))
    if a == 0 or b == 0 or rem != 0 or a.has(x, y) or b.has(x, y):
        return None
    ratio = sp.simplify(b / a)
    if sp.simplify(ratio - sp.I) == 0:
        return {"family": "cauchy_riemann", "a": a, "x": x, "y": y}
    if sp.simplify(ratio + sp.I) == 0:
        return {"family": "anti_cauchy_riemann", "a": a, "x": x, "y": y}
    return None


def recognize_advanced_kernel_problem(eq_or_expr, dep_expr_or_func, indep_vars):
    vars_ = tuple(indep_vars)
    dep = _dep_expr(dep_expr_or_func, vars_)
    eq = _as_eq(eq_or_expr)
    op, source = _split_operator_and_source(eq)
    locs = _dirac_locations(source, vars_)
    for matcher in (
        _match_cr,
        _match_heat_nd,
        _match_wave_nd,
        _match_helmholtz_nd,
        _match_laplace_nd,
    ):
        res = matcher(op, dep, vars_)
        if res is not None:
            res["source_point"] = locs
            res["has_source"] = source is not None
            return res
    return None


def _laplace_free_nd(scale, coords, source):
    n = len(coords)
    diffsq = sum((c - s) ** 2 for c, s in zip(coords, source))
    if n == 2:
        return sp.log(diffsq) / (4 * sp.pi * scale)
    omega_n = 2 * sp.pi ** (sp.Rational(n, 2)) / sp.gamma(sp.Rational(n, 2))
    return diffsq ** (sp.Rational(2 - n, 2)) / (scale * (n - 2) * omega_n)


def _helmholtz_free_nd(scale, lam, coords, source):
    n = len(coords)
    r = sp.sqrt(sum((c - s) ** 2 for c, s in zip(coords, source)))
    if sp.simplify(lam) == 0:
        return _laplace_free_nd(scale, coords, source)
    h = -sp.I * sp.sqrt(lam)
    return (
        -(1 / scale)
        * (1 / (2 * sp.pi))
        * (h / (2 * sp.pi * r)) ** (sp.Rational(n, 2) - 1)
        * sp.besselk(sp.Rational(n, 2) - 1, h * r)
    )


def _heat_free_nd(a, space, t, source):
    spatial_source = source[: len(space)]
    tau = source[len(space)]
    n = len(space)
    diffsq = sum((c - s) ** 2 for c, s in zip(space, spatial_source))
    return (
        sp.Heaviside(t - tau)
        * sp.exp(-diffsq / (4 * a * (t - tau)))
        / (4 * sp.pi * a * (t - tau)) ** sp.Rational(n, 2)
    )


def _wave_free_nd(c, space, t, source):
    spatial_source = source[: len(space)]
    tau = source[len(space)]
    r = sp.sqrt(sum((c0 - s) ** 2 for c0, s in zip(space, spatial_source)))
    n = len(space)
    if n == 1:
        return sp.Heaviside(c * (t - tau) - sp.Abs(space[0] - spatial_source[0])) / (
            2 * c
        )
    if n == 2:
        return sp.Heaviside((t - tau) - r / c) / (
            2 * sp.pi * c * sp.sqrt((t - tau) ** 2 - r**2 / c**2)
        )
    if n == 3:
        return sp.DiracDelta((t - tau) - r / c) / (4 * sp.pi * r)
    raise NotImplementedError(
        "Wave kernel currently supported for 1-3 spatial dimensions only."
    )


def _laplace_half_plane(scale, x, y, xi, eta, *, boundary="dirichlet"):
    if boundary == "neumann":
        return -(
            sp.log((x - xi) ** 2 + (y - eta) ** 2)
            + sp.log((x - xi) ** 2 + (y + eta) ** 2)
        ) / (4 * sp.pi * scale)
    return sp.log(
        ((x - xi) ** 2 + (y + eta) ** 2) / ((x - xi) ** 2 + (y - eta) ** 2)
    ) / (4 * sp.pi * scale)


def _laplace_quadrant(scale, x, y, xi, eta, *, boundary="dirichlet"):
    rho1 = (x - xi) ** 2 + (y - eta) ** 2
    rho2 = (x - xi) ** 2 + (y + eta) ** 2
    rho3 = (x + xi) ** 2 + (y - eta) ** 2
    rho4 = (x + xi) ** 2 + (y + eta) ** 2
    if boundary == "neumann":
        return -(sp.log(rho1) + sp.log(rho2) + sp.log(rho3) + sp.log(rho4)) / (
            4 * sp.pi * scale
        )
    return (sp.log(rho2 * rho3 / (rho1 * rho4))) / (4 * sp.pi * scale)


def _laplace_strip(scale, x, y, xi, eta, a, *, boundary="dirichlet"):
    num = -sp.cos(sp.pi * (eta + y) / a) + sp.cosh(sp.pi * (x - xi) / a)
    den = -sp.cos(sp.pi * (y - eta) / a) + sp.cosh(sp.pi * (x - xi) / a)
    if boundary == "neumann":
        return (sp.log(1 / den) + sp.log(1 / num)) / (4 * sp.pi * scale)
    return sp.log(num / den) / (4 * sp.pi * scale)


def _laplace_semi_infinite_strip(scale, x, y, xi, eta, a, *, boundary="dirichlet"):
    t1 = sp.cosh(sp.pi * (x - xi) / a) - sp.cos(sp.pi * (y + eta) / a)
    t2 = sp.cosh(sp.pi * (x - xi) / a) - sp.cos(sp.pi * (y - eta) / a)
    t3 = sp.cosh(sp.pi * (x + xi) / a) - sp.cos(sp.pi * (y + eta) / a)
    t4 = sp.cosh(sp.pi * (x + xi) / a) - sp.cos(sp.pi * (y - eta) / a)
    if boundary == "neumann":
        return (sp.log(1 / t2) + sp.log(1 / t1) + sp.log(1 / t4) + sp.log(1 / t3)) / (
            4 * sp.pi * scale
        )
    return (sp.log(t1 / t2) - sp.log(t3 / t4)) / (4 * sp.pi * scale)


def _helmholtz_quadrant(scale, lam, x, y, xi, eta, *, boundary="dirichlet"):
    rho1 = sp.sqrt((x - xi) ** 2 + (y - eta) ** 2)
    rho2 = sp.sqrt((x - xi) ** 2 + (y + eta) ** 2)
    rho3 = sp.sqrt((x + xi) ** 2 + (y - eta) ** 2)
    rho4 = sp.sqrt((x + xi) ** 2 + (y + eta) ** 2)
    if sp.simplify(lam) == 0:
        return _laplace_quadrant(scale, x, y, xi, eta, boundary=boundary)
    k = sp.sqrt(lam)

    def helmholtz_kernel(radius):
        return -sp.I / 4 * sp.hankel2(0, k * radius)

    if boundary == "neumann":
        return (
            helmholtz_kernel(rho1)
            + helmholtz_kernel(rho2)
            + helmholtz_kernel(rho3)
            + helmholtz_kernel(rho4)
        ) / scale
    return (
        helmholtz_kernel(rho1)
        - helmholtz_kernel(rho2)
        - helmholtz_kernel(rho3)
        + helmholtz_kernel(rho4)
    ) / scale


def _helmholtz_strip(scale, lam, x, y, xi, eta, a, *, boundary="dirichlet"):
    n = sp.Symbol("n", integer=True, nonnegative=True)
    qn = sp.pi * n / a
    beta = sp.sqrt(qn**2 - lam)
    if boundary == "neumann":
        term = (
            ((2 - sp.KroneckerDelta(n, 0)) / beta)
            * sp.exp(-beta * sp.Abs(x - xi))
            * sp.cos(qn * y)
            * sp.cos(qn * eta)
        )
        return (1 / (2 * a * scale)) * sp.Sum(term, (n, 0, sp.oo))
    n = sp.Symbol("n", integer=True, positive=True)
    qn = sp.pi * n / a
    beta = sp.sqrt(qn**2 - lam)
    term = sp.exp(-beta * sp.Abs(x - xi)) * sp.sin(qn * y) * sp.sin(qn * eta) / beta
    return (1 / (a * scale)) * sp.Sum(term, (n, 1, sp.oo))


def _helmholtz_semi_infinite_strip(
    scale, lam, x, y, xi, eta, a, *, boundary="dirichlet"
):
    n = sp.Symbol("n", integer=True, nonnegative=True)
    qn = sp.pi * n / a
    beta = sp.sqrt(qn**2 - lam)
    if boundary == "neumann":
        term = (
            ((2 - sp.KroneckerDelta(n, 0)) / beta)
            * (sp.exp(-beta * sp.Abs(x - xi)) + sp.exp(-beta * sp.Abs(x + xi)))
            * sp.cos(qn * y)
            * sp.cos(qn * eta)
        )
        return (1 / (2 * a * scale)) * sp.Sum(term, (n, 0, sp.oo))
    n = sp.Symbol("n", integer=True, positive=True)
    qn = sp.pi * n / a
    beta = sp.sqrt(qn**2 - lam)
    term = (
        (sp.exp(-beta * sp.Abs(x - xi)) - sp.exp(-beta * sp.Abs(x + xi)))
        * sp.sin(qn * y)
        * sp.sin(qn * eta)
        / beta
    )
    return (1 / (a * scale)) * sp.Sum(term, (n, 1, sp.oo))


def _mirror_point(coords, source, axis):
    out = list(source)
    out[axis] = -out[axis]
    return tuple(out)


def _heat_half_space(a, space, t, source, *, axis=-1, boundary="dirichlet"):
    base = _heat_free_nd(a, space, t, source)
    image = _heat_free_nd(
        a, space, t, _mirror_point(space, source[:-1], axis) + (source[-1],)
    )
    return sp.simplify(base - image if boundary == "dirichlet" else base + image)


def _wave_half_space(c, space, t, source, *, axis=-1, boundary="dirichlet"):
    base = _wave_free_nd(c, space, t, source)
    image = _wave_free_nd(
        c, space, t, _mirror_point(space, source[:-1], axis) + (source[-1],)
    )
    return sp.simplify(base - image if boundary == "dirichlet" else base + image)


def _laplace_rectangle_dirichlet(scale, x, y, xi, eta, a, b):
    n, m = sp.symbols("n m", integer=True, positive=True)
    pn = sp.pi * n / a
    qm = sp.pi * m / b
    return (4 / (a * b * scale)) * sp.Sum(
        sp.sin(pn * x)
        * sp.sin(pn * xi)
        * sp.sin(qm * y)
        * sp.sin(qm * eta)
        / (pn**2 + qm**2),
        (n, 1, sp.oo),
        (m, 1, sp.oo),
    )


def _helmholtz_half_plane(scale, lam, x, y, xi, eta, *, boundary="dirichlet"):
    rho1 = sp.sqrt((x - xi) ** 2 + (y - eta) ** 2)
    rho2 = sp.sqrt((x - xi) ** 2 + (y + eta) ** 2)
    if sp.simplify(lam) == 0:
        return _laplace_half_plane(scale, x, y, xi, eta, boundary=boundary)
    k = sp.sqrt(lam)
    term1 = -sp.I / 4 * sp.hankel2(0, k * rho1)
    term2 = -sp.I / 4 * sp.hankel2(0, k * rho2)
    return (term1 - term2 if boundary == "dirichlet" else term1 + term2) / scale


def _helmholtz_rectangle_dirichlet(scale, lam, x, y, xi, eta, a, b):
    n, m = sp.symbols("n m", integer=True, positive=True)
    pn = sp.pi * n / a
    qm = sp.pi * m / b
    return (4 / (a * b * scale)) * sp.Sum(
        sp.sin(pn * x)
        * sp.sin(pn * xi)
        * sp.sin(qm * y)
        * sp.sin(qm * eta)
        / (pn**2 + qm**2 - lam),
        (n, 1, sp.oo),
        (m, 1, sp.oo),
    )


def _laplace3d_half_space(scale, x, y, z, xi, eta, zeta, *, boundary="dirichlet"):
    rho1 = sp.sqrt((x - xi) ** 2 + (y - eta) ** 2 + (z - zeta) ** 2)
    rho2 = sp.sqrt((x - xi) ** 2 + (y - eta) ** 2 + (z + zeta) ** 2)
    if boundary == "neumann":
        return (1 / (4 * sp.pi * scale)) * (1 / rho1 + 1 / rho2)
    return (1 / (4 * sp.pi * scale)) * (1 / rho1 - 1 / rho2)


def _extract_extent(geom, name, default=None):
    if geom is None:
        return default
    ext = (geom.extents or {}).get(name)
    if isinstance(ext, tuple) and len(ext) == 2:
        return ext
    return default


def execute_advanced_green_plan(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars,
    *,
    bcs=None,
    geometry=None,
    assumptions=True,
):
    vars_ = tuple(indep_vars)
    dep = _dep_expr(dep_expr_or_func, vars_)
    recog = recognize_advanced_kernel_problem(eq_or_expr, dep, vars_)
    if recog is None:
        raise NotImplementedError(
            "No advanced kernel/Green-function support recognized for this operator."
        )
    family = recog["family"]
    source = tuple(recog["source_point"])
    geom = geometry if isinstance(geometry, DomainGeometry) else None
    geom_kind = (
        getattr(geom, "kind", None)
        or (geometry if isinstance(geometry, str) else None)
        or (
            "full_space"
            if family in {"laplace_nd", "helmholtz_nd", "heat_nd", "wave_nd"}
            else "free"
        )
    )
    if geom_kind == "unspecified_spacetime" and family in {
        "laplace_nd",
        "helmholtz_nd",
        "heat_nd",
        "wave_nd",
    }:
        geom_kind = "full_space"
    boundary = "free"
    text = repr(bcs)
    if "Dirichlet" in text or "dirichlet" in text:
        boundary = "dirichlet"
    elif "Neumann" in text or "neumann" in text:
        boundary = "neumann"

    if family == "laplace_nd":
        scale = recog["scale"]
        if recog["dimension"] == 2:
            x, y = vars_[:2]
            xi, eta = source[:2]
            if geom_kind in {"full_space", "full_plane", "free"}:
                kernel = _laplace_free_nd(scale, vars_, source)
            elif geom_kind == "half_plane":
                kernel = _laplace_half_plane(scale, x, y, xi, eta, boundary=boundary)
            elif geom_kind == "quadrant":
                kernel = _laplace_quadrant(scale, x, y, xi, eta, boundary=boundary)
            elif geom_kind == "strip":
                a = (
                    _extract_extent(geom, str(y), (0, sp.Symbol("a", positive=True)))
                    or (0, sp.Symbol("a", positive=True))
                )[1]
                kernel = _laplace_strip(scale, x, y, xi, eta, a, boundary=boundary)
            elif geom_kind == "semi_infinite_strip":
                a = (
                    _extract_extent(geom, str(y), (0, sp.Symbol("a", positive=True)))
                    or (0, sp.Symbol("a", positive=True))
                )[1]
                kernel = _laplace_semi_infinite_strip(
                    scale, x, y, xi, eta, a, boundary=boundary
                )
            elif geom_kind == "rectangle":
                a = _extract_extent(geom, "x", (0, sp.Symbol("a", positive=True)))[1]
                b = _extract_extent(geom, "y", (0, sp.Symbol("b", positive=True)))[1]
                kernel = _laplace_rectangle_dirichlet(scale, x, y, xi, eta, a, b)
            else:
                raise NotImplementedError(f"Unsupported Laplace geometry: {geom_kind}")
        elif recog["dimension"] == 3 and geom_kind == "half_space":
            x, y, z = vars_[:3]
            xi, eta, zeta = source[:3]
            kernel = _laplace3d_half_space(
                scale, x, y, z, xi, eta, zeta, boundary=boundary
            )
        else:
            kernel = _laplace_free_nd(scale, vars_, source)
        rtype = (
            FundamentalSolutionResult
            if boundary == "free" and geom_kind in {"free", "full_space", "full_plane"}
            else GreenFunctionResult
        )
    elif family == "helmholtz_nd":
        scale, lam = recog["scale"], recog["lambda"]
        if recog["dimension"] == 2:
            x, y = vars_[:2]
            xi, eta = source[:2]
            if geom_kind in {"full_space", "full_plane", "free"}:
                kernel = _helmholtz_free_nd(scale, lam, vars_, source)
            elif geom_kind == "half_plane":
                kernel = _helmholtz_half_plane(
                    scale, lam, x, y, xi, eta, boundary=boundary
                )
            elif geom_kind == "quadrant":
                kernel = _helmholtz_quadrant(
                    scale, lam, x, y, xi, eta, boundary=boundary
                )
            elif geom_kind == "strip":
                a = (
                    _extract_extent(geom, str(y), (0, sp.Symbol("a", positive=True)))
                    or (0, sp.Symbol("a", positive=True))
                )[1]
                kernel = _helmholtz_strip(
                    scale, lam, x, y, xi, eta, a, boundary=boundary
                )
            elif geom_kind == "semi_infinite_strip":
                a = (
                    _extract_extent(geom, str(y), (0, sp.Symbol("a", positive=True)))
                    or (0, sp.Symbol("a", positive=True))
                )[1]
                kernel = _helmholtz_semi_infinite_strip(
                    scale, lam, x, y, xi, eta, a, boundary=boundary
                )
            elif geom_kind == "rectangle":
                a = _extract_extent(geom, "x", (0, sp.Symbol("a", positive=True)))[1]
                b = _extract_extent(geom, "y", (0, sp.Symbol("b", positive=True)))[1]
                kernel = _helmholtz_rectangle_dirichlet(scale, lam, x, y, xi, eta, a, b)
            else:
                raise NotImplementedError(
                    f"Unsupported Helmholtz geometry: {geom_kind}"
                )
        else:
            kernel = _helmholtz_free_nd(scale, lam, vars_, source)
        rtype = (
            FundamentalSolutionResult
            if boundary == "free" and geom_kind in {"free", "full_space", "full_plane"}
            else GreenFunctionResult
        )
    elif family == "heat_nd":
        if geom_kind in {"half_plane", "half_space"} and boundary in {
            "dirichlet",
            "neumann",
        }:
            kernel = _heat_half_space(
                recog["a"],
                recog["space"],
                recog["time"],
                source,
                axis=-1,
                boundary=boundary,
            )
            rtype = GreenFunctionResult
        else:
            kernel = _heat_free_nd(recog["a"], recog["space"], recog["time"], source)
            rtype = FundamentalSolutionResult
    elif family == "wave_nd":
        if geom_kind in {"half_plane", "half_space"} and boundary in {
            "dirichlet",
            "neumann",
        }:
            kernel = _wave_half_space(
                recog["c"],
                recog["space"],
                recog["time"],
                source,
                axis=-1,
                boundary=boundary,
            )
            rtype = GreenFunctionResult
        else:
            kernel = _wave_free_nd(recog["c"], recog["space"], recog["time"], source)
            rtype = FundamentalSolutionResult
    elif family in {"cauchy_riemann", "anti_cauchy_riemann"}:
        x, y = vars_[:2]
        a = recog["a"]
        kernel = (
            1 / (2 * a * (x + sp.I * y))
            if family == "cauchy_riemann"
            else 1 / (2 * a * (x - sp.I * y))
        )
        rtype = FundamentalSolutionResult
    else:
        raise NotImplementedError(f"Unsupported advanced family: {family}")

    verification = verify_kernel_representation(
        eq_or_expr,
        kernel,
        dep,
        vars_,
        geometry=geom,
        bcs=bcs,
        operator_family=family,
        boundary_family=boundary,
    )
    meta = {
        "operator_family": family,
        "geometry_kind": geom_kind,
        "boundary_family": boundary,
        "source_point": source,
        "recognition": recog,
        "subsystem": "advanced_green",
        "verification": verification,
    }
    if rtype is FundamentalSolutionResult:
        return FundamentalSolutionResult(
            method="advanced_fundamental_solution",
            solution=kernel,
            classification=family,
            metadata=meta,
            operator_family=family,
            kernel=kernel,
            source_point=source,
        )
    return GreenFunctionResult(
        method="advanced_green_function",
        solution=kernel,
        classification=family,
        metadata=meta,
        operator_family=family,
        kernel=kernel,
        source_point=source,
        boundary_type=boundary,
    )


def solve_linear_ode_green_function(
    eq_or_expr, dep_expr_or_func, var, *, conditions=None
):
    x = var
    dep = _dep_expr(dep_expr_or_func, (x,))
    eq = _as_eq(eq_or_expr)
    conds = list(conditions or ())
    try:
        ics = {}
        for c in conds:
            if isinstance(c, sp.Equality):
                ics[c.lhs] = c.rhs
        sol = sp.dsolve(eq, dep, ics=ics or None)
        kernel = sol.rhs if isinstance(sol, sp.Equality) else sol
        constants = sorted(
            [s for s in kernel.free_symbols if s.name.startswith("C")],
            key=lambda s: s.name,
        )
        if constants:
            kernel = sp.simplify(kernel.subs({c: 0 for c in constants}))
        return GreenFunctionResult(
            method="ode_green_function",
            solution=kernel,
            classification="linear_ode_green",
            metadata={"subsystem": "advanced_green", "conditions": tuple(conds)},
            operator_family="linear_ode",
            kernel=kernel,
            source_point=(sp.Integer(0),),
            boundary_type="ivp_or_bvp",
        )
    except Exception as exc:
        raise NotImplementedError(
            "Could not construct linear ODE Green function with current symbolic backend."
        ) from exc


__all__ = [
    "AdvancedGreenPlan",
    "recognize_advanced_kernel_problem",
    "execute_advanced_green_plan",
    "solve_linear_ode_green_function",
]
