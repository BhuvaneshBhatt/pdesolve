import pytest
import sympy as sp

from pdesolve import pdesolve


def _build_examples():
    x, y, _z, t, s, r = sp.symbols("x y z t s r", real=True)
    a, b, c, m, q, T, sigma, k, eps, hbar, _lam = sp.symbols(
        "a b c m q T sigma k eps hbar lam", real=True
    )
    theta = sp.symbols("theta", real=True)
    u = sp.Function("u")
    v = sp.Function("v")
    Psi = sp.Function("Psi")
    f = sp.Function("f")
    C = sp.Function("C")
    Y = sp.Function("Y")
    phi0 = sp.Function("phi0")
    psiF = sp.Function("psi")
    CaputoD = sp.Function("CaputoD")
    G = sp.Function("G")
    examples = {}
    examples["sol_pde_01"] = (
        sp.Eq(sp.diff(u(x, y), x) + 3 * sp.diff(u(x, y), y) + u(x, y), 1),
        u(x, y),
        (x, y),
    )
    examples["sol_pde_02"] = (
        sp.Eq(3 * sp.diff(u(x, y), x) + 5 * sp.diff(u(x, y), y), x),
        u(x, y),
        (x, y),
    )
    examples["sol_pde_03"] = (
        [
            sp.Eq(
                x * sp.diff(u(x, y), y) + y * sp.diff(u(x, y), x), -4 * x * y * u(x, y)
            ),
            sp.Eq(u(x, 0), sp.exp(-(x**2))),
        ],
        u(x, y),
        (x, y),
    )
    examples["sol_pde_04"] = (
        sp.Eq(sp.diff(u(t, x), t) + c * sp.diff(u(t, x), x), 0),
        u(t, x),
        (t, x),
    )
    examples["sol_pde_05"] = (
        [
            sp.Eq(sp.diff(u(t, x), t) + c * sp.diff(u(t, x), x), 0),
            sp.Eq(u(0, x), sp.exp(-(x**2))),
        ],
        u(t, x),
        (t, x),
    )
    examples["sol_pde_06"] = (
        [
            sp.Eq(sp.diff(u(t, x), t) + sp.diff(u(t, x), x), 0),
            sp.Eq(u(t, 0), 0),
            sp.Eq(u(0, x), sp.sin(x)),
        ],
        u(t, x),
        (t, x),
    )
    examples["sol_pde_07"] = (
        sp.Eq(2 * sp.diff(u(x, y), x) + 5 * sp.diff(u(x, y), y), u(x, y) ** 2 + 1),
        u(x, y),
        (x, y),
    )
    examples["sol_pde_08"] = (
        [
            sp.Eq(sp.diff(u(x, y), x) + u(x, y) * sp.diff(u(x, y), y), 0),
            sp.Eq(u(x, 0), 1 / (x + 1)),
        ],
        u(x, y),
        (x, y),
    )
    examples["sol_pde_09"] = (
        sp.Eq(
            u(x, y),
            x * sp.diff(u(x, y), x)
            + y * sp.diff(u(x, y), y)
            + sp.sin(sp.diff(u(x, y), x) + sp.diff(u(x, y), y)),
        ),
        u(x, y),
        (x, y),
    )
    examples["sol_pde_10"] = (
        sp.Eq(
            sp.diff(u(x, t), t)
            + sp.diff(u(x, t), x, 3)
            + 6 * u(x, t) * sp.diff(u(x, t), x),
            0,
        ),
        u(x, t),
        (x, t),
    )
    examples["sol_pde_11"] = (
        [
            sp.Eq(
                sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x),
                eps * sp.diff(u(x, t), x, 2),
            ),
            sp.Eq(u(x, 0), sp.Piecewise((1, x < 0), (0, True))),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_12"] = (
        sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)),
        u(x, t),
        (x, t),
    )
    examples["sol_pde_13"] = (
        [
            sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)),
            sp.Eq(u(x, 0), sp.exp(-(x**2))),
            sp.Eq(sp.diff(u(x, t), t).subs(t, 0), 1),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_14"] = (
        [
            sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)),
            sp.Eq(u(x, 0), sp.exp(-((x - 6) ** 2)) + sp.exp(-((x + 6) ** 2))),
            sp.Eq(sp.diff(u(x, t), t).subs(t, 0), sp.S(1) / 2),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_15"] = (
        [
            sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2) + m),
            sp.Eq(u(x, 0), sp.sin(x) - sp.cos(3 * x) / sp.exp(sp.Abs(x) / 6)),
            sp.Eq(sp.diff(u(x, t), t).subs(t, 0), 0),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_16"] = (
        [
            sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)),
            sp.Eq(u(0, t), 0),
            sp.Eq(u(sp.pi, t), 0),
            sp.Eq(u(x, 0), sp.sin(m * x)),
            sp.Eq(sp.diff(u(x, t), t).subs(t, 0), 0),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_17"] = (
        [
            sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2)),
            sp.Eq(u(0, t), 0),
            sp.Eq(u(sp.pi, t), 0),
            sp.Eq(u(x, 0), x * (sp.pi - x) ** 2),
            sp.Eq(sp.diff(u(x, t), t).subs(t, 0), 0),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_18"] = (
        sp.Eq(r * sp.diff(u(r, t), t, 2), sp.diff(r * sp.diff(u(r, t), r), r)),
        u(r, t),
        (r, t),
    )
    examples["sol_pde_19"] = (
        [
            sp.Eq(r * sp.diff(u(r, t), t, 2), sp.diff(r * sp.diff(u(r, t), r), r)),
            sp.Eq(u(1, t), 0),
            sp.Eq(u(r, 0), 0),
            sp.Eq(sp.diff(u(r, t), t).subs(t, 0), 1),
        ],
        u(r, t),
        (r, t),
    )
    examples["sol_pde_20"] = (
        sp.Eq(
            12 * sp.diff(u(x, t), x, 2), sp.diff(u(x, t), t, 2) + sp.diff(u(x, t), x, t)
        ),
        u(x, t),
        (x, t),
    )
    examples["sol_pde_21"] = (
        sp.Eq(
            3 * sp.diff(u(x, t), x, 2)
            - sp.diff(u(x, t), t, 2)
            + sp.diff(u(x, t), x, t),
            1,
        ),
        u(x, t),
        (x, t),
    )
    examples["sol_pde_22"] = (
        [
            sp.Eq(
                sp.diff(u(x, y, t), t, 2),
                sp.diff(u(x, y, t), x, 2) + sp.diff(u(x, y, t), y, 2),
            ),
            sp.Eq(u(x, y, 0), sp.S(1) / 10 * (x - x**2) * (2 * y - y**2)),
            sp.Eq(sp.diff(u(x, y, t), t).subs(t, 0), 0),
            sp.Eq(u(x, 0, t), 0),
            sp.Eq(u(0, y, t), 0),
            sp.Eq(u(1, y, t), 0),
            sp.Eq(u(x, 2, t), 0),
        ],
        u(x, y, t),
        (x, y, t),
    )
    examples["sol_pde_23"] = (
        [
            sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
            sp.Eq(u(x, 0), sp.exp(-(x**2))),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_24"] = (
        [
            sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2) + m),
            sp.Eq(u(x, 0), sp.sin(x)),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_25"] = (
        [
            sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
            sp.Eq(u(x, 0), x * (3 - x) ** 2),
            sp.Eq(u(0, t), 0),
            sp.Eq(u(3, t), 0),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_26"] = (
        [
            sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
            sp.Eq(u(x, 0), x * (3 - x)),
            sp.Eq(sp.diff(u(x, t), x).subs(x, 0), 0),
            sp.Eq(sp.diff(u(x, t), x).subs(x, 3), 0),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_27"] = (
        [
            sp.Eq(r * sp.diff(u(r, t), t), sp.diff(r * sp.diff(u(r, t), r), r)),
            sp.Eq(u(r, 0), 1 - r),
            sp.Eq(u(1, t), 0),
        ],
        u(r, t),
        (r, t),
    )
    examples["sol_pde_28"] = (
        [
            sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
            sp.Eq(u(0, t), 20),
            sp.Eq(u(1, t), 50),
            sp.Eq(u(x, 0), 0),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_29"] = (
        [
            sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2)),
            sp.Eq(sp.diff(u(x, t), x).subs(x, 0), 0),
            sp.Eq(sp.diff(u(x, t), x).subs(x, 1), 0),
            sp.Eq(u(x, 0), 20 + 80 * x),
        ],
        u(x, t),
        (x, t),
    )
    examples["sol_pde_30"] = (
        [
            sp.Eq(CaputoD(Y(t, x), (t, sp.S(19) / 10)), sp.diff(Y(t, x), x, 2)),
            sp.Eq(Y(0, x), sp.sin(sp.pi * x)),
            sp.Eq(sp.diff(Y(t, x), t).subs(t, 0), 0),
            sp.Eq(Y(t, 0), 0),
            sp.Eq(Y(t, 1), 0),
        ],
        Y(t, x),
        (t, x),
    )
    examples["sol_pde_31"] = (
        sp.Eq(
            sp.diff(u(x, y), x, 2)
            - 2 * sp.sin(x) * sp.diff(u(x, y), x, y)
            - sp.cos(x) ** 2 * sp.diff(u(x, y), y, 2)
            - sp.cos(x) * sp.diff(u(x, y), y),
            0,
        ),
        u(x, y),
        (x, y),
    )
    examples["sol_pde_32"] = (
        [
            sp.Eq(sp.diff(u(x, t), t), sp.diff(v(x, t), x) + 1),
            sp.Eq(sp.diff(v(x, t), t), -sp.diff(u(x, t), x) - 1),
            sp.Eq(u(x, 0), sp.cos(x) ** 2),
            sp.Eq(v(x, 0), sp.sin(x)),
        ],
        (u(x, t), v(x, t)),
        (x, t),
    )
    examples["sol_pde_33"] = (
        [
            sp.Eq(sp.diff(u(x, y), x), sp.diff(v(x, y), y)),
            sp.Eq(sp.diff(v(x, y), x), -sp.diff(u(x, y), y)),
            sp.Eq(u(x, 0), x**3),
            sp.Eq(v(x, 0), 0),
        ],
        (u(x, y), v(x, y)),
        (x, y),
    )
    examples["sol_pde_34"] = (
        [
            sp.Eq(sp.diff(f(x, y), x), x * y * sp.cos(x * y) + sp.sin(x * y)),
            sp.Eq(sp.diff(f(x, y), y), -sp.exp(-y) + x**2 * sp.cos(x * y)),
        ],
        f(x, y),
        (x, y),
    )
    examples["sol_pde_35"] = (
        [sp.Eq((1 - x) * sp.diff(u(x, y), x), y * u(x, y)), sp.Eq(u(0, y), 1)],
        u(x, y),
        (x, y),
    )
    examples["sol_pde_36"] = (
        sp.Eq(sp.diff(u(x, y), x) + sp.diff(u(x, y), y) ** 2, 3),
        u(x, y),
        (x, y),
    )
    examples["sol_pde_37"] = (
        sp.Eq(sp.diff(u(x, y), x) + 2 * sp.diff(u(x, y), y), 1),
        u(x, y),
        (x, y),
    )
    examples["sol_pde_38"] = (
        [
            sp.Eq(sp.diff(u(x, y), x, 2) + y * sp.diff(u(x, y), y, 2), 0),
            sp.Eq(u(x, 0), 0),
            sp.Eq(sp.diff(u(x, y), y).subs(y, 0), x**2),
        ],
        u(x, y),
        (x, y),
    )
    examples["sol_pde_39"] = (
        [
            sp.Eq(sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2), 0),
            sp.Eq(u(0, y), sp.Piecewise((1, sp.Eq(y, 0)), (sp.sin(y) / y, True))),
        ],
        u(x, y),
        (x, y),
    )
    examples["sol_pde_40"] = (
        [
            sp.Eq(sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2), 0),
            sp.Eq(u(x, 0), x**2 * (1 - x)),
            sp.Eq(u(x, 2), 0),
            sp.Eq(u(0, y), 0),
            sp.Eq(u(1, y), 0),
        ],
        u(x, y),
        (x, y),
    )
    examples["sol_pde_41"] = (
        [
            sp.Eq(
                sp.diff(u(r, theta), r, 2)
                + (1 / r) * sp.diff(u(r, theta), r)
                + (1 / r**2) * sp.diff(u(r, theta), theta, 2),
                0,
            ),
            sp.Eq(u(3, theta), sp.sin(6 * theta)),
        ],
        u(r, theta),
        (r, theta),
    )
    examples["sol_pde_42"] = (
        [
            sp.Eq(
                sp.diff(u(r, theta), r, 2)
                + (1 / r) * sp.diff(u(r, theta), r)
                + (1 / r**2) * sp.diff(u(r, theta), theta, 2),
                0,
            ),
            sp.Eq(u(1, theta), 0),
            sp.Eq(u(2, theta), 5),
        ],
        u(r, theta),
        (r, theta),
    )
    examples["sol_pde_43"] = (
        [
            sp.Eq(sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2), 6 * x - 6 * y),
            sp.Eq(u(x, 0), 1 + 11 * x + x**3),
            sp.Eq(u(x, 2), -7 + 11 * x + x**3),
            sp.Eq(u(0, y), 1 - y**3),
            sp.Eq(u(4, y), 109 - y**3),
        ],
        u(x, y),
        (x, y),
    )
    examples["sol_pde_44"] = (
        [
            sp.Eq(sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2) + 5 * u(x, y), 0),
            sp.Eq(u(x, 0), G(x)),
            sp.Eq(u(x, 2), 0),
            sp.Eq(u(0, y), 0),
            sp.Eq(u(4, y), 0),
        ],
        u(x, y),
        (x, y),
    )
    examples["sol_pde_45"] = (
        [
            sp.Eq(sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2), 0),
            sp.Eq(u(x, 0), 0),
            sp.Eq(u(x, 1), 0),
            sp.Eq(u(0, y), y**2 - y),
            sp.Eq(u(1, y), 0),
        ],
        u(x, y),
        (x, y),
    )
    examples["sol_pde_46"] = (
        [
            sp.Eq(
                sp.diff(v(t, s), t)
                + sp.S.Half * sigma**2 * s**2 * sp.diff(v(t, s), s, 2)
                + (r - q) * s * sp.diff(v(t, s), s)
                - r * v(t, s),
                0,
            ),
            sp.Eq(v(T, s), psiF(s)),
        ],
        v(t, s),
        (t, s),
    )
    examples["sol_pde_47"] = (
        [
            sp.Eq(
                sp.diff(C(t, s), t)
                + sp.S.Half * sigma**2 * s**2 * sp.diff(C(t, s), s, 2)
                + r * s * sp.diff(C(t, s), s)
                - r * C(t, s),
                0,
            ),
            sp.Eq(C(T, s), sp.Max(s - k, 0)),
        ],
        C(t, s),
        (t, s),
    )
    examples["sol_pde_48"] = (
        [
            sp.Eq(
                sp.I * hbar * sp.diff(Psi(x, t), t),
                -(hbar**2 / (2 * m)) * sp.diff(Psi(x, t), x, 2),
            ),
            sp.Eq(Psi(a, t), 0),
            sp.Eq(Psi(b, t), 0),
        ],
        Psi(x, t),
        (x, t),
    )
    examples["sol_pde_49"] = (
        [
            sp.Eq(sp.I * sp.diff(Psi(x, t), t), -2 * sp.diff(Psi(x, t), x, 2)),
            sp.Eq(Psi(5, t), 0),
            sp.Eq(Psi(10, t), 0),
            sp.Eq(Psi(x, 2), phi0(x)),
        ],
        Psi(x, t),
        (x, t),
    )
    examples["sol_pde_50"] = (
        [
            sp.Eq(
                sp.I * sp.diff(Psi(x, t), t),
                -sp.diff(Psi(x, t), x, 2) + 2 * x**2 * Psi(x, t),
            ),
            sp.Eq(Psi(-sp.oo, t), 0),
            sp.Eq(Psi(sp.oo, t), 0),
        ],
        Psi(x, t),
        (x, t),
    )
    examples["sol_pde_51"] = (
        [
            sp.Eq(sp.diff(f(x, y), x), x * y * sp.cos(x * y) + sp.sin(x * y)),
            sp.Eq(sp.diff(f(x, y), y), -sp.exp(-y) + x**2 * sp.cos(x * y)),
        ],
        f(x, y),
        (x, y),
    )
    examples["sol_pde_52"] = (
        sp.Eq(
            sp.diff(u(x, t), t)
            + sp.diff(u(x, t), x, 3)
            + 6 * u(x, t) * sp.diff(u(x, t), x),
            0,
        ),
        u(x, t),
        (x, t),
    )
    return examples


EXAMPLES = _build_examples()


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_reference_pde_example_runs(name):
    eq_or_expr, dep, vars_ = EXAMPLES[name]
    res = pdesolve(eq_or_expr, dep, vars_)
    assert res is not None
    assert getattr(res, "solution", None) is not None
    assert getattr(res, "method", None)


def test_reference_quasilinear_ivp_routes_to_implicit_characteristics():
    eq_or_expr, dep, vars_ = EXAMPLES["sol_pde_08"]
    res = pdesolve(eq_or_expr, dep, vars_)
    assert res.method == "quasilinear_implicit_characteristics"
