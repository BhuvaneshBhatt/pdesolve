from __future__ import annotations

import sympy as sp
import pdesolve as pds


def main() -> None:
    x, y, z, t = sp.symbols("x y z t", real=True)
    u = sp.Function("u")

    print("=" * 78)
    print("Auto planner")
    print("=" * 78)
    eq_auto = sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0)
    print(pds.pdesolve(eq_auto, u(x, t), (x, t), method="auto"))

    print("\n" + "=" * 78)
    print("Constant-coefficient inverse differential operator")
    print("=" * 78)
    ic_eq = sp.Eq(u(x, 0), sp.exp(-(x**2)))
    print(
        pds.pdesolve(
            eq_auto,
            u(x, t),
            (x, t),
            method="constant_coefficient_inverse_operator",
            ics={
                "equation": ic_eq,
                "initial_profile": sp.exp(-(x**2)),
                "curve_value": 0,
            },
        )
    )

    print("\n" + "=" * 78)
    print("Symmetry reduction")
    print("=" * 78)
    print(pds.pdesolve(eq_auto, u(x, t), (x, t), method="symmetry_reduction"))

    print("\n" + "=" * 78)
    print("Charpit")
    print("=" * 78)
    eq_char = sp.Eq(sp.diff(u(x, y), x) ** 2 + sp.diff(u(x, y), y) ** 2, 1)
    print(pds.pdesolve(eq_char, u(x, y), (x, y), method="charpit"))

    print("\n" + "=" * 78)
    print("Complete integral")
    print("=" * 78)
    eq_ci = sp.Eq(sp.diff(u(x, y), x) ** 2 + sp.diff(u(x, y), y), 0)
    print(pds.pdesolve(eq_ci, u(x, y), (x, y), method="complete_integral"))

    print("\n" + "=" * 78)
    print("Invariant reduction")
    print("=" * 78)
    eq_inv = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)
    print(pds.pdesolve(eq_inv, u(x, t), (x, t), method="invariant_reduction_auto"))

    print("\n" + "=" * 78)
    print("Unified transform")
    print("=" * 78)
    eq_s = sp.Eq(sp.I * sp.diff(u(x, t), t) + sp.diff(u(x, t), x, 2), 0)
    bc_eqs = [sp.Eq(u(0, t), 0)]
    print(
        pds.pdesolve(
            eq_s,
            u(x, t),
            (x, t),
            method="unified_transform",
            ics=ic_eq,
            bcs={"equations": bc_eqs},
            domain="half_line",
        )
    )

    print("\n" + "=" * 78)
    print("Hyperbolic system")
    print("=" * 78)
    u1 = sp.Function("u1")
    u2 = sp.Function("u2")
    eqs = [
        sp.Eq(sp.diff(u1(t, x), t), sp.diff(u2(t, x), x)),
        sp.Eq(sp.diff(u2(t, x), t), sp.diff(u1(t, x), x)),
    ]
    ics_sys = [sp.Eq(u1(0, x), sp.sin(x)), sp.Eq(u2(0, x), sp.cos(x))]
    print(pds.pdesolve(eqs, (u1, u2), (t, x), method="hyperbolic_system", ics=ics_sys))


if __name__ == "__main__":
    main()
