from __future__ import annotations

import sympy as sp

import pdesolve as rle


def main() -> None:
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")

    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)

    print("=" * 80)
    print("Invariant-reduction candidate enumeration demo")
    print("=" * 80)
    candidates = rle.enumerate_invariant_reduction_candidates(eq, u(x, t), (x, t))
    print("Number of candidates:", len(candidates))
    for i, cand in enumerate(candidates, 1):
        print(f"Candidate {i}: method={cand.method} score={cand.score}")
        print("  invariants =", cand.invariants)
        print("  transverse_parameters =", cand.transverse_parameters)
        print("  reduced_equation =", cand.reduced_equation)
        print("  ansatz =", cand.ansatz)

    print("\n" + "=" * 80)
    print("Invariant-reduction solve")
    print("=" * 80)
    result = rle.solve_via_invariant_reduction(eq, u(x, t), (x, t))
    print("Method:", getattr(result, "method", None))
    print("Solution:", getattr(result, "solution", None))
    print("Details keys:", sorted(getattr(result, "metadata", {}).keys()))


if __name__ == "__main__":
    main()
