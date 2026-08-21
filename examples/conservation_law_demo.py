from __future__ import annotations

import sympy as sp
import pdesolve as rle


def main() -> None:
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")

    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)

    print("=" * 80)
    print("Conservation-law / Burgers-family recognition demo")
    print("=" * 80)
    analysis = rle.analyze_first_order_nonlinear_pde(eq, u(x, t), (x, t))
    print("Burgers family:", analysis.burgers_family)
    print("Recommended methods:", analysis.recommended_methods)

    print("\n" + "=" * 80)
    print("Explicit conservation-law method")
    print("=" * 80)
    record = rle.pdesolve(eq, u(x, t), (x, t), method="conservation_law")
    print("Method:", getattr(record, "method", None))
    print("Solution:", getattr(record, "solution", None))
    print("Summary:", rle.summarize_solution_record(record))
    for step in rle.extract_solution_trace(record).steps:
        print(
            f"[{step.stage}] {step.method}: success={step.success} message={step.message}"
        )


if __name__ == "__main__":
    main()
