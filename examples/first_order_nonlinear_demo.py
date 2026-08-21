from __future__ import annotations

import sympy as sp
import pdesolve as rle


def main() -> None:
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")

    eq = sp.Eq(sp.diff(u(x, t), t) + u(x, t) * sp.diff(u(x, t), x), 0)

    print("=" * 80)
    print("First-order nonlinear PDE analysis demo")
    print("=" * 80)
    analysis = rle.analyze_first_order_nonlinear_pde(eq, u(x, t), (x, t))
    print("Is first order:", analysis.is_first_order)
    print("Is quasilinear:", analysis.is_quasilinear)
    print("Burgers family:", analysis.burgers_family)
    print("Recommended methods:", analysis.recommended_methods)

    print("\n" + "=" * 80)
    print("Canonical auto-dispatch summary")
    print("=" * 80)
    record = rle.pdesolve(eq, u(x, t), (x, t), method="auto")
    print(rle.summarize_solution_record(record))
    trace = rle.extract_solution_trace(record)
    for step in trace.steps:
        print(
            f"[{step.stage}] {step.method}: success={step.success} message={step.message}"
        )

    print("\n" + "=" * 80)
    print("Direct first-order nonlinear auto solver")
    print("=" * 80)
    result = rle.solve_first_order_nonlinear_auto(eq, u(x, t), (x, t))
    print("Method:", getattr(result, "method", None))
    print("Solution:", getattr(result, "solution", None))


if __name__ == "__main__":
    main()
