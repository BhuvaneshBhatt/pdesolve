import sympy as sp

from pdesolve.classical import (
    pdesolve,
    extract_solution_trace,
    summarize_solution_record,
)


def test_trace_and_verification_present():
    x, t = sp.symbols("x t", real=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2))
    rec = pdesolve(
        eq,
        u(x, t),
        (x, t),
        ics={"initial_displacement": sp.sin(x), "initial_velocity": 0},
        method="auto",
    )
    trace = extract_solution_trace(rec)
    assert trace is not None
    assert trace.attempted_methods
    assert "verified" in rec.verification
    summary = summarize_solution_record(rec)
    assert summary["method"] == rec.method
