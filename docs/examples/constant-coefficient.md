# Constant-coefficient linear PDEs: operator algebra

Consider

\[
u_t-u_x=e^{x+t}.
\]

The constant-coefficient subsystem represents the differential operator by its polynomial symbol, constructs a homogeneous family, finds a particular solution, and combines the two.

```python
x, t = sp.symbols("x t", real=True)
u = sp.Function("u")
eq = sp.Eq(sp.diff(u(x,t), t) - sp.diff(u(x,t), x), sp.exp(x+t))

from pdesolve.classical import pdesolve_constant_coefficient

result = pdesolve_constant_coefficient(eq, u(x,t), (x,t))
print(type(result).__name__, result.method)
print(result.solution)
print(result.metadata["method_family_report"])
```

The method-family report distinguishes the selected outer method, homogeneous construction, and particular-solution engine. Supported forcing engines include exponential and exponential-amplitude forms, with additional handling for polynomial, trigonometric/hyperbolic, products, resonance, and fitted conditions.

## Planning

In the canonical planner, a constant-coefficient inverse-operator candidate can rank highly whenever the operator and forcing match this algebraic structure. It may coexist with characteristics, transforms, or separation candidates; the scores express applicability/preference rather than mathematical uniqueness.

## Verification

This family has a strong exact residual check: apply the detected differential operator to the constructed expression and subtract the forcing. Condition fitting has an additional linear-algebra layer and reports its fitting strategy in result metadata.

## Limitations

The subsystem is symbolic and ansatz-driven. A constant-coefficient PDE can still have data or forcing outside the supported closed symbolic families. Recognition of the operator does not guarantee that all imposed conditions can be fitted.
