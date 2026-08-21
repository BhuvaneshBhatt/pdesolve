# Getting started

## Installation

From the repository root:

```bash
pip install -e .
```

PDESolve currently requires Python 3.10+ and SymPy 1.12+.

## Your first solve

```python
import sympy as sp
import pdesolve as pds

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")

eq = sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0)
result = pds.pdesolve(eq, u(x, t), (x, t), method="auto")

print(result.method)
print(result.solution)
print(result.verification)
```

`method="auto"` builds a canonical problem profile, recognizes structure, parses conditions and geometry, ranks candidate methods, executes a candidate, standardizes the result, and attaches verification/trace metadata where available.

## Supplying initial and boundary data

```python
ic = sp.Eq(u(x, 0), sp.exp(-x**2))
eq = sp.Eq(sp.diff(u(x, t), t), sp.diff(u(x, t), x, 2))

result = pds.pdesolve([eq, ic], u(x, t), (x, t), method="auto")
```

For finite intervals, boundary equations can be bundled with the PDE or supplied through `ics=` / `bcs=`. The structured parser converts supported equations to `ConditionModel` records and then infers a `DomainGeometry` where possible.

## Inspect before solving

```python
problem = pds.build_pde_problem([eq, ic], u(x, t), (x, t))
recognitions = pds.recognize_canonical_problem(problem)
plan = pds.plan_canonical_problem(problem)

for step in plan.steps:
    print(step.method, step.score, step.reasons)
```


## Request a method explicitly

```python
result = pds.pdesolve(
    [eq, ic],
    u(x, t),
    (x, t),
    method="fourier_heat",
)
```

Explicit methods use the common execution/standardization layer. Not every focused helper in submodules is an automatic method key; see the [method inventory](method-inventory.md).

## Read the result conservatively

A nonempty symbolic expression does not imply that the solution has been fully verified. Inspect `result.method`, `result.verification`, `result.warnings`, and `result.metadata`. Implicit and weak solutions may intentionally report method-specific or inconclusive verification rather than trigger expensive generic simplification.
