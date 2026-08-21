# Symmetry and invariant reduction

Symmetry methods are different from direct solvers: their first goal is to reduce the number of independent variables. A reduced ODE can then be handed to a post-solver.

```python
x, t = sp.symbols("x t", real=True)
u = sp.Function("u")
eq = sp.Eq(sp.diff(u(x,t),t) + sp.diff(u(x,t),x), 0)

result = pds.pdesolve(
    eq, u(x,t), (x,t), method="auto",
    prefer_symmetry=True, max_symmetry_steps=1,
)
print(type(result).__name__, result.method)
print(result.solution)
```

For a simple transport equation a direct first-order method may still outrank or replace symmetry reduction; tests intentionally accept either a successful `symmetry_reduction_plus_postsolve` result or a strong direct first-order result. This illustrates that `prefer_symmetry=True` is a planning preference, not a command to ignore a simpler exact route.

The lower-level reduction APIs expose invariant coordinates and reduced equations. `solve_reduced_equation_auto` can solve supported reduced ODEs using SymPy's ODE machinery.

## Verification

When reduction plus postsolve returns an explicit PDE solution, the final expression can be checked against the original PDE. When only a reduction succeeds, PDESolve may return an `UnsolvedButReducedResult`: that is useful mathematical progress but is not a solved PDE.

## Limitations

The symmetry subsystem does not claim complete Lie-algebra classification for arbitrary PDE systems. Finding no usable reduction under the configured search budget is not proof that the PDE has no symmetries.
