# Transform methods: Fourier, Laplace, and the unified transform

Transforms are useful when evolution in one variable becomes algebraic or an ODE after transforming the spatial coordinate. PDESolve distinguishes concrete transform solvers from formal transform representations.

For the half-line Schrödinger-like equation

\[
i u_t+u_{xx}=0,\quad x>0,\qquad u(0,t)=0,
\]

```python
x, t = sp.symbols("x t", positive=True, real=True)
u = sp.Function("u")
eq = sp.Eq(sp.I*sp.diff(u(x,t),t) + sp.diff(u(x,t),x,2), 0)
ic = sp.Eq(u(x,0), sp.sin(x))
bc = [sp.Eq(u(0,t), 0)]

profile = pds.recognize_evolution_pde(eq, u, (x,t))
print(profile.family_name)  # schrodinger_like

result = pds.solve_unified_transform(
    eq, u, (x,t), initial_condition=ic,
    boundary_conditions=bc, domain="half_line"
)
print(type(result).__name__, result.method_family, result.is_formal)
```

The tested route reports `unified_transform_half_line_schrodinger_dirichlet_zero` and is not merely a formal placeholder for this supported case.

## Planning and result semantics

The automatic planner also has a `structured_transform` framework candidate. A framework label can delegate to Fourier, Laplace, or unified-transform machinery. Inspect both `result.method`/`method_family` and formal-status metadata rather than assuming that all transform results have the same semantic strength.

## Verification

A closed transformed-and-inverted expression can be checked conventionally. A formal transform representation may instead be verified structurally: the transform equation, transformed data, or representation is sound even if symbolic inversion is intentionally left unevaluated.

## Limitations

Transform conventions, domains, and boundary data matter. The unified transform support is specialized rather than a complete implementation for arbitrary linear PDEs on arbitrary polygonal domains.
