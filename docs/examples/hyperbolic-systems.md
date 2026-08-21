# Linear hyperbolic systems: characteristic variables

Consider the diagonal system

\[
u_t=u_x,\qquad v_t=-v_x,
\]

with $u(x,0)=\sin x$ and $v(x,0)=\cos x$. The two components propagate in opposite characteristic directions.

```python
x, t = sp.symbols("x t", real=True)
u, v = sp.Function("u"), sp.Function("v")
eqs = [
    sp.Eq(sp.diff(u(x,t),t),  sp.diff(u(x,t),x)),
    sp.Eq(sp.diff(v(x,t),t), -sp.diff(v(x,t),x)),
]
ics = [sp.Eq(u(x,0), sp.sin(x)), sp.Eq(v(x,0), sp.cos(x))]

canonical = pds.extract_canonical_linear_system_form(eqs, (u,v), (x,t))
print(canonical.diagonalizable)

result = pds.solve_hyperbolic_system(eqs, ics, (u,v), vars=(x,t))
print(type(result).__name__, result.system_size)
print(result.metadata["solver"])
print(result.solution)
```

The canonical extractor identifies the matrix multiplying spatial derivatives and records whether it is diagonalizable. The solver then uses characteristic diagonalization or a matrix-exponential route; the result records which strategy was used.

## Verification

For an explicit vector solution, verification means substituting all components back into all system equations and checking the initial vector data. Metadata about diagonalization is explanatory evidence, not a substitute for residual checking.

## Limitations

The demonstrated subsystem is linear and first order with a supported diagonalizable canonical form. It should not be read as general support for nonlinear hyperbolic systems, shocks in systems, or arbitrary variable-coefficient systems.
