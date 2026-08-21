# Fundamental solutions and Green functions

Kernel methods are source- and geometry-driven. For the heat operator

\[
\partial_t-a\partial_{xx},\qquad a>0,
\]

the whole-line fundamental solution is the Gaussian heat kernel with causal time support.

```python
x, t, a = sp.symbols("x t a", positive=True)
u = sp.Function("u")
eq = sp.Eq(sp.diff(u(x,t),t) - a*sp.diff(u(x,t),x,2), 0)

result = pds.solve_fundamental_solution(eq, u(x,t), (x,t))
print(type(result).__name__, result.method)
print(result.solution)
```

The result type is `FundamentalSolutionResult` and the method is `kernel_fundamental_solution`.

For a half-line with homogeneous Dirichlet boundary data, `solve_green_function(..., geometry="half_line")` constructs the image-kernel combination. Laplace half-plane Green functions use the analogous image geometry and logarithmic 2-D fundamental solution. Interval wave Green functions can be returned as symbolic eigenfunction sums.

A PDE containing an explicit Dirac source can also trigger conservative automatic kernel routing:

```python
xi, tau, L = sp.symbols("xi tau L", positive=True)
forced = sp.Eq(
    sp.diff(u(x,t),t) - a*sp.diff(u(x,t),x,2),
    sp.DiracDelta(x-xi)*sp.DiracDelta(t-tau),
)
result = pds.pdesolve(
    forced, u(x,t), (x,t),
    bcs=(sp.Eq(u(0,t),0), sp.Eq(u(L,t),0)),
)
print(result.method)  # kernel_green_function
```

## Verification

Kernel verification is distributional/source-aware: the defining operator applied to a fundamental solution should reproduce the source distribution, together with the required boundary behavior for a Green function. This is conceptually different from checking an ordinary smooth residual everywhere.

## Limitations

Coverage is operator- and geometry-specific. PDESolve has tested kernels for several half-line/half-plane/half-space, interval, strip, rectangle, Helmholtz, and ODE cases, but it does not construct Green functions for arbitrary domains.
