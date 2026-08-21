# Scalar conservation laws: shocks and rarefactions

For

\[
u_t + f(u)_x=0,\qquad f(u)=\frac{u^2}{2},
\]

Riemann data with $u_L=1$, $u_R=0$ produce a shock. The Rankine–Hugoniot speed is

\[
s=\frac{f(u_L)-f(u_R)}{u_L-u_R}=\frac12.
\]

```python
u0 = sp.symbols("u0", real=True)
x, t = sp.symbols("x t", positive=True)

from pdesolve.classical import solve_scalar_conservation_law_riemann_general

result = solve_scalar_conservation_law_riemann_general(
    u0**2/2, 1, 0, x=x, t=t, u_symbol=u0
)
print(type(result).__name__, result.method)
print(result.solution)
```

The tested concrete method is `scalar_conservation_riemann_shock`. Reversing the states for a convex flux selects a rarefaction branch instead.

## Recognition and planning

The conservation-law recognizer first tries to write a PDE in canonical flux/source form. Structured routing can then distinguish profile propagation, an implicit characteristic solution, and Riemann shock/rarefaction cases. This is why “conservation law” is a family, not a single algorithm.

## Verification

Classical residual substitution is insufficient at a discontinuity. PDESolve therefore has separate weak/admissibility machinery: Rankine–Hugoniot checks jump propagation, while entropy selection distinguishes the physically admissible branch for supported scalar convex-flux cases.

## Limitations

The strongest support is for **one-dimensional scalar** conservation laws. Entropy selection is not a universal entropy-solution theorem prover, and multidimensional systems of conservation laws are outside the demonstrated capability.
