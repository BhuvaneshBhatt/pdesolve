# Nonlinear first-order PDEs: implicit characteristics and complete integrals

A useful dividing line is whether characteristics can be eliminated explicitly. For inviscid Burgers,

\[
u_t+u u_x=0,\qquad u(x,0)=x^2,
\]

the characteristic value of $u$ is constant, but the map from the initial coordinate to $(x,t)$ is generally inverted only implicitly.

```python
x, t = sp.symbols("x t", real=True)
u = sp.Function("u")
eq = sp.Eq(sp.diff(u(x,t), t) + u(x,t)*sp.diff(u(x,t), x), 0)
ic = sp.Eq(u(x, 0), x**2)

problem = pds.build_pde_problem(eq, u(x,t), (x,t), ics=ic)
plan = pds.plan_canonical_problem(problem)
print([(s.method, s.score) for s in plan.steps])

result = pds.pdesolve(eq, u(x,t), (x,t),
                      ics={"initial_profile": x**2},
                      method="quasilinear_implicit")
print(type(result).__name__, result.method)
print(result.solution)
```

The concrete method is `quasilinear_implicit_characteristics`. The important semantic point is that an implicit characteristic relation is a legitimate symbolic solution representation; it should not be forced through the same generic simplification pipeline used for an explicit $u=F(x,t)$.

## Generalized Clairaut structure

PDESolve also recognizes specialized nonlinear equations such as

\[
u=xu_x+yu_y+\sin(u_x+u_y),
\]

for which `generalized_clairaut_complete_integral` constructs a parameterized complete integral. This is a different mathematical route from quasilinear characteristic evolution even though both live under the broad “nonlinear first-order” heading.

## Verification

Explicit complete integrals can often be checked by substitution. Implicit characteristic answers instead use method-specific or inconclusive verification when generic symbolic elimination would be inappropriate or prohibitively expensive.

## Limitations

Charpit, complete-integral, Clairaut, and invariant-reduction methods are structural methods, not a decision procedure for arbitrary nonlinear first-order PDEs. Failure to find a complete integral is not a proof that none exists.
