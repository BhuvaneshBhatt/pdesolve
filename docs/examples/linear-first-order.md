# Linear first-order PDE: transport by characteristics

Consider

\[
u_t+2u_x=0,\qquad u(x,0)=x^2.
\]

Along a characteristic, $x-2t$ is constant, so the exact solution is $u(x,t)=(x-2t)^2$.

```python
import sympy as sp
import pdesolve as pds

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")
eq = sp.Eq(sp.diff(u(x, t), t) + 2*sp.diff(u(x, t), x), 0)
ic = sp.Eq(u(x, 0), x**2)

problem = pds.build_pde_problem(eq, u(x, t), (x, t), ics=ic)
plan = pds.plan_canonical_problem(problem)
for step in plan.steps[:4]:
    print(step.method, step.score)

result = pds.pdesolve([eq, ic], u(x, t), (x, t), method="auto")
print(type(result).__name__, result.method)
print(result.solution)
print(result.verification)
```

For this canonical problem the planner ranks `transport_ivp` first. The standardized result is a `PDESolutionRecord`; the concrete executor may report `constant_coefficient_characteristics_ivp`, because the final method identity describes the solver that actually constructed the answer rather than merely repeating the planner label.

## Why it is recognized

The first-order parser can isolate the coefficients of $u_x$, $u_t$, $u$, and the forcing. Here they are constant and the condition model contains an initial profile on $t=0$. Those facts make characteristic propagation substantially more specific than a generic first-order solve.

## Verification

For this explicit classical solution PDESolve can substitute the answer into both the PDE and initial condition. A successful record therefore has zero PDE residual and zero initial residual and can report symbolic verification as complete.

## Boundary of the example

This does **not** imply that every variable-coefficient first-order PDE has an explicit elementary characteristic invariant. PDESolve also has adapted-coordinate and general first-order routes; those can return arbitrary-function, implicit, reduced, or unresolved representations.
