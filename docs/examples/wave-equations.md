# Wave equations: d'Alembert on the full line

For wave speed $c=1$,

\[
u_{tt}=u_{xx},\qquad u(x,0)=\sin x,\qquad u_t(x,0)=0,
\]

d'Alembert's formula gives the solution as the sum of left- and right-moving waves.

```python
x, t = sp.symbols("x t", real=True)
u = sp.Function("u")
eq = sp.Eq(sp.diff(u(x,t), t, 2), sp.diff(u(x,t), x, 2))
ics = {"initial_displacement": lambda z: sp.sin(z),
       "initial_velocity": lambda z: 0}

from pdesolve.classical import rank_pde_solution_methods

profile, candidates = rank_pde_solution_methods(eq, u(x,t), (x,t), ics=ics)
print([(c.method, c.score) for c in candidates[:4]])

result = pds.pdesolve(eq, u(x,t), (x,t), ics=ics, method="auto")
print(type(result).__name__, result.method)
print(result.solution)
print(result.verification)
```

The planner places `wave_dalembert` first; the standardized concrete result uses `dAlembert_wave_ivp`. For explicit initial data, symbolic verification can check the PDE, displacement condition, and velocity condition.

On a finite interval with homogeneous Dirichlet boundaries, the preferred representation changes to a sine eigenfunction series (`wave_dirichlet_sine_series`). Thus the planner is using both differential structure and geometry.

## Limitations

The demonstrated d'Alembert route is the canonical one-dimensional constant-speed full-line IVP. Variable wave speed, higher-dimensional domains, or general nonhomogeneous boundaries require other reductions, kernels, transforms, or may remain unsupported.
