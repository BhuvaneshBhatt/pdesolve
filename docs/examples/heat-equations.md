# Heat equations: interval eigenfunctions and whole-line kernels

Consider the homogeneous Dirichlet problem

\[
u_t=u_{xx},\quad 0<x<\pi,\qquad u(0,t)=u(\pi,t)=0,
\qquad u(x,0)=x(\pi-x).
\]

```python
x, t = sp.symbols("x t", positive=True, real=True)
u = sp.Function("u")
eq = sp.Eq(sp.diff(u(x,t), t), sp.diff(u(x,t), x, 2))

ics = {"initial_profile": lambda z: z*(sp.pi-z)}
bcs = {"type": "dirichlet_homogeneous_interval", "length": sp.pi, "terms": 5}
problem = pds.build_pde_problem(eq, u(x,t), (x,t), ics=ics, bcs=bcs)
plan = pds.plan_canonical_problem(problem)
print([(s.method, s.score) for s in plan.steps[:4]])

result = pds.pdesolve(eq, u(x,t), (x,t), ics=ics, bcs=bcs, method="auto")
print(type(result).__name__, result.method)
print(result.solution)
```

The planner recognizes a heat-like parabolic PDE, finite-interval geometry, and homogeneous Dirichlet boundaries, so `heat_dirichlet_series` is preferred. The concrete standardized method is `heat_dirichlet_sine_series`.

## Verification nuance: truncated series

With a finite `terms` value, every retained sine mode satisfies the PDE and homogeneous boundary conditions exactly, but a general initial profile is only approximated by the truncated Fourier series. Consequently the PDE and boundary residuals can be zero while the initial-condition verification remains incomplete. This is expected and is a useful example of why `result.verification` is structured rather than a single Boolean.

For the whole line, the same PDE can instead route to a heat-kernel or Fourier representation. The mathematical solution family is the same parabolic evolution problem, but geometry and supplied data change the preferred representation.

## Limitations

Finite-interval support is strongest for structured homogeneous Dirichlet, Neumann, and Robin patterns. Arbitrary time-dependent or nonseparable boundary data are not implied by this example.
