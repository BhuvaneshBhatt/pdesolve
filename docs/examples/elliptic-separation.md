# Elliptic and separable boundary-value problems

For Laplace's equation on a rectangle,

\[
u_{xx}+u_{yy}=0,
\]

boundary geometry is as important as differential classification. PDESolve's separation framework builds a geometry/condition plan and chooses a basis adapted to the boundary conditions.

```python
x, y = sp.symbols("x y", real=True)
u = sp.Function("u")
eq = sp.Eq(sp.diff(u(x,y),x,2) + sp.diff(u(x,y),y,2), 0)

problem = pds.build_pde_problem(eq, u(x,y), (x,y))
plan = pds.plan_canonical_problem(problem)
print([(s.method, s.score) for s in plan.steps[:4]])

from pdesolve.classical import solve_rectangle_dirichlet_laplace_series
result = solve_rectangle_dirichlet_laplace_series(
    u(x,y), x=x, y=y, boundary_top=x*(sp.pi-x), terms=5
)
print(type(result).__name__, result.method)
print(result.solution)
```

The generic plan shows which broad methods recognize the differential structure, while the focused rectangle API supplies the geometry-specific construction. Its concrete method is `laplace_rectangle_dirichlet_series`. This is an intentional example of a capability that is available through a more specialized direct API than through automatic routing. Internally, the separation framework records geometry and a boundary-adapted eigenbasis rather than treating “separation of variables” as an unstructured guess.

A particularly useful diagnostic feature is condition consistency analysis. For a 2-D heat equation on a rectangle, supplying boundaries only at the two $x$ edges is recognized as an incomplete rectangular boundary specification rather than silently pretending that a full BVP was supplied.

## Verification

For explicit finite/closed expressions PDESolve can check the elliptic PDE residual and supported boundary traces. Infinite symbolic series may require structural or conditional verification rather than full simplification.

## Limitations

Rectangle Dirichlet support is substantially more systematic than arbitrary-domain elliptic solving. Disk/annulus/polar and Helmholtz support exists for selected structures, but the package does not synthesize solutions for arbitrary geometries or arbitrary boundary functions.
