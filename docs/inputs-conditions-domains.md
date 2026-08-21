# Inputs, conditions, and domains

## Equation inputs

The main solver accepts a SymPy equation/expression or, for supported workflows, a sequence bundling the PDE with initial/boundary equations. `build_pde_problem(...)` is the recommended way to inspect how an input was normalized.

## Dependent and independent variables

Be explicit about both the dependent expression and independent-variable order:

```python
x, t = sp.symbols("x t")
u = sp.Function("u")
problem = pds.build_pde_problem(eq, u(x, t), (x, t))
```

For systems, dependent variables may be supplied as a tuple/list of functions/expressions depending on the focused API. Coordinate-role inference exists, but consistent ordering reduces ambiguity.

## `ConditionModel`

`parse_conditions(...)` normalizes supported conditions into:

- initial conditions,
- boundary conditions,
- mixed conditions,
- event conditions,
- metadata describing constant slices and derivative order.

Each normalized `ConditionEquation` stores the original equality, a role, a variable/location when identifiable, a derivative multi-index, and metadata.

## Domain inference

`infer_domain_geometry(...)` can construct structured geometries including:

- `IntervalDomain`
- `RectangleDomain`
- `DiskDomain`
- `HalfLineDomain`
- `FullLineDomain`
- `PolarAnnulusDomain`

The domains module also contains half-plane/half-space support used by kernel workflows even though those classes are not all re-exported at package top level.

Domain inference is evidence-based: finite boundary locations can imply an interval; infinite endpoints/full-line metadata can imply an unbounded domain; multiple coordinate boundaries can imply a rectangle. Absence of enough data yields a less specific geometry rather than a fabricated domain.

## `BoundaryModel`

`build_boundary_model(...)` binds normalized boundary equations to geometry components. This supports separation, transform, condition analysis, and kernel planning. Boundary models distinguish, where recognizable, Dirichlet, Neumann, Robin, and related condition families.

## Condition diagnostics

Condition consistency is analyzed before some solver choices. The engine can report warnings such as an incompletely specified rectangle boundary. A condition-analysis warning is a diagnostic, not proof that a PDE has no solution.

## Practical guidance

- Prefer exact SymPy objects (`sp.Rational`, symbolic constants) when exact symbolic comparison matters.
- Supply both displacement and velocity data for a classical wave IVP.
- Supply a recognizable initial profile for heat/transport/conservation-law IVPs.
- For finite-interval series methods, provide boundary equations that make the interval and boundary family inferable.
- For kernel/Green-function work, explicit kernel APIs are often clearer than relying on automatic source recognition.
