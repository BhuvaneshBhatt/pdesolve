# Capability matrix

This matrix is derived from the solver registry, recognizers, focused solver APIs, and regression suite. The current execution registry contains 35 canonical method keys. “Auto” means the canonical planner/executor can route the family in at least the documented cases; “Direct” means a focused public or module-level solver API exists.

| PDE/problem family | Representative support | Auto | Direct | Evidence/status | Important boundaries |
|---|---|---:|---:|---|---|
| Linear first-order scalar PDEs | characteristic invariants, adapted coordinates, variable/constant coefficients | Yes | Yes | **Verified path** | strongest for two independent variables |
| Transport IVPs | constant-coefficient profile propagation | Yes | Via classical API | **Verified path** | profile-style initial data expected |
| Quasilinear first-order PDEs | characteristic ODEs; implicit IVP solution; nonautonomous formal characteristic systems | Yes | Yes | **Verified path** | implicit answers may not receive generic symbolic verification |
| General nonlinear first-order PDEs | Charpit, complete integrals, invariant reduction | Yes | Yes | **Verified/partial** | method applicability depends strongly on algebraic structure |
| Generalized Clairaut first-order PDEs | recognizer, complete integral, singular envelope metadata | Yes | Yes | **Verified path** | specialized structure only |
| Higher-dimensional complete integrals | Jacobi-style complete-integral search | Yes | Yes | **Verified path** | primarily constructive/search based |
| Scalar conservation laws | canonical form, profile propagation, Riemann data, implicit characteristics | Yes | Yes | **Verified path** | currently centered on 1-D scalar laws |
| Shocks/rarefactions | Rankine–Hugoniot, convex-flux Riemann solutions, admissibility/weak checks | Yes/structured | Yes | **Verified path** | entropy logic is not a universal theorem prover |
| Inviscid Burgers | implicit characteristics; shock/rarefaction helpers | Yes | Yes | **Verified path** | classical solution breaks after characteristic crossing; weak solution handling is separate |
| Viscous Burgers | formal Cole–Hopf representation | Limited | Module API | **Formal/partial** | formal representation, not a general viscous conservation-law solver |
| 1-D wave IVP on full line | d’Alembert solution with displacement/velocity data | Yes | Module API | **Verified path** | canonical 1-D wave structure |
| 1-D wave on finite interval | Dirichlet eigenfunction series; some mixed-series helpers | Yes | Module API | **Verified path** for Dirichlet | mixed boundary support is narrower than Dirichlet |
| 1-D heat IVP on full line | heat kernel; Fourier transform | Yes | Yes/module API | **Verified path** | requires recognizable heat-like structure/profile data |
| 1-D heat on finite interval | Dirichlet, Neumann, Robin eigenfunction series | Yes | Module API | **Verified path** | supported homogeneous/structured boundary patterns are narrower than arbitrary BCs |
| 1-D heat on half-line | sine/cosine transform routes; image/Green kernels | Yes | Yes/module API | **Verified path** | automatic routing supports recognized homogeneous Dirichlet/Neumann half-line data; direct APIs provide formulation control |
| Laplace/Poisson on rectangles | separation/Dirichlet series; selected polynomial-forcing reference cases | Yes | Module API | **Verified path** | arbitrary nonseparable boundary data are not generally solved |
| Laplace in disks/annuli/polar form | geometry recognition and selected separation/reference examples | Partial | Framework APIs | **Supported/partial** | less systematic than rectangle support |
| Helmholtz | classification/separation; Green-function constructions for selected geometries | Partial | Yes | **Verified/direct** | general Helmholtz BVPs are not universally solved |
| Constant-coefficient linear PDEs | operator symbol, factorization, homogeneous families, inverse operator, resonant forcing, data fitting | Yes | Module API | **Verified path** | forcing/data fitting depends on supported symbolic ansatz families |
| Fourier/Laplace transform workflows | heat/wave transform constructions; structured transform planner | Yes | Module API | **Verified/formal mix** | some named transform methods intentionally return formal representations |
| Unified transform | whole-line and half-line evolution PDE workflows, including Schrödinger-like recognition | Yes | Yes | **Verified path** | domain/data combinations are specialized; some results may be formal |
| Linear hyperbolic systems | canonical extraction, diagonalization/characteristic variables, initial data | Yes | Yes | **Verified path** | assumes a supported diagonalizable linear first-order form |
| Symmetry/invariant reduction | Lie/invariant reduction and post-reduction solving | Yes | Module APIs | **Verified/partial** | reduction may end in `UnsolvedButReducedResult` rather than a closed form |
| Fundamental solutions | heat, wave, Laplace in several dimensions/geometries | Source-driven auto | Yes | **Verified path** | auto kernel routing is intentionally conservative |
| Green functions | half-lines/half-planes/half-spaces, intervals, strips, rectangles, selected Helmholtz and ODE cases | Source-driven/partial | Yes | **Verified path** | coverage is geometry/operator specific, not arbitrary-domain Green-function synthesis |
| Linear ODE Green functions | dedicated Green-function API | No | Yes | **Verified direct API** | auxiliary capability, not PDE auto routing |
| General second-order type classification | elliptic/parabolic/hyperbolic and higher-dimensional signatures | Recognition | Yes | **Verified recognition** | classification does not guarantee a solver exists |
| Fractional-time PDE reference forms | parser/reference-suite execution | Limited | No general API | **Experimental/reference** | no general fractional PDE solver is claimed |
| Black–Scholes / Schrödinger-like reference PDEs | recognized/routed through general constant-coefficient, transform, separation or reference fallbacks where applicable | Partial | Specialized helpers only | **Reference/partial** | not domain-specific finance/quantum packages |

## How to interpret this table

A tested reference example demonstrates that a path executes; it does not establish completeness for the entire mathematical family. Conversely, “partial” often means PDESolve has useful canonicalization, reduction, or formal representations even when it cannot finish every instance.

For exact method keys and their handlers, see [Methods and solver inventory](method-inventory.md). For known routing and verification boundaries, see [Limitations](limitations.md).


## From matrix to mathematics

For representative equations from each major family, continue to the [worked examples](examples/index.md). Each example shows recognition evidence, planning, concrete result identity, verification semantics, and the boundary of the demonstrated capability.
