# Limitations and support boundaries

PDESolve is a growing symbolic toolbox, not a complete decision procedure for PDEs. This page records boundaries visible in the current implementation and tests.

## Automatic routing versus direct support

A helper can be public without being an auto-planner target. Conversely, planner/coordinator labels may delegate to other methods. Always distinguish the canonical execution-registry key from a focused module API.

Half-line heat handling is routed through the canonical `heat_half_line_transform` method. Direct transform or Green-function APIs remain useful when deterministic formulation control is required.

## Geometry is not arbitrary

Finite intervals, rectangles, disks/annuli, half-lines, full lines, half-planes, half-spaces, strips, and selected source geometries are represented in parts of the package. PDESolve does not claim general symbolic Green functions or separation solutions for arbitrary curved domains.

## Boundary conditions are pattern-dependent

Dirichlet/Neumann/Robin families are recognized in structured cases, but arbitrary mixed, nonlocal, nonlinear, or geometrically complicated boundary operators may not map to an executable method.

## Nonlinear first-order methods are constructive

Charpit, complete-integral, Jacobi, and invariant-reduction paths depend on recognizable algebraic/differential structure. Failure to find a complete integral is not a proof none exists.

## Conservation-law scope

The strongest structured support is for one-dimensional scalar conservation laws, including autonomous flux, profile/Riemann data, shocks, rarefactions, and implicit characteristics. Multi-dimensional systems of conservation laws and general entropy-solution theory are outside the current documented scope.

## Formal transforms are not closed forms

Some Laplace/Fourier/unified-transform and Cole–Hopf paths intentionally return formal integral/transform representations. The package may not invert every transform symbolically.

## Verification can be inconclusive

Symbolic residual simplification can fail, be expensive, or be mathematically inappropriate for implicit/weak/distributional results. PDESolve therefore distinguishes verified, method-specific, and unverified/inconclusive outcomes.

## Systems

The hyperbolic-system solver targets supported linear first-order systems with extractable canonical matrices and diagonalizable/usable characteristic structure. General nonlinear systems are not claimed.

## Fractional and domain-specific equations

The reference suite includes unusual equations, including a fractional-time example and finance/quantum-style PDEs. Their presence demonstrates parser/routing robustness for those examples; it does not establish a general fractional-calculus, quantitative-finance, or quantum-PDE subsystem.

## Numerical solving

`NumericalFallbackResult` exists in the result model, but the current package is principally symbolic. Do not interpret the class name as evidence of a comprehensive numerical PDE backend.
