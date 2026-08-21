# Symbolic infrastructure

PDESolve uses reusable symbolic infrastructure across solver families rather than duplicating equation-specific logic.

## Representation-aware verification

`pdesolve.result_verification` selects verification logic from the representation being returned rather than forcing every result through the same residual simplifier.

The strategy families are:

- `ClassicalResidualVerifier` for explicit classical solutions;
- `ImplicitSolutionVerifier` for implicit characteristic relations;
- `WeakSolutionVerifier` for shocks, rarefactions, and other weak conservation-law results;
- `KernelVerifier` for Green and fundamental-solution representations;
- `SeriesVerifier` for eigenfunction/series results, where PDE and boundary residuals can be exact even if truncated initial data are approximate;
- `TransformVerifier` for formal Fourier/Laplace/unified-transform representations.

`verify_result(...)` is the public representation-aware entry point. The dispatcher uses this layer when a solver does not already provide method-specific verification.

## General product separation

`separate_product_pde(...)` derives a multiplicative ansatz from the supplied PDE itself instead of relying on a named heat/wave/Laplace template. For a two-variable scalar PDE it substitutes

\[
u(x,t)=X(x)T(t)
\]

and attempts to split the normalized residual into independent-variable pieces. For the heat equation this produces separated equations equivalent to

\[
-\frac{X''}{X}=\lambda,
\qquad
\frac{T'}{T}=-\lambda.
\]

The current general engine deliberately targets the robust two-variable multiplicative case and rejects genuinely coupled expressions rather than pretending they separate.

## Sturm–Liouville subsystem

`SturmLiouvilleProblem` and `SturmLiouvilleSpectrum` provide a first-class representation for regular spectral problems

\[
-(pX')'+qX=\lambda wX.
\]

The initial solver handles constant `p`, `q`, and `w` on finite intervals with homogeneous Dirichlet/Dirichlet or Neumann/Neumann conditions. It returns eigenvalues, eigenfunctions, normalization, orthogonality weight, zero-mode information, and coefficient projection.

Structured interval separation now attaches this spectral object to sine/cosine eigenfunction plans.

## Transform postprocessing

`evaluate_inner_transforms(...)` and `postprocess_transform_result(...)` simplify safe profile transforms while intentionally leaving outer inverse transforms unevaluated. This complements the bounded normalization policy: a formal transform is a valid symbolic endpoint, but simple inner sine/cosine/Laplace profile integrals can still be evaluated when inexpensive.

For example, a half-line heat representation can evaluate

\[
\int_0^\infty e^{-\xi}\sin(\omega\xi)\,d\xi
\]

without asking SymPy to simplify the entire nested inverse-transform expression.

## Lie point-symmetry analysis

`analyze_lie_point_symmetries(...)` provides a high-level route from a SymPy PDE to:

1. jet-space conversion;
2. exact prolongation;
3. determining equations;
4. polynomial-ansatz solution of the determining system;
5. a basis of point-symmetry generators.

`invariants_from_point_generator(...)` then constructs local invariants from a supplied point generator using the existing Frobenius-coordinate machinery.

The determining equations are exact; the automatic determining-equation solver is currently restricted to a polynomial ansatz. Full arbitrary functional determining-system solution and optimal-subalgebra classification remain future work.

## Scope and limitations

Symbolic infrastructure is infrastructure, not a claim that arbitrary PDEs are now solvable. In particular:

- general product separation currently supports two independent variables;
- the first-class Sturm–Liouville solver presently covers constant-coefficient regular problems and two common homogeneous boundary pairs;
- transform postprocessing deliberately avoids expensive outer inversion;
- Lie point-symmetry analysis currently solves determining equations with a bounded polynomial ansatz.

These restrictions define the supported scope without weakening reliability.
