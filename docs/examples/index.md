# Worked mathematical examples

worked examples turns the capability matrix into worked examples. Each chapter answers the same six questions:

1. **What PDE is being solved?**
2. **What structural facts does PDESolve recognize?**
3. **What does the generated plan look like?**
4. **What result type and concrete method should the caller expect?**
5. **What does verification mean for this family?**
6. **Where does the demonstrated capability stop?**

The examples deliberately separate a *planner label* from a *concrete result method*. A planner can select a framework or family and the executor can return a more specific method name.

## Example map

| Chapter | Main family | Central idea |
|---|---|---|
| [Linear first order](linear-first-order.md) | transport/linear characteristics | characteristic propagation |
| [Nonlinear first order](nonlinear-first-order.md) | quasilinear, Clairaut | implicit characteristics and complete integrals |
| [Conservation laws](conservation-laws.md) | scalar 1-D laws | shocks, rarefactions, weak solutions |
| [Heat equations](heat-equations.md) | parabolic evolution | kernels and eigenfunction series |
| [Wave equations](wave-equations.md) | hyperbolic evolution | d'Alembert and interval series |
| [Elliptic/separation problems](elliptic-separation.md) | Laplace/Poisson/Helmholtz | geometry + boundary-adapted bases |
| [Constant-coefficient PDEs](constant-coefficient.md) | linear operator algebra | symbols, factorization, particular solutions |
| [Transform methods](transform-methods.md) | Fourier/Laplace/unified transform | spectral evolution and formal representations |
| [Hyperbolic systems](hyperbolic-systems.md) | linear first-order systems | diagonalization and characteristic variables |
| [Symmetry reduction](symmetry-reduction.md) | Lie/invariant methods | reduce a PDE before solving |
| [Fundamental solutions and Green functions](kernels-green.md) | source-driven kernels | operator + geometry + source structure |

All snippets assume `import sympy as sp` and `import pdesolve as pds`. See [Getting started](../getting-started.md) for installation.
