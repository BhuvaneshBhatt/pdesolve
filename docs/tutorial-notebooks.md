# Executable tutorial notebooks — PDESolve documentation

PDESolve documentation turns the worked examples into an executable tutorial curriculum. The maintained notebooks live in `notebooks/tutorials/` and mirror the major solver families in the capability matrix.

The generated [notebook index](notebook-index.md) is the authoritative file inventory.

## Tutorial map

| Notebook | Solver family | Derivation | Planner/recognizer introspection | Independent verification | Visualization |
|---|---|---:|---:|---:|---:|
| `01_linear_first_order.ipynb` | linear first-order / transport | ✓ | ✓ | classical PDE + IC residuals | profile translation |
| `02_nonlinear_first_order.ipynb` | nonlinear first-order | ✓ | ✓ | implicit characteristic identity | parametric characteristics |
| `03_conservation_laws.ipynb` | scalar conservation laws | ✓ | ✓ | Rankine–Hugoniot + entropy inequalities | shock propagation |
| `04_heat_equations.ipynb` | heat / eigenfunction series | ✓ | ✓ | PDE + boundary residuals; truncation nuance | diffusive decay |
| `05_wave_equations.ipynb` | wave / d'Alembert | ✓ | ✓ | PDE + displacement + velocity residuals | wave evolution |
| `06_elliptic_separation.ipynb` | elliptic / rectangle separation | ✓ | ✓ | Laplace + boundary residuals | harmonic contour field |
| `07_constant_coefficient.ipynb` | constant-coefficient operator algebra | ✓ | ✓ | exact particular-solution residual | resonant particular solution |
| `08_transform_methods.ipynb` | transform / unified transform | ✓ | ✓ | structural transform semantics | dispersion relation |
| `09_hyperbolic_systems.ipynb` | linear hyperbolic systems | ✓ | canonical-system introspection | all component PDE + IC residuals | characteristic propagation |
| `10_symmetry_reduction.ipynb` | symmetry / invariant reduction | ✓ | ✓ | original-PDE residual | invariant-coordinate solution |
| `11_kernels_green.ipynb` | kernels / Green functions | ✓ | kernel-plan introspection | smooth residual + mass; distributional caveat | heat kernel |

`00_index.ipynb` provides a compact curriculum index.

## What makes these notebooks executable documentation

Each family notebook follows the same contract:

1. state a representative PDE and derive the relevant mathematics,
2. construct the PDESolve problem representation,
3. inspect recognition and planning where the canonical coordinator applies,
4. execute the supported solver path,
5. inspect result type and method identity,
6. independently verify the returned mathematics with SymPy or family-specific identities,
7. visualize the solution or the mathematical structure when a plot is informative,
8. run at least one variation, and
9. finish with exercises that extend the same solver family without claiming unsupported generality.

The independent checks contain Python `assert` statements. A notebook therefore fails during execution if a future implementation change violates the mathematical contract assumed by the tutorial.

## Running the tutorials

Install the optional tutorial dependencies:

```bash
python -m pip install -e '.[tutorials]'
```

Start Jupyter from the repository root:

```bash
jupyter lab
```

and open `notebooks/tutorials/00_index.ipynb`.

To execute every tutorial noninteractively, use:

```bash
python tools/execute_tutorial_notebooks.py
```

By default the runner executes every notebook in lexical order and writes the executed notebooks back to disk. Use `--check` to execute without rewriting the committed files.

## Verification policy in the tutorials

The notebooks intentionally do **not** force every solver family into a smooth pointwise residual model:

- explicit classical solutions are checked by PDE and condition substitution;
- implicit first-order characteristic relations are checked against the characteristic elimination identity;
- conservation-law shocks are checked through Rankine–Hugoniot and entropy inequalities rather than pointwise differentiation at the jump;
- transform results may be verified structurally when inversion is represented by an integral;
- heat kernels are checked away from the source together with unit mass, while the source identity is understood distributionally.

This mirrors the result/verification semantics described in [Results and verification](results-verification.md).

## Notebook organization

The notebooks directly under `notebooks/` are focused examples. `notebooks/tutorials/` provides the systematic tutorial curriculum. Tutorial material should be linked to a capability-matrix row and a regression-tested solver path.
