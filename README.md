# pdesolve

`pdesolve` is a symbolic PDE toolbox built on SymPy. `pdesolve(...)` is the main solving entry point; the package also exposes focused APIs for first-order PDEs, complete integrals, conservation laws, constant-coefficient operators, transforms/separation, hyperbolic systems, and Green/fundamental solutions.

## Quick start

```python
import sympy as sp
import pdesolve as pds

x, t = sp.symbols("x t", real=True)
u = sp.Function("u")
eq = sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0)

res = pds.pdesolve(eq, u(x, t), (x, t), method="auto")
print(res.method)
print(res.solution)
```

## PDESolve documentation

The documentation is built from the package API, canonical execution registry, recognizers, planners, regression tests, worked examples, and executable tutorials. Start at [`docs/index.md`](docs/index.md). In particular:

- [`docs/capability-matrix.md`](docs/capability-matrix.md) — supported PDE families and support level
- [`docs/method-inventory.md`](docs/method-inventory.md) — exact canonical method keys, recognizers, direct APIs, and formal methods
- [`docs/public-api-inventory.md`](docs/public-api-inventory.md) — all package-level exports and signatures
- [`docs/inputs-conditions-domains.md`](docs/inputs-conditions-domains.md) — structured inputs and geometry
- [`docs/results-verification.md`](docs/results-verification.md) — result classes and verification semantics
- [`docs/planning-diagnostics.md`](docs/planning-diagnostics.md) — recognition/ranking/trace inspection
- [`docs/limitations.md`](docs/limitations.md) — support boundaries and partial/formal capabilities
- [`docs/developer-guide.md`](docs/developer-guide.md) — extending the package safely
- [`docs/tutorial-notebooks.md`](docs/tutorial-notebooks.md) — PDESolve documentation executable tutorial curriculum

A `mkdocs.yml` navigation file is included so the Markdown tree can be rendered with MkDocs if desired.

## Inspecting automatic planning

```python
problem = pds.build_pde_problem(eq, u(x, t), (x, t))
plan = pds.plan_canonical_problem(problem)
for step in plan.steps:
    print(step.method, step.score, step.reasons)
```


## Kernel and Green-function APIs

Use `solve_fundamental_solution(...)` and `solve_green_function(...)` for explicit kernel work. Automatic kernel routing is conservative and is primarily source-driven, so ordinary homogeneous heat/wave IVPs continue to use condition-aware IVP, transform, or series paths.

## Installation

```bash
pip install -e .
```

Requires Python 3.11+ and SymPy 1.12+.

## Tests

```bash
python -m pytest -q
```

The suite covers solver families, coordinator layers, structured conditions/domains, verification/trace behavior, benchmarks, Green functions, and a 52-example heterogeneous PDE reference matrix.


## License and source

PDESolve is licensed under the GNU General Public License v3.0 only (GPL-3.0-only). The canonical source repository is https://github.com/BhuvaneshBhatt/pdesolve.

## Example materials

Runnable examples include `examples/first_order_nonlinear_demo.py`, `examples/conservation_law_demo.py`, `examples/invariant_reduction_demo.py`, `examples/complete_integral_methods.py`, and `examples/pdesolve_demo.py`. Focused notebooks include `notebooks/invariant_reduction_demo.ipynb`; the systematic tutorial curriculum is under `notebooks/tutorials/`; the capability matrix and method inventory provide the authoritative overview of supported methods.

