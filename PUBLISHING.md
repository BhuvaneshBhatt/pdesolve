# Publishing checklist

This checklist covers packaging and publication tasks that are separate from solver correctness.

## Package metadata

- [x] Centralize package metadata in `pyproject.toml`.
- [x] Add author and maintainer metadata, keywords, classifiers, source URLs, and GPL-3.0-only licensing.
- [x] Audit and document the package-level public API.
- [x] Remove temporary development scripts, caches, and bytecode artifacts from release archives.

## Source layout

- Runtime package code lives under `src/pdesolve/`.
- Install the package before running tests, for example `python -m pip install -e . --no-build-isolation` in an offline environment or `python -m pip install -e .` normally.
- Tests, documentation, notebooks, examples, and repository maintenance tools stay outside `src/` and are not runtime packages.

## Distribution validation

- [ ] Build both source and wheel distributions with `python -m build`.
- [ ] Run strict distribution metadata validation.
- [ ] Inspect source and wheel contents.
- [ ] Install the wheel in a clean environment and run public-API smoke tests.
- [ ] Test the supported Python/SymPy compatibility matrix and narrow metadata if needed.

## Publication automation

- [ ] Add CI for the supported interpreter/dependency matrix.
- [ ] Configure TestPyPI/PyPI trusted publishing.
- [ ] Publish the documentation and add its stable URL to project metadata.
- [ ] Make tagged releases build from clean source and publish immutable artifacts.
