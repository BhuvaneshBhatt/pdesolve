from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = ROOT / "notebooks" / "tutorials"
EXPECTED = {
    "00_index.ipynb",
    "01_linear_first_order.ipynb",
    "02_nonlinear_first_order.ipynb",
    "03_conservation_laws.ipynb",
    "04_heat_equations.ipynb",
    "05_wave_equations.ipynb",
    "06_elliptic_separation.ipynb",
    "07_constant_coefficient.ipynb",
    "08_transform_methods.ipynb",
    "09_hyperbolic_systems.ipynb",
    "10_symmetry_reduction.ipynb",
    "11_kernels_green.ipynb",
}


def test_tutorial_inventory_is_complete():
    assert {p.name for p in TUTORIAL_DIR.glob("*.ipynb")} == EXPECTED


def test_family_tutorials_have_required_pedagogical_and_executable_sections():
    for path in sorted(TUTORIAL_DIR.glob("[0-9][1-9]_*.ipynb")) + sorted(
        TUTORIAL_DIR.glob("1[01]_*.ipynb")
    ):
        nb = nbformat.read(path, as_version=4)
        markdown = "\n".join(
            cell.source for cell in nb.cells if cell.cell_type == "markdown"
        )
        code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
        assert "Derivation" in markdown, path.name
        assert "Variation and exercises" in markdown, path.name
        assert "show_plan" in code or "extract_canonical_linear_system_form" in code, (
            path.name
        )
        assert "assert " in code, path.name
        assert "plt." in code, path.name


def test_committed_tutorial_outputs_contain_no_execution_errors():
    for path in sorted(TUTORIAL_DIR.glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        errors = [
            output
            for cell in nb.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert errors == [], path.name


def test_docs_and_tutorials_use_canonical_result_metadata_api():
    """Keep maintained examples on the single documented result metadata API."""
    doc_paths = sorted((ROOT / "docs").rglob("*.md"))
    notebook_paths = sorted((ROOT / "notebooks").rglob("*.ipynb"))

    offenders = []
    for path in doc_paths + notebook_paths:
        text = path.read_text(encoding="utf-8")
        if "result.details" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == [], (
        "Use result.metadata in maintained documentation/tutorials; "
        f"result.details found in: {offenders}"
    )
