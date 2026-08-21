from pathlib import Path


def test_examples_and_notebooks_exist():
    root = Path(__file__).resolve().parents[1]
    expected_examples = {
        "complete_integral_methods.py",
        "first_order_nonlinear_demo.py",
        "conservation_law_demo.py",
        "invariant_reduction_demo.py",
    }
    expected_notebooks = {
        "complete_integral_methods.ipynb",
        "first_order_nonlinear_demo.ipynb",
        "conservation_law_demo.ipynb",
        "invariant_reduction_demo.ipynb",
    }
    assert expected_examples.issubset({p.name for p in (root / "examples").iterdir()})
    assert expected_notebooks.issubset({p.name for p in (root / "notebooks").iterdir()})


def test_readme_mentions_example_materials():
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    assert "Example materials" in text
    assert "first_order_nonlinear_demo.py" in text
    assert "invariant_reduction_demo.ipynb" in text
