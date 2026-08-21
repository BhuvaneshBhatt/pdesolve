from importlib.metadata import version
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _metadata():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_project_metadata_has_static_version():
    project = _metadata()["project"]
    assert project["version"] == "0.1.0"
    assert "version" not in project.get("dynamic", [])


def test_installed_version_matches_project_metadata():
    project = _metadata()["project"]
    assert version(project["name"]) == project["version"]


def test_documented_development_extras_exist():
    extras = _metadata()["project"]["optional-dependencies"]
    assert {"test", "docs", "release", "tutorials"} <= set(extras)


def test_top_level_benchmark_exports():
    import pdesolve

    names = (
        "BenchmarkCase",
        "BenchmarkOutcome",
        "BenchmarkSuite",
        "build_benchmark_suite",
        "run_benchmark_case",
        "run_benchmark_suite",
    )
    assert all(hasattr(pdesolve, name) for name in names)
