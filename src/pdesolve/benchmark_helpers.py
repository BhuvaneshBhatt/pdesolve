from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PDEBenchmarkMeasurement:
    rows: tuple[dict, ...]

    def as_csv_rows(self) -> list[dict]:
        return list(self.rows)


def _benchmark_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "benchmarks"


def load_benchmark_measurement_csv() -> PDEBenchmarkMeasurement:
    path = _benchmark_data_dir() / "pde_benchmark_measurement.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = tuple(csv.DictReader(fh))
    return PDEBenchmarkMeasurement(rows)


def load_benchmark_measurement_json() -> PDEBenchmarkMeasurement:
    path = _benchmark_data_dir() / "pde_benchmark_measurement.json"
    with path.open("r", encoding="utf-8") as fh:
        rows = tuple(json.load(fh))
    return PDEBenchmarkMeasurement(rows)
