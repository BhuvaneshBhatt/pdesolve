from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "test_reports" / "pytest-shards.json",
    )
    args = parser.parse_args()
    files = sorted((ROOT / "tests").glob("test_*.py"))
    records = []
    start_all = time.perf_counter()

    for index, path in enumerate(files, 1):
        rel = path.relative_to(ROOT)
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(rel)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        elapsed = time.perf_counter() - start
        records.append(
            {
                "file": str(rel),
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
        status = "PASS" if proc.returncode == 0 else "FAIL"
        print(
            f"[{index:02d}/{len(files):02d}] {status} {rel} ({elapsed:.1f}s)",
            flush=True,
        )

    report = {
        "test_module_count": len(files),
        "passed_modules": sum(row["returncode"] == 0 for row in records),
        "failed_modules": sum(row["returncode"] != 0 for row in records),
        "elapsed_seconds": round(time.perf_counter() - start_all, 3),
        "modules": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}")
    return 1 if report["failed_modules"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
