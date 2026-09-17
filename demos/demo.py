"""Run the executable correctness-and-performance receipt."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks/run.py"), run_name="__main__")
