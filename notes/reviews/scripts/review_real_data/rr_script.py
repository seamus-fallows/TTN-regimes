"""Run an experiments/real_data analysis script's main() with the data dir redirected.
Usage: uv run python -I rr_script.py <module> args..."""
import os
import sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import importlib
from pathlib import Path

REPO = Path("c:/Users/seamu/Iliad-ML/TTN_regimes")
DATA = REPO / "data/review_real_data/data"
sys.path.insert(0, str(REPO / "experiments/real_data"))
import data as _data  # noqa: E402

_data.DATA_DIR = DATA
_data.load_task.__defaults__ = (DATA,)

if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1])
    mod.main(*sys.argv[2:])
