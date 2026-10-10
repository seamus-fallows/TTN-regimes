"""Wrapper around experiments/real_data/run.py that redirects the hardcoded data dir.
Usage: uv run python -I rr_run.py <config.yaml> <out_dir> <workers>"""
import os
import sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
from pathlib import Path

REPO = Path("c:/Users/seamu/Iliad-ML/TTN_regimes")
DATA = REPO / "data/review_real_data/data"
sys.path.insert(0, str(REPO / "experiments/real_data"))
import data as _data  # noqa: E402

_data.DATA_DIR = DATA
_data.load_task.__defaults__ = (DATA,)
import run as _run  # noqa: E402

if __name__ == "__main__":
    _run.main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
