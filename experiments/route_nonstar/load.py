"""Read the result CSVs with every pattern-string column as text.

Pattern strings such as 0110 look like numbers: pandas.read_csv would turn them into 110. Use
    from load import read; tr = read("transitions")     # (with experiments/route_nonstar on sys.path)
or pass dtype=str for the columns in PATTERN_COLUMNS yourself.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
PATTERN_COLUMNS = [
    "pattern", "missing", "order_learned", "forgotten", "route_string", "route_nodes",
    "pred_paper_next", "pred_route_next", "pred_pair_next", "pred_weight_next", "next_learned",
    "next_lost", "next_top_change", "next_eff_depth_amp", "off_strings", "off_recombinants",
    "paper_depths", "J", "J_near", "J_before", "key", "ident", "level_tag",
]


def read(name: str, results: Path = RESULTS) -> pd.DataFrame:
    path = results / (f"{name}.csv.gz" if name == "transitions" else f"{name}.csv")
    head = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, dtype={c: str for c in PATTERN_COLUMNS if c in head})
    for c in PATTERN_COLUMNS:
        if c in df:
            df[c] = df[c].fillna("")
    return df
