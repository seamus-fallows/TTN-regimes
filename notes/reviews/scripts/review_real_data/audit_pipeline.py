"""Single-process audit pipeline: summarize audit runs, route amplitudes for binary runs, write JSON.

Usage: uv run python audit_pipeline.py <out_dir> <runs_dir>...   (one process; no Pool)
Writes <out_dir>/audit_summary.json and <out_dir>/audit_routes.json.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "code"))
torch.set_num_threads(1)
from analysis import load  # noqa: E402
from route_amplitudes import amplitudes, plateau_state  # noqa: E402
from summarize import summarize  # noqa: E402


def main(out_dir, *runs_dirs):
    out = Path(out_dir)
    rows, routes = [], []
    for d in runs_dirs:
        for p in sorted(Path(d).glob("*.npz")):
            row = summarize(p)
            row["runs_dir"] = str(d)
            rows.append(row)
            if row.get("tree") == "binary" and len(row.get("drops", [])) >= 2:
                run = load(p)
                rp, theta, tau_mid = plateau_state(run, row)
                if rp is not None:
                    res = amplitudes(rp, theta)
                    res.update(
                        name=row["name"],
                        tau_mid=tau_mid,
                        ordering=row["ordering"],
                        sigma=row["sigma"],
                        seed=row["seed"],
                        first_rank2_edges=row.get("first_rank2_edges"),
                        edges=run["meta"]["edges"],
                    )
                    routes.append(res)
            print("done", p.name, flush=True)
    dump = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)  # noqa: E731
    json.dump(rows, open(out / "audit_summary.json", "w"), default=dump)
    json.dump(routes, open(out / "audit_routes.json", "w"), default=dump)


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
