"""The hopper on the worked examples of the theory notes: does it reproduce their hand predictions?

Each row: example, quantity, value stated in the note, hopper value. This checks the implementation
against the theory (no experimental data). Writes results/theory_examples.csv and
results/theory_sequences.txt (one line per predicted sequence).
usage: uv run python experiments/saddle_hopper/examples.py [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import predict  # noqa: E402
from hopper import summarize  # noqa: E402

RES = HERE / "results"


def pat(name, patterns, weights, tree, n, chi, d=2, init="fanin", **kw):
    return name, dict(
        teacher=dict(kind="pattern", patterns=patterns, weights=weights),
        tree=tree,
        n=n,
        d=d,
        chi=chi,
        init=init,
        **kw,
    )


B4 = [[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]], [1.0, 0.3, 0.6]
E8 = [[0] * 8, [0, 0, 0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0]], [1.0, 0.4, 0.7]
N4 = [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], [1.0, 0.6, 0.4]
STAR = [[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]
CASES = dict(
    [
        pat("B4 mps4", *B4, "mps", 4, 3),
        pat("B4 binary4", *B4, "binary", 4, 3),
        pat("B4 ht4", *B4, "ht", 4, 3),
        pat("E8 mps8", *E8, "mps", 8, 3),
        pat("E8 binary8", *E8, "binary", 8, 3),
        pat("N4 mps4", *N4, "mps", 4, 3),
        pat("N4 binary4", *N4, "binary", 4, 3),
        pat("star mps4", STAR, [2.0, 1.6, 1.6], "mps", 4, 3),
        pat("star binary4", STAR, [2.0, 1.6, 1.6], "binary", 4, 3),
        pat("star control mps4", STAR, [2.0, 1.2, 1.2], "mps", 4, 3),
        pat("star mps4 at q", STAR, [2.0, 1.6, 1.6], "mps", 4, 3, init="saddle", start_component=0),
        pat("star ht4 at q", STAR, [2.0, 1.6, 1.6], "ht", 4, 3, init="saddle", start_component=0),
        pat(
            "odeco wrong order binary4",
            [[0] * 4, [1] * 4],
            [1.0, 0.5],
            "binary",
            4,
            2,
            init="saddle",
            start_component=1,
        ),
        pat(
            "odeco wrong order mps4",
            [[0] * 4, [1] * 4],
            [1.0, 0.5],
            "mps",
            4,
            2,
            init="saddle",
            start_component=1,
        ),
        (
            "mixed mps4",
            dict(
                teacher=dict(kind="cp_sites", weights=[1.0, 0.6], overlaps=[0.2, 0.5, 0.7, 0.3]),
                tree="mps",
                n=4,
                d=2,
                chi=2,
                init="fanin",
            ),
        ),
        (
            "sym0.5 mps4",
            dict(
                teacher=dict(kind="nonorth", c=0.5, weights=[1.0, 0.6]),
                tree="mps",
                n=4,
                d=2,
                chi=2,
                init="fanin",
            ),
        ),
    ]
)
# (example, quantity, stated value, path into the predicted sequence: (sequence selector, event, key))
CHECKS = [
    ("B4 mps4", "q_A log slope (theory_saddles P10)", 2.333, ("main0", 1, "slope_small")),
    ("B4 mps4", "q_B degree", 4, ("main0", 2, "k")),
    ("B4 binary4", "degrees", "3, 3", None),
    ("B4 ht4", "tie q_A / q_B (k = 5)", "random", None),
    ("E8 mps8", "q_A log slope (1 - 0.4)/0.4", 1.5, ("main0", 1, "slope_small")),
    ("E8 mps8", "q_B coupling 0.7 x 1.088", 0.7616, ("main0", 2, "rate")),
    ("E8 binary8", "q_A after q_B coupling 0.4 x 0.837", 0.3348, None),
    ("N4 mps4", "levels 0.26, 0.08, 0.0569", "0.26, 0.08, 0.0569", None),
    ("N4 mps4", "partial-step rate 0.4 x 0.474", 0.1897, ("main0", 2, "rate")),
    ("star mps4 at q", "lambda_min = -a^(K-2)(rho - s_q)", -0.371573, ("main0", 1, "lam_old")),
    ("star ht4 at q", "lambda_min", -0.431073, ("main0", 1, "lam_old")),
    (
        "odeco wrong order binary4",
        "piggyback lambda -(1-0.5) 0.5^(1/3)",
        -0.39685,
        ("main0", 1, "lam_old"),
    ),
    (
        "mixed mps4",
        "edge order 0 -> 2 -> 1, slopes 6.823 / 12.178 / 14.467 (cumulative)",
        "6.823, 12.178, 14.467",
        None,
    ),
    ("sym0.5 mps4", "rank-one plateau 0.16631, A = 5.853", 5.853, ("main0", 1, "slope_small")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=predict.workers_default())
    args = ap.parse_args()
    preds = predict.run_many(list(CASES.values()), RES / "predictions_examples.json", args.workers)
    lines = []
    for name, case in CASES.items():
        p = preds[predict.key_of(case)]
        lines.append(f"== {name}  (wall {p['wall']:.1f} s)")
        for c in p["candidates"]:
            lines.append(
                f"   candidate g={c['value']:.4f} n_unstable={c['n_unstable']} gamma={c['gamma']:.3f} "
                f"overlaps={c['overlaps']}"
            )
        for s in p["sequences"]:
            lines.append(f"   {s['pclass']:4s} {summarize(s)}")
    (RES / "theory_sequences.txt").write_text("\n".join(lines) + "\n")
    rows = []
    for ex, quantity, stated, path in CHECKS:
        p = preds[predict.key_of(CASES[ex])]
        main = [s for s in p["sequences"] if s["pclass"] == "main"]
        value = None
        if path is not None and main:
            _, j, key = path
            evs = main[0]["events"]
            value = evs[j].get(key) if j < len(evs) else None
        elif ex == "B4 binary4":
            value = ", ".join(str(e.get("k")) for e in main[0]["events"][1:])
        elif ex == "B4 ht4":
            value = main[0]["branches"][1]
        elif ex == "E8 binary8":
            value = [
                s["events"][2]["rate"] for s in main if s["events"][1]["newly_learned"] == [2]
            ][0]
        elif ex == "N4 mps4":
            value = ", ".join(f"{e['to_level']:.4g}" for e in main[0]["events"][:-1])
        elif ex == "mixed mps4":
            cum, out = 0.0, []
            for e in main[0]["events"][1:]:
                cum += e["slope_small"]
                out.append(
                    f"edge {[i for i, (a, b) in enumerate(zip(e['from_profile'], e['to_profile'])) if b > a]}"
                    f" {cum:.3f}"
                )
            value = ", ".join(out)
        rows.append(dict(example=ex, quantity=quantity, stated=stated, hopper=value))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "theory_examples.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(df.to_string())


if __name__ == "__main__":
    main()
