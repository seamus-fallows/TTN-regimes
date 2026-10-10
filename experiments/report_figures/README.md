# report_figures: figures drawn only for the report

No new runs. Scripts here redraw committed results of other experiments for the report.

## Teaser figure (Fig. 1, Sec. 1)

- Panel (a), the routes of teacher B4 on MPS4 and binary4, is TikZ: `report/tikz/teaser.tex`. The learning
  orders it shows (MPS4: 0000, 0110, 1001 in all 60 runs; binary4: 0000, 1001, 0110 in all 48) are those of
  `experiments/locality/results/levels_summary.csv`.
- Panels (b, c): `make_teaser.py`.
  - (b) loss of the MPS4 run of the Sec. 4 saddle-to-saddle figure (s2s_core, sweep `main`,
    `mps4_chi3_w3-2-1_fanin_s0.01_seed0`), read from `results/teaser_run.csv` (602 rows: log grid plus the
    refined event windows).
  - (c) measured exponent against k - 2 for the rows with k = 3-5 of
    `experiments/locality/results/twocomp_table.csv` (d = chi = 2, weights (1, 0.5)).

```
uv run python experiments/report_figures/make_teaser.py                        # from committed data
uv run python experiments/report_figures/make_teaser.py --raw data/s2s_core/main   # also rewrite teaser_run.csv
bash report/collect_figures.sh                                                   # copy into report/figures/intro/
```
