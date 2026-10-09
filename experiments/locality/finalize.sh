#!/usr/bin/env bash
# Regenerate every table and figure in experiments/locality/results/ from the raw run records in
# /root/outputs/locality/ (written by run.py with the configs in configs/).
set -euo pipefail
cd "$(dirname "$0")/../.."
R=experiments/locality/results
O=/root/outputs/locality
py() { uv run python "$@" 2>&1 | grep -v -i "warn" || true; }

# step 1/2: two-component exponent table, exponents figure, Hessian figure (+ balanced-init twins)
py experiments/locality/make_figures.py $O/twocomp --out $R/twocomp > /dev/null
[ -n "$(ls $O/twocomp_balanced.part* 2>/dev/null)" ] && \
  py experiments/locality/make_figures.py $O/twocomp_balanced --out $R/twocomp_balanced > /dev/null
# step 2: single-run escape degree from the growth exponent (theory_saddles.md P4)
py experiments/locality/growth.py $O/twocomp --out $R/growth --workers 8 > /dev/null
# step 3: predicted sequences, level occupation times, side-by-side trees figures
[ -f $R/route_sequences.json ] || py experiments/locality/predict_sequences.py
py experiments/locality/levels.py $O/multicomp $O/multicomp_wide --out $R/levels --workers 8 > /dev/null
py experiments/locality/plot_trees.py $O/multicomp --teacher B4 --n 4 --sigma 0.07 --seed 1 \
  --start small_balanced --levels $R/levels_runs.csv --trees ht --out $R/trees_B4_ht_balanced.png
py experiments/locality/plot_trees.py $O/multicomp --teacher B4 --n 4 --sigma 0.03 --seed 1 \
  --levels $R/levels_runs.csv --trees mps binary --out $R/trees_B4.png
py experiments/locality/plot_trees.py $O/multicomp --teacher E8 --n 8 --sigma 0.1 --seed 1 \
  --levels $R/levels_runs.csv --trees mps binary --out $R/trees_E8.png
py experiments/locality/plot_trees.py $O/multicomp --teacher TA --n 4 --sigma 0.03 --seed 1 \
  --levels $R/levels_runs.csv --out $R/trees_TA.png
# step 4: bond spectra at events
py experiments/locality/bond_events.py $O/twocomp --out $R/bond_events.csv --workers 8 > $R/bond_events_summary.txt
py experiments/locality/plot_bonds.py $O/twocomp --cases mps8:25 binary8:34 ht8:12s --sigma 0.05 \
  --out $R/bond_spectra.png
# step 5: robustness (chi, d, weight ratio)
py experiments/locality/make_figures.py $O/robust --out $R/robust > /dev/null
echo done
