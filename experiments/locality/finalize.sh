#!/usr/bin/env bash
# Regenerate every table and figure from the raw run records in /root/outputs/locality/ (written by
# run.py with the configs in configs/). Order: (1) run.py for each config, (2) predict_sequences.py,
# (3) this script.
# usage: finalize.sh <output_dir> [workers]   e.g. finalize.sh /tmp/locality_rerun 8
# (the committed results live in experiments/locality/results/; pass that path only to refresh them)
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <output_dir> [workers]"; exit 1; }
R=$(realpath -m "$1")
W=${2:-8}
mkdir -p "$R"
cd "$(dirname "$0")/../.."
O=/root/outputs/locality
py() { uv run python "$@" 2>&1 | grep -v -i "warn" || true; }

# step 1/2: two-component exponent table, exponents figure, Hessian figure (+ balanced-init twins)
py experiments/locality/make_figures.py $O/twocomp --out $R/twocomp > /dev/null
[ -n "$(ls $O/twocomp_balanced.part* 2>/dev/null)" ] && \
  py experiments/locality/make_figures.py $O/twocomp_balanced --out $R/twocomp_balanced > /dev/null
# step 2: single-run escape degree from the growth exponent (theory_saddles.md P4)
py experiments/locality/growth.py $O/twocomp --out $R/growth --workers $W > /dev/null
# step 3: predicted sequences, level occupation times, side-by-side trees figures
[ -f experiments/locality/results/route_sequences.json ] || py experiments/locality/predict_sequences.py
py experiments/locality/levels.py $O/multicomp $O/multicomp_wide --out $R/levels --workers $W > /dev/null
py experiments/locality/plot_trees.py $O/multicomp --teacher B4 --n 4 --sigma 0.07 --seed 1 \
  --start small_balanced --levels $R/levels_runs.csv --trees ht --out $R/trees_B4_ht_balanced.png
py experiments/locality/plot_trees.py $O/multicomp --teacher B4 --n 4 --sigma 0.03 --seed 1 \
  --levels $R/levels_runs.csv --trees mps binary --out $R/trees_B4.png
py experiments/locality/plot_trees.py $O/multicomp --teacher E8 --n 8 --sigma 0.1 --seed 1 \
  --levels $R/levels_runs.csv --trees mps binary --out $R/trees_E8.png
py experiments/locality/plot_trees.py $O/multicomp --teacher TA --n 4 --sigma 0.03 --seed 1 \
  --levels $R/levels_runs.csv --out $R/trees_TA.png
# step 4: bond spectra at events
py experiments/locality/bond_events.py $O/twocomp --out $R/bond_events.csv --workers $W > $R/bond_events_summary.txt
py experiments/locality/plot_bonds.py $O/twocomp --cases mps8:25 binary8:34 ht8:12s --sigma 0.05 \
  --out $R/bond_spectra.png
# step 5: robustness (chi, d, weight ratio)
py experiments/locality/make_figures.py $O/robust --out $R/robust > /dev/null
echo done
