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
O=${LOCALITY_OUT:-/root/outputs/locality}
py() { uv run python "$@" 2>&1 | grep -v -i "warn" || true; }

# step 1/2: two-component exponent table, exponents figure, Hessian figure (+ balanced-init twins)
py experiments/locality/make_figures.py $O/twocomp --out $R/twocomp > /dev/null
for x in twocomp_balanced twocomp_chi4; do
  [ -n "$(ls $O/$x.part* 2>/dev/null)" ] && py experiments/locality/make_figures.py $O/$x --out $R/$x > /dev/null
done
# lead's request (c): route-entry amplitudes at the plateau midpoint (pinned ends)
py experiments/locality/route_amps.py $O/twocomp $O/twocomp_balanced $O/k5ext --cases mps6:135 mps8:15 mps6:05 \
  mps6:012345 mps4:03 mps8:25 mps6:14 binary8:34 binary8:07 ht4:12 ht4:03 mps4:012 binary4:12 \
  --out $R/route_amps.csv --workers $W > /dev/null
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
for f in 0.03 0.05; do  # threshold sensitivity (notes/diagnostics.md: f in [0.03, 0.1])
  py experiments/locality/bond_events.py $O/twocomp --f $f --max-runs 1500 --out $R/bond_events_f$f.csv \
    --workers $W > $R/bond_events_f${f}_summary.txt
done
py experiments/locality/plot_bonds.py $O/twocomp --cases mps8:25 binary8:34 ht8:12s --sigma 0.05 \
  --out $R/bond_spectra.png
# step 5: robustness (chi, d, weight ratio)
py experiments/locality/make_figures.py $O/robust --out $R/robust > /dev/null
echo done
