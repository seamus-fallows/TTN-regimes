#!/usr/bin/env bash
# Regenerate every table and figure from the raw run records (written by run.py with the configs in
# configs/) into a given directory, without touching the committed results unless that directory is given.
# Order: (1) run.py for each config (README), (2) predict_sequences.py, (3) this script.
# usage: finalize.sh <output_dir> [workers]       raw records: $LOCALITY_OUT (default /root/outputs/locality)
# Trajectories: the path stored in each record, else $LOCALITY_OUT/traj/<file> (analysis.load_records), so a
# local copy needs only LOCALITY_OUT. The script stops at the first failing step.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <output_dir> [workers]"; exit 1; }
R=$(realpath -m "$1")
W=${2:-8}
mkdir -p "$R"
cd "$(dirname "$0")/../.."
O=${LOCALITY_OUT:-/root/outputs/locality}
C=experiments/locality/configs
X=experiments/locality
# Python warnings are silenced (also in worker processes); errors stay on the terminal and stop the script
py() {
  PYTHONWARNINGS=ignore uv run python "$@" || { echo "finalize.sh: step failed: python $*" >&2; exit 1; }
}

# Hessians at every plateau midpoint: raw and fixed gauge projection (needed by the tables)
py $X/hess_recompute.py $O/twocomp $O/twocomp_balanced $O/twocomp_chi4 $O/robust $O/k5ext $O/khigh $O/lambda2K \
   $O/multicomp $O/multicomp_wide --out $R/hessians.csv --workers $W > /dev/null
# step 1/2/5: exponent tables (restricted to the committed configs), exponents and Hessian figures
for x in twocomp:twocomp twocomp_balanced:twocomp_balanced twocomp_chi4:twocomp_chi4 robust:robust; do
  IFS=: read pref cfg <<< "$x"
  py $X/make_figures.py $O/$pref --config $C/$cfg.yaml --hessians $R/hessians.csv --out $R/$pref > /dev/null
done
py $X/make_figures.py $O/khigh $O/k5ext $O/lambda2K --config $C/khigh_extension.yaml $C/k5_extension.yaml \
   $C/lambda2_K.yaml --hessians $R/hessians.csv --out $R/extensions > /dev/null
# single-run escape degree (growth exponent), route-end amplitudes, log-correction test
py $X/growth.py $O/twocomp --out $R/growth --workers $W > /dev/null
py $X/route_amps.py $O/twocomp $O/twocomp_balanced $O/k5ext --cases mps6:135 mps8:15 mps6:05 mps6:012345 mps4:03 \
   mps8:25 mps6:14 binary8:34 binary8:07 ht4:12 ht4:03 mps4:012 binary4:12 --out $R/route_amps.csv --workers $W > /dev/null
py $X/logcorr.py $O/twocomp $O/twocomp_balanced $O/k5ext --amps $R/route_amps_summary.csv --out $R/logcorr.csv > /dev/null
# step 3: predicted sequences (committed in results/), detected plateaus and occupation times, trees figures
[ -f $X/results/route_sequences.json ] || py $X/predict_sequences.py
py $X/levels.py $O/multicomp $O/multicomp_wide --out $R/levels --workers $W > /dev/null
py $X/plot_trees.py $O/multicomp --teacher TA --n 4 --sigma 0.03 --seed 0 --levels $R/levels_runs.csv --out $R/trees_TA.png
py $X/plot_trees.py $O/multicomp --teacher B4 --n 4 --sigma 0.03 --seed 0 --levels $R/levels_runs.csv --trees mps binary \
   --out $R/trees_B4.png
py $X/plot_trees.py $O/multicomp --teacher B4 --n 4 --sigma 0.1 --seed 0 --start small_balanced --levels $R/levels_runs.csv \
   --trees ht --out $R/trees_B4_ht_balanced.png
py $X/plot_trees.py $O/multicomp --teacher E8 --n 8 --sigma 0.1 --seed 0 --levels $R/levels_runs.csv --trees mps binary \
   --out $R/trees_E8.png
# step 4: bond spectra at events (two components), rank profiles at predicted saddles (several components)
py $X/bond_events2.py $O/twocomp --out $R/bond_events2_twocomp.csv --workers $W > $R/bond_events2_twocomp_summary.txt
py $X/bond_events2.py $O/multicomp $O/multicomp_wide --profiles --out $R/bond_profiles_multicomp.csv --workers $W \
   > $R/bond_profiles_multicomp_summary.txt
py $X/bond_events2.py $O/multicomp $O/multicomp_wide --out $R/bond_events2_multicomp_events.csv --workers $W \
   > $R/bond_events2_multicomp_events_summary.txt
py $X/plot_bonds.py $O/twocomp --cases mps8:25 binary8:34 ht8:12s --sigma 0.05 --out $R/bond_spectra.png
# dead routes: run classes, sigma-dependent dropout, capped route ends; stall-diagnosis reruns and R12's criterion
py $X/stall_tables.py --out $R
py $X/r12.py $O/stalldiag --out $R/r12.csv --workers $W > $R/r12_summary.txt
echo done
