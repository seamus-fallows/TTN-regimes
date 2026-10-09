#!/usr/bin/env bash
# Copy the report's figures from experiments/*/results/ into report/figures/.
#
# The list of figures is report/figures.txt (format documented at its top): one
# "<source> <destination>" pair per line, <source> relative to the repository root and inside
# experiments/<name>/results/, <destination> relative to report/figures/.
#
# Usage (Git Bash on Windows, or Linux/macOS), from any directory:
#   bash report/collect_figures.sh        copy every listed figure
#   bash report/collect_figures.sh -n     dry run: only check that every source exists
#
# Figures are not regenerated here: each experiment's README says which script produces them
# (see also Appendix C of the report). Exits non-zero if a line is malformed or a source is missing.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
list="$here/figures.txt"
dest_dir="$here/figures"

dry_run=0
if [[ "${1:-}" == -n ]]; then
  dry_run=1
elif [[ $# -gt 0 ]]; then
  echo "usage: bash collect_figures.sh [-n]" >&2
  exit 2
fi

copied=0
missing=0
bad=0
line_no=0
# Strip carriage returns so that a list edited on Windows (CRLF) still parses.
while IFS= read -r line || [[ -n "$line" ]]; do
  line_no=$((line_no + 1))
  line="${line%$'\r'}"
  line="${line%%#*}"                       # drop comments
  read -r src dst extra <<<"$line" || true
  [[ -z "${src:-}" ]] && continue
  if [[ -z "${dst:-}" || -n "${extra:-}" ]]; then
    echo "figures.txt:$line_no: expected '<source> <destination>'" >&2
    bad=$((bad + 1)); continue
  fi
  if [[ "$src" != experiments/*/results/* || "$src" == *..* ]]; then
    echo "figures.txt:$line_no: source must be inside experiments/<name>/results/: $src" >&2
    bad=$((bad + 1)); continue
  fi
  if [[ "$dst" == /* || "$dst" == *..* ]]; then
    echo "figures.txt:$line_no: destination must be a relative path inside figures/: $dst" >&2
    bad=$((bad + 1)); continue
  fi
  if [[ ! -f "$root/$src" ]]; then
    echo "missing: $src" >&2
    missing=$((missing + 1)); continue
  fi
  if [[ $dry_run -eq 0 ]]; then
    mkdir -p "$(dirname "$dest_dir/$dst")"
    cp -p "$root/$src" "$dest_dir/$dst"
  fi
  copied=$((copied + 1))
done <"$list"

verb=$([[ $dry_run -eq 1 ]] && echo "found" || echo "copied")
echo "collect_figures.sh: $verb $copied figure(s); $missing missing; $bad malformed line(s)"
[[ $missing -eq 0 && $bad -eq 0 ]]
