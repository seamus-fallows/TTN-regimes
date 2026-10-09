#!/usr/bin/env bash
# Build the report: pdflatex -> bibtex -> pdflatex -> pdflatex (plus up to two more pdflatex
# passes while LaTeX still reports changed labels), then check the final log.
#
# Usage (Git Bash on Windows, or Linux/macOS), from any directory:
#   bash report/build.sh           build report/main.pdf
#   bash report/build.sh clean     remove LaTeX build files (keeps main.pdf)
#
# Needs pdflatex and bibtex in PATH (MiKTeX or TeX Live). Exits non-zero if pdflatex or bibtex
# fails, or if the final log reports undefined references or citations, multiply defined labels,
# or labels that are still changing. Overfull boxes and BibTeX warnings are listed but do not fail
# the build.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
main=main
build_files=(aux log out toc bbl blg fls fdb_latexmk synctex.gz lof lot)
max_latex_runs=5

if [[ "${1:-}" == clean ]]; then
  for ext in "${build_files[@]}"; do rm -f "$main.$ext"; done
  echo "build.sh: removed build files"
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "usage: bash build.sh [clean]" >&2
  exit 2
fi

for tool in pdflatex bibtex; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "build.sh: $tool not found in PATH" >&2
    exit 1
  fi
done

stderr_file=$(mktemp)
trap 'rm -f "$stderr_file"' EXIT

# Run a command with stdout discarded; pass its stderr on, minus MiKTeX's update reminder.
quiet() {
  local status=0
  "$@" >/dev/null 2>"$stderr_file" || status=$?
  grep -v "you have not checked for MiKTeX updates" "$stderr_file" >&2 || true
  return $status
}

latex_runs=0
run_latex() {
  latex_runs=$((latex_runs + 1))
  echo "build.sh: pdflatex (pass $latex_runs)"
  if ! quiet pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "$main.tex"; then
    echo "build.sh: pdflatex failed; end of $main.log:" >&2
    tail -n 40 "$main.log" >&2
    exit 1
  fi
}

run_bibtex() {
  echo "build.sh: bibtex"
  # Some bibtex versions exit with status 1 on warnings only; fail on errors.
  local status=0
  quiet bibtex "$main" || status=$?
  if [[ $status -gt 1 ]] || grep -Eq "error message|I couldn't open|I found no" "$main.blg"; then
    echo "build.sh: bibtex failed (status $status); $main.blg:" >&2
    cat "$main.blg" >&2
    exit 1
  fi
}

labels_changed() {
  grep -Eq "Rerun to get|Label\(s\) may have changed" "$main.log"
}

run_latex
run_bibtex
run_latex
run_latex
while labels_changed && [[ $latex_runs -lt $max_latex_runs ]]; do
  run_latex
done

# ---- checks on the final log ----------------------------------------------------------------
log="$main.log"
undefined=$(grep -E "(Reference|Citation) .* undefined|There were undefined (references|citations)" "$log" || true)
multiple=$(grep -E "multiply[- ]defined" "$log" || true)
rerun=$(grep -E "Rerun to get|Label\(s\) may have changed" "$log" || true)
overfull=$(grep -E "^Overfull \\\\[hv]box" "$log" || true)
underfull=$(grep -cE "^Underfull \\\\[hv]box" "$log" || true)
bibwarn=$(grep -E "^Warning--" "$main.blg" || true)

status=0
if [[ -n "$undefined" ]]; then
  echo "build.sh: UNDEFINED references or citations:"; echo "$undefined"; status=1
fi
if [[ -n "$multiple" ]]; then
  echo "build.sh: MULTIPLY DEFINED labels:"; echo "$multiple"; status=1
fi
if [[ -n "$rerun" ]]; then
  echo "build.sh: labels still changing after $latex_runs passes:"; echo "$rerun"; status=1
fi
if [[ -n "$overfull" ]]; then
  echo "build.sh: overfull boxes:"; echo "$overfull"
fi
if [[ -n "$bibwarn" ]]; then
  echo "build.sh: BibTeX warnings:"; echo "$bibwarn"
fi
echo "build.sh: $underfull underfull box(es)"
pages=$(grep -Eo "Output written on $main.pdf \([0-9]+ pages?" "$log" | grep -Eo "[0-9]+ pages?" || true)
if [[ $status -eq 0 ]]; then
  echo "build.sh: OK, $main.pdf ($pages, $latex_runs pdflatex passes)"
else
  echo "build.sh: built $main.pdf ($pages) but the checks above failed"
fi
exit $status
