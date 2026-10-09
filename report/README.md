# Report

LaTeX source of the project report, *Training dynamics of tree tensor networks*. The compiled
`main.pdf` is committed; LaTeX build files are gitignored.

## Build

```bash
bash report/build.sh          # pdflatex, bibtex, pdflatex, pdflatex -> report/main.pdf
bash report/build.sh clean    # remove build files (keeps main.pdf)
```

The script works in Git Bash on Windows (MiKTeX) and on Linux/macOS (TeX Live). It needs
`pdflatex` and `bibtex` in `PATH`; `make` and `latexmk` are not needed. It fails if pdflatex or
BibTeX fails, or if the final log has undefined references, undefined citations or multiply
defined labels; it lists overfull boxes and BibTeX warnings without failing. All packages used are
in a standard MiKTeX or TeX Live installation (bibliography: natbib + BibTeX with `plainnat`).

## Layout

| path | contents |
| --- | --- |
| `main.tex` | preamble (packages, theorem environments, status tags), front matter, section order |
| `macros.tex` | notation macros used in all sections |
| `sections/` | one file per section: `01_introduction` ... `07_open_questions`, appendices `A_derivations`, `B_experimental_details`, `C_reproducibility` |
| `tikz/` | TikZ figures drawn in LaTeX (`trees.tex`: the three tree topologies) |
| `figures/` | figures copied from the experiments by `collect_figures.sh` (do not edit by hand) |
| `figures.txt` | list of figures to copy (format below) |
| `references.bib` | bibliography; every entry was checked against the publisher, proceedings or arXiv page (the `% verified:` line above each entry names the source) |

## Figures

Figures are produced by the experiments, never in this folder:

1. Each experiment's plotting script writes its figures to `experiments/<name>/results/`; the
   experiment's README gives the command and which raw data it needs (raw data live in
   `data/<name>/`, outside git). Appendix C of the report lists, for every figure, the script,
   config and data that regenerate it.
2. `report/figures.txt` lists the figures used in the report, one per line:

   ```text
   <source>                                        <destination>
   experiments/s2s_core/results/escape_vs_sigma.pdf  s2s/escape_vs_sigma.pdf
   ```

   `<source>` is relative to the repository root and must lie in `experiments/<name>/results/`;
   `<destination>` is relative to `report/figures/` and is the name passed to `\includegraphics`
   (the graphics path is `figures/`). No spaces in paths; `#` starts a comment.
3. `bash report/collect_figures.sh` copies them (`-n` only checks that every source exists). Then
   rebuild the report.

When a figure is added, add its line to `figures.txt` and its row to the table in Appendix C
(`sections/C_reproducibility.tex`).

## Conventions

- **Status tags.** Every claim carries one tag, typeset as a small coloured label:
  `\proven` (proof given or cited), `\derived` (derivation under stated assumptions),
  `\empirical` (experiments only), `\conjecture` (neither). Put the tag at the start of a
  proposition's body or right after the claim in running text.
- **New versus known.** Results believed new are marked `\newresult` (optional qualifier:
  `\newresult[for TTNs]`); known results `\knownresult[<reference>]`, e.g.
  `\knownresult[\citealp{saxe2014exact}]`. Unmarked statements are definitions or standard
  background. The introduction's summary table must give every finding a status tag and a
  new/known marker.
- **Environments.** `theorem`, `proposition`, `lemma`, `corollary`, `conjecture`, `definition`,
  `remark` share one counter, numbered within sections; refer to them with `\cref`. Define new
  ones with `\newsharedtheorem{<env>}{<Name>}` in `main.tex` (it gives each environment an alias
  of the shared counter; with plain `\newtheorem{env}[theorem]{...}` the installed cleveref
  version names every such environment "Theorem"). The environment `conjecture` and the tag
  `\conjecture` coexist (the environment restores the tag inside its body).
- **Notation.** All symbols are macros in `macros.tex` (e.g. `\W`, `\Wt`, `\G{v}`, `\Kc`,
  `\bd{v}`, `\isc`, `\fsc`, `\tw{r}`, `\bs{e}{k}`, `\ov{r}`, `\TK`, `\dTK`, `\sep{i}`), following
  `notes/model.md` and `notes/diagnostics.md`. Deliberate differences from the notes: the tangent
  kernel is `\TK` ($\Theta$; the notes' $K_t$ clashes with the number of cores $K$), aligned-init
  amplitudes are $q_{v,r}$ (the notes' $a_{v,r}$ clashes with the overlaps $a_r$), and the
  gradient-flow preconditioner is $D$ (the probe-set size is $P$).
- **Drafting aids.** `\placeholder{...}` marks unwritten parts and `\TODO{...}` inline notes; both
  are shown in red and disappear with `\showtodosfalse` in `main.tex`. Planned content lives in
  `% TODO` comments in each section file.
- **Citations.** natbib author-year: `\citet{key}` (Author (year)), `\citep{key}` ((Author,
  year)). Keys are `<first author><year><first title word>`, e.g. `saxe2014exact`. Add an entry
  only after checking it against the publication, with a `% verified:` comment.
