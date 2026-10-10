# Report: final-edit specification (lead, 2026-10-10)

Applies to every editor of `report/`. Sources of the items: `notes/reviews/final_review_readability.md`
(outside reader), `notes/reviews/final_review_claims_A.md` (Sec. 1-4), `notes/reviews/final_review_claims_B.md`
(Sec. 5-7, App. A), `notes/report_todo.md`. Claims fixes come first; readability second; never lose a result.

## Structure (the user's outline is kept)

1. Introduction (`01`): the five-minute version.
   - Teaser figure (three panels): the B4 routes on MPS4 and binary4 with the Steiner edges highlighted; a loss
     curve with partial-fit levels; measured exponent vs k - 2.
   - Six plain bullets without inline tags, then Table 1 in words (it carries the tags; define every symbol in
     its caption or replace it by words).
   - Main findings: at most five lines each (claim, scale of evidence, tag); numbers live in the sections.
   - 1.3 "What is known and what is new" is the ONLY place where novelty is discussed in prose (tags elsewhere).
2. Setup (`02`): model, data, teachers, training, diagnostics, plus the definitions the later sections need
   early: route, route end, Steiner subtree, escape degree k and k*, saddle order (informal), cut dimension,
   stall taxonomy (below), event timing (one paragraph; per-experiment rules in App. B), and a one-page notation
   table (symbol | meaning | where defined) replacing the pointer to macros.tex.
3. Map of regimes (`03`): the overview: phase diagrams, boundaries, what selects each regime, with forward
   references. No duplication with Sec. 5: each statement lives in ONE place (the overview states the result;
   Sec. 5 gives the detail). Delete the step-size subsection here (Sec. 5 has it).
4. Saddle-to-saddle in detail (`04`):
   - opens with a half-page box "How to read every plateau law" (scalar toy model; why routes; example teacher B4
     on MPS4 vs binary4 -- check in the data which component binary4 learns first before using it);
   - phenomenology, aligned init, escape from the origin, locality experiments;
   - NEW subsection "When is k* the saddle order?": the theorem informally in about four lines, the piggyback
     idea in words, the MPS6 counterexample in one sentence, formal statement in App. A.4;
   - order of learning; failure modes (absorbs old Sec. 5.3 "Stalled runs");
   - the parameter-dependence table (old 4.7) MOVES to Sec. 7.
5. Other regimes (`05`): lazy, rich, large chi, then steps, noise and optimizers (condensed; details to App. D).
6. Covariance and teacher structure, incl. real data (`06`).
7. Summary by parameter and open questions (`07`): 7.1 the parameter table (moved from 4.7) with the
   "landscape vs clock" paragraph; 7.2 open questions.
Appendices: A derivations; B experimental details (per-experiment event rules live here); C reproducibility
(incl. provenance); D supplementary results (material moved out of the main text; one file per main section:
`D_s2s.tex`, `D_regimes.tex`, `D_optim.tex`, `D_covariance.tex`, each starting with \subsection; each may end
with a short "Provenance and corrections after review" paragraph that collects the review history removed from
the main text).

Target: main text (Sec. 1-7) about 55-60 pages, down from about 80. Moves keep every label; use \cref, never a
hard-coded section number (the numbering changes).

## Notation: one symbol, one meaning

- Sites are numbered from 0 everywhere (fix the Sec. 2 example, the Fig. 1 TikZ labels, panel titles of the
  figures that use 1-based sites).
- Patterns: digit strings over {0, ..., d-1}, site 0 first (q = 0000, r = 1010); powers mark repeats
  (0^3 1^2 0^3). Bipartitions: 01|23 (digits are sites); ranges 0-3|4-7.
- rho_i: separation ratio only. Weight ratio: w (= s_2/s_1). Input correlation: c (as in App. A.6). Spearman:
  write "Spearman correlation r_S". Spectral radius: r(.).
- Preconditioned flow: theta' = -eta (.) grad L (diagonal eta); no D for the preconditioner.
- D: the set of sites where a component differs from the learned one. Total dimension: prod_i d_i (or d^N), never D.
- V_D: the cores that carry sites of D (was N(D)). m = |V_D|.
- k: route size (cores of the Steiner subtree) only; k* its minimum. Singular values on edge e: s_j^e. Drop the
  L_k notation (eq. 10); use L_S.
- alpha: the function-space scale prod_v sigma_v only. gamma: the order exponent P(smaller first) ~ sigma^gamma
  only. delta_Theta: the relative kernel change (delta_Theta^probe for the probe kernel). p(v): parent only
  ("p-value" in words; fitted exponents in words or with another letter). sigma: init scale only.
- Other clashes (amplitudes, growth exponents, c_gamma, gamma_e, kappa_*, m_v, m_l, the Gram scale p): rename
  within your section and record every symbol you use in the notation table (Sec. 2 editor collects them).
- Translate symbols of cited papers into ours (e.g. Furman's N, Kumar-Haupt's L).

## Stalls and events (definitions in Sec. 2, used everywhere)

- "Stalled" (umbrella, Table 4: loss above 1e-3 L(0) when the budget ends) with three subclasses: dead route
  (a route end at roundoff; cannot escape; the conservation-law trap), slow (a live route whose budget ran out),
  floor (unrealizable centred target). Drop "trapped" as a separate label; say "dead route (trap)".
- Event times: say once that experiments use different rules that agree unless a component dips or a transient
  crosses 1/2; App. B's table gives the rule per experiment.

## Style

- No internal paths, code names, CSV names or review history in the main text. Captions end "(source: App. B,
  Table N)". Provenance and correction history go to App. C / D.
- Status tags (\proven, \derived, \empirical, \conjecture) stay on every claim; \newresult / \knownresult per
  notes/reviews/citation_check.md part b. Heuristic derivations: \conjecture, or \empirical with "heuristic
  mechanism" when the data support them.
- Withdrawn claims are not restated anywhere.
- Pending inputs keep one short, clearly marked placeholder each: saddle-dyn (theorem dynamics: Sec. 4 new
  subsection, Sec. 1, App. A.4), regime-map round 2 (Sec. 3 large K), trap theory (Sec. 4 failure modes).

## Additions (lead, after the Sec. 2 edit)

- Event width: omega_r (w is the weight ratio). Correlation-flow blow-up time: \mathcal{T}(Z) (T is the run's end time).
- Macros: \Pre = diag(eta) (no D); \mask = fraktur m (m = |V_D|).
- Internal labels H1, H1b, H2, B1, B5 are replaced by words everywhere in the report.
- The notation table is tab:notation in Sec. 2; the Conventions paragraph points to it.
