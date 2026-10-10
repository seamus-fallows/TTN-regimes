# Readability review of the report (outside reader)

Reviewer stance: an ML theorist who knows linear algebra, tensors and deep linear networks (DLNs), but
not tensor networks. I read the abstract and Sections 1–4 closely (pp. 1–52) and skimmed Sections 5–7
and Appendices A–C. Page numbers refer to `report/main.pdf` built on 10 Oct 2026 (138 pp.; main text
pp. 3–82, references pp. 82–88).

## Verdict

- **The story is simple and strong, but the text hides it.** The main results fit in five lines.
  (1) The first plateau lasts ∝ σ^{-(K-2)}, with a prefactor that can be computed for each
  initialization. (2) Each later plateau is set by k*, the number of cores the next component must
  recruit (none, ln(1/σ), or σ^{-(k*-2)}). (3) As a result, locality beats weight. (4) A lazy regime
  needs every bond to be full. (5) Which invariance an optimizer has decides which plateau features
  survive. A reader has to dig these out of about 80 pages of per-seed numbers.
- **Five-minute test: not passed.** A busy reader gets "first plateau σ^{-(K-2)}; later the tree
  decides" from the abstract, but not *what* decides it or *why*. The one-paragraph answer (p. 4) and
  Table 1 (p. 5) use about 15 terms that are defined 10–30 pages later or only in Appendix A: Steiner
  subtree, route, odeco, cut dimension, centred model, piggyback graph, head start, T(Z), Λ_r, ϱ, H_⊥,
  light root, injective norm, separated teacher and route end. The ten numbered findings fill 3.5
  pages of run counts and error bars. The first figure is on p. 13 and the first plot on p. 23.
  Realistically the summary takes 15–20 minutes, with look-ups.
- **The main text reads like a results archive.** Sections 2–6 are full of per-seed numbers, run
  counts, threshold variants, integrator settings, review history ("withdrawn", "verification
  review", "round-1") and internal file paths. About 20–28 pages can move to the appendices without
  losing a result (cut list at the end).

## What works (keep)

- The "Takeaway" sentence in every caption. Figures 3, 7, 9 and 18 are excellent and need nothing
  else.
- Table 1, with one row per question. The idea of status tags. The honest scope sentences.
- Several synthesis paragraphs deserve promotion to the introduction:
  - "landscape vs clock" after Table 10 (pp. 51–52);
  - "Interpretation: which invariance removes which feature" (pp. 65–66);
  - "When does gradient flow describe practical training?" (p. 61);
  - the summary in 6.10 (p. 78).
- The best simple example is already in the report, buried on p. 39: teacher B4 on MPS4 versus
  binary4 ("locality beats weight"). Use it as the running example (HI-3).

## High priority

### HI-1. Key terms are used long before they are defined, or are defined only in the appendix

| term | first used | defined | fix |
|---|---|---|---|
| cut dimension | abstract; 1.1(iii) p. 3; item 8 p. 8; 3.1–3.2; 5.1 | **only App. A.8** (`A_derivations.tex` l. 1300). Eq. (3) gives the cap but does not name it | After eq. (3): "We call min(∏_{i∈A_v} d_i, ∏_{i∉A_v} d_i), the largest rank W^{⟨e_v⟩} can have, the *cut dimension* of e_v; a bond is *full* when χ_v reaches it (binary4 with d = 3: both bonds have cut dimension 9)." Abstract: "...to a lazy one, which exists only when every bond dimension reaches the rank of the matricization it cuts." |
| bond rank | abstract, 1.2 | implicit in Prop. 2.1; "effective bond rank" only on p. 19 | In 2.1: "the *bond rank* of e is rank W^{⟨e⟩} ≤ χ_e, the number of nonzero values in its bond spectrum." |
| Steiner subtree | 1.2 p. 4, items 3–4 p. 6 | 4.4, pp. 33–34. The concept appears **unnamed** in Prop. 2.1(b), p. 13 | Name it in Prop. 2.1(b): "...if e lies in the *Steiner subtree* of the cores that carry the sites of D (the smallest subtree containing them)". Add the same parenthesis at the first use in 1.2. |
| route | abstract (loosely); 1.2; 4.1 "Prediction" (forward reference to eq. 22) | 4.4, p. 34, "at no Gauss–Newton cost"; the mechanism only in Lemma A.16 | See HI-3. |
| saddle order, higher-order saddle | abstract; items 4–5 | 2.7, operationally (scaling of λ_min); formally only in App. A.4 | In 2.7: "The *order* of a saddle θ_S is the largest p with L(θ) − L(θ_S) ≥ −C‖θ−θ_S‖^p near θ_S. It is 2 at a strict saddle (a negative Hessian eigenvalue) and at least 3 at a higher-order saddle, where the escape direction is flat to second order." |
| piggyback operator / graph | items 4–5, pp. 6–7; 4.4 p. 34 | only Def. A.29 and Prop. A.22 | See HI-4. |
| k*, k, m | abstract and 1.2 use k*; Table 7 and Fig. 7 use k; Fig. 9 uses k*; 4.5 adds m | k: eq. 22, p. 34. k*: the end of a dense paragraph on p. 34 ("k* = k for two components"). m: 4.5, p. 42 | Define all three once, together, at the start of Section 4 (wording in HI-3). |
| odeco, pattern teacher, separated/close teacher, centred model, light root, μP-TTN / injective norm, head start, T(Z), Λ_r, ϱ, H_⊥ | Table 1 and items 1–8, pp. 4–8 | 2.3, 2.4, 2.7, 3.4, 4.1, 4.3–4.5, 5.4 (injective norm: A.2), 6.1 | Define each in a short glossary box before the findings, or use words in the summary ("orthogonal teacher", "the bound computed from the initialization"). |
| route end, route-end slice, dead route, trapped | item 7, p. 7; 4.6, pp. 46–49; Fig. 13 | never in the main text (only in Prop. A.28) | In 4.6: "the *ends* of a route are its terminal cores; an end's *slice* is the core's new column on the route's channel; a route is *dead* when an end slice has decayed to roundoff." |
| H1, H1b, H2 | 4.4, p. 39 ("literal reading of H2"); 4.5, p. 46 ("Status") | **never**: internal labels from `notes/prior_knowledge.md` | Replace each label by the statement it stands for. |
| B1, B5 | 1.1, p. 4; Table 1 | internal numbering (where are B2–B4?) | Rename to Q9 (optimizers) and Q10 (real data). |
| binary4, MPS8, HT4, binary16, star3, midroot6 | item 1, p. 4 | binary4 only in Table 1's caption and in 3.1; MPS names on p. 35. star3 and midroot6 are never drawn | In 2.2: "We write binaryN, MPSN, HTN for the tree with N sites." Add star3 and midroot6 to Figure 1. |

Smaller cases of the same problem:
- "site" (1.1(i), p. 3) is used before 2.1. Write "input position (site)" at first use.
- HT, ALS and DMRG are not expanded at first use in the introduction.
- "Gauss–Newton cost" (p. 6, p. 34) is never explained.

### HI-2. Notation changes between sections

- **Site numbering.**
  - 1-based in: Fig. 1 (the comment in `tikz/trees.tex` says the code is 0-based); 2.3 (π_1 = (1,1,1,1),
    D = {1,2}); the Fig. 3 caption ("sites numbered from 1"); the Fig. 18 caption ({1,2}|{3,4}).
  - 0-based from 4.4 on: p. 33 ("numbered from 0, as in Table 16"); Table 7; Fig. 10; the 4.5 setup;
    6.4–6.7; App. A.
  - A reader who compares Fig. 10 with Fig. 1 gets the wrong edges.
  - Fix: use one convention. 0-based needs the fewest edits: the 2.3 example, the labels of Fig. 1, and
    the panel titles of Figs. 3 and 18.
- **Patterns.**
  - 2.3 writes tuples over {1..d}.
  - Later sections write strings: 000000 / 110001 (p. 34), 0000 / 0110 / 1001 (p. 39), powers
    0^3 1^2 0^3 (p. 39), q = 0000, r = 1010 (6.8).
  - Fix: define the string form in 2.3, e.g. "π_2 = 1100: basis vector 1 on sites 0, 1 and basis
    vector 0 elsewhere; powers mark repeats".
- **Bipartitions.**
  - Written {1,2}|{3,4} (Figs. 3, 18), {1–4}|{5–8} (Fig. 3), (01|23) (6.5), 01|23 (p. 70) and
    {0123}|{4567} (6.7).
  - Fix: pick one form.
- **Symbol clashes, worst first.**
  - ρ, five meanings:
    - ρ_i, the separation ratio (eq. 19);
    - ϱ, the weight ratio (4.5, Table 8, Fig. 11). It prints almost the same as ρ;
    - ρ, the correlation in 6.4 ("ρ is the c of Section A.6");
    - Spearman ρ (6.9);
    - ρ_X, a spectral radius (Thm. A.31).
    - Fix: keep ρ_i for the separation ratio. Write the weight ratio as w (6.8 already uses w for
      weights). Write the correlation as c, as App. A.6 does. Write "Spearman correlation" in words.
  - D means the preconditioner in θ̇ = −D∇L (eq. 15) and also the site set D (2.3, 4.4–4.6).
    - Fix: write θ̇ = −η ⊙ ∇L; the preconditioner is diagonal.
  - k is the route size, the index of s_k^e, and the number of learned components in L_k (eq. 10).
    - Fix: drop eq. 10 and use L_S (eq. 20) everywhere.
  - N(D) is a set of cores, while N is the number of sites. On p. 35, N also stands for Furman et
    al.'s number of cores, and L for the homogeneity degree, which clashes with the loss.
    - Fix: write V_D for the cores that carry D, and translate cited symbols into the report's
      notation.
  - Smaller clashes:
    - α: the function-space scale σ^K, and also an exponent σ^α (p. 49, 6.6).
    - γ: an amplitude (4.2), a growth exponent (4.4, "Reading k"), the exponent of P(smaller first)
      (Table 8), c_γ (5.5) and γ_e (6.2).
    - δ: δ_θ, δ_Θ, δ_* and δ_K.
    - p: the parent p(v), fitted exponents, the Gram scale p = χσ²/d² (3.2), and p-values.
    - m: small cores (4.2), |N(D)| (4.5), messages m_v (eq. 2), mask entries m_j, and the leg
      dimensions m_ℓ (5.4).
    - κ: κ_K, κ_n, κ_e and κ_c.
    - σ also denotes singular values, σ_m(M_e) (6.2).
  - Fix: add a one-page notation table (symbol, meaning, where defined). It replaces the reference
    to `macros.tex` in "Conventions" (p. 3), which an outside reader does not have.
- **"Stalled" has three definitions.**
  - Table 4: L(T) ≥ 10⁻³ L(0) when the budget ends.
  - 4.6, odeco runs: the budget ended with the route-end slices above 10⁻⁸σ.
  - 4.6, pattern runs: every gradient entry stays below 10⁻¹² for 5,000 evaluations.
  - Related labels add to the confusion: "trapped", "dead", "unresolved", "stalled at a floor",
    "stalled at a partial fit".
  - Fix: keep Table 4's label as the umbrella term, with three subclasses: *dead route* (an end at
    roundoff, so the route cannot escape), *slow* (a live route whose budget ran out) and *floor*
    (an unrealizable centred target). Drop "trapped". The clear list of four causes of stalls in 5.3
    (p. 54) belongs in 2.7, before 4.6 uses the word.
- **Event times.**
  - Different sections time events differently: final rise (2.7, 4.1); last upward crossing, called
    "persistent" (4.4, 5.5); first crossing (4.5); onset of the bond spectra (6); loss-rate peaks
    (6.9).
  - Fix: say once in 2.7 that experiments use different rules, that the rules agree unless a
    component dips or a transient crosses ½, and that Table 17 lists the rule per experiment. Then
    delete the list of rules per experiment from 2.7.

### HI-3. The central mechanism, routes, gets no intuition and no worked example in the main text

**Where.** The Section 4 introduction (p. 26), 4.1 "Prediction" (p. 27) and 4.4 (pp. 33–34).

**Problem.**
- Routes are defined as "connected sets of cores whose new entries combine into a term proportional to
  the residual at no Gauss–Newton cost (Lemma A.16)". The phrase "at no Gauss–Newton cost" is opaque.
- The main text never explains why a later plateau behaves like the first plateau of a smaller TTN.
  That argument is only in App. A.3.
- The best intuition is in the report but scattered. The scalar trichotomy of 4.2 (m small cores give
  O(1), ln(1/γ), γ^{-(m-2)}) is never connected to eq. (23). Teacher B4 is on p. 39.

**Fix.** Open Section 4 with a box of about half a page, "How to read every plateau law", plus a small
figure. Proposed text:

> *Toy model.* A product of m scalars started at γ, ȧ = a^{m−1}(s − a^m), reaches O(1) in time O(1)
> for m = 1, in time ln(1/γ)/s for m = 2, and in time γ^{−(m−2)}/((m−2)s) for m ≥ 3. The exponent of
> every plateau law in this report comes from this model, with m the number of cores that must move
> together. The prefactors and the k = 2 head start need the full analysis.
>
> *Why routes.* At a saddle that has learned r components, every bond uses r channels and the other
> channels are zero. A new channel opened in one core meets zeros in its neighbour and leaves W
> unchanged. W moves only when every core of a connected set C opens the channel together, and then by
> a product of |C| small entries. Near the saddle the residual is therefore fitted by a fresh |C|-core
> TTN that starts near zero, so the escape follows the first-escape law with K replaced by |C|. Let
> k* be the size of the smallest such set with a nonzero residual. The next plateau is then absent
> (k* = 1), lasts ∝ ln(1/σ) (k* = 2), or lasts ∝ σ^{−(k*−2)} (k* ≥ 3).
>
> For two components that differ on the sites D, let m be the number of cores that carry a site of
> D. Then k* = k, the number of cores in the Steiner subtree of those m cores, which also includes
> the cores on the paths between them.
>
> *Example (teacher B4: patterns 0000, 0110, 1001 with weights 1, 0.3, 0.6).*
> - On MPS4, component 0110 differs on sites 1 and 2, which sit on adjacent cores, so k = 2 and the
>   plateau is logarithmic. Component 1001 differs on sites 0 and 3, so its route is the whole chain:
>   k = 4 and the plateau is ∝ σ^{−2}. The smaller component is learned first: *locality beats
>   weight*.
> - On binary4, both components need the two bottom cores and the root, so k = 3 for both, and
>   each comes after a σ^{−1} plateau. (Section 4.4 does not say in which order binary4 learns
>   them; check this before using the example.)

Draw the two routes on MPS4 and on binary4 with the Steiner edges highlighted, and use the same
picture as the teaser figure in Section 1 (HI-5).

### HI-4. The saddle-order theorem comes without intuition, before the experiments, and too large in the summary

**Where.** Items 4–5 (pp. 6–7); 4.4 "When the saddle order is k*" (p. 34).

**Problem.**
- These passages rely on phrases that cannot be parsed without Def. A.29: "generalized piggyback
  operator", "sets whose Steiner subtrees have fewer than k*/2 cores", and "new entries that point
  at a non-residual pattern together with rotations of two learned cores".
- The theorem sits in the middle of the route rule's prediction, before any data.

**Fix.**

(i) Explain the piggyback idea in words. Proposed text:

> Besides opening new channels, the flow can rotate the site vectors of single learned cores,
> "piggybacking" on the learned channel. Two such rotations together produce a component that
> differs from the learned one on exactly those two cores.
>
> The piggyback graph has one node per pair (core, rotated pattern), and one edge of weight s_r per
> residual component r that differs from the learned one on exactly two cores. If the graph's
> largest eigenvalue exceeds the learned weight s_j, the learned saddle is strict. This happens when
> a larger component is still waiting, or when several components share a core (a "star").
>
> From k* = 5 on, small clusters of cores can piggyback in the same way and lower the order below
> k*.

(ii) Give the theorem its own subsection after the two-component tests, for example "When is k* the
saddle order?". 4.4 is 9 pages long, so this split also helps navigation. State the theorem
informally in four lines, describe the MPS6 counterexample in one sentence, and leave the formal
statement in A.4. The counterexample paragraph on p. 34 repeats A.4 ("Counterexample with a positive
semidefinite Hessian").

(iii) Shrink item 4 of the introduction to:

> Theorem: at a saddle with one learned component j, the saddle order is exactly k*, unless a
> "piggyback" coupling of the residual on small sets of cores exceeds s_j. This never happens for
> two-component teachers learned in weight order. From k* = 5 on it can happen, and then the order
> drops below k* (MPS6 example).

Remove the red Pending placeholder from this item.

### HI-5. The introduction does not get a busy reader to the results in 5 minutes

**Problem.**
- "The answer in one paragraph" (p. 4) carries 20 status tags in 17 lines, plus jargon
  (Steiner subtree, route, cut dimension, ε, ALS/DMRG).
- Table 1 uses undefined symbols and terms: T(Z), Λ_r s_r and Λ_r^{1/2} s_r, m, ϱ, "gap A/σ + B",
  "route-amplitude share" and "centred model".
- The ten numbered findings run for 3.5 pages (pp. 4–9) with run counts and error ranges, e.g.
  "4·10⁻⁸ to 2.2·10⁻³ ... 231 initialization–teacher pairs" and "1,847 of 1,848 runs".
- Novelty is stated three times within Section 1: the Table 1 column, a "Novelty:" sentence per
  item, and the "What is new" paragraph (p. 11).
- Red Pending placeholders sit inside items 4 and 8 (pp. 7, 8). Others are on p. 24 (3.3) and in
  App. C. App. C still shows "[Skeleton ...]", and C.1 is empty, although Table 26 exists.
- There is no figure before p. 13.

**Fix.**
- (a) Add a teaser figure on pp. 3–4 with three panels:
  - the route example of HI-3 on MPS4 and on binary4;
  - the top-left panel of Fig. 3 (loss in linear time, with the partial-fit levels);
  - the left panel of Fig. 7 (measured exponent against k − 2).
- (b) Replace the one-paragraph answer with six plain bullets without inline tags (Table 1 carries
  the tags). For example:
  - From small initialization, gradient flow learns one component at a time. Each plateau is a
    critical point with the bond ranks fixed, and each escape adds one rank along a set of cores
    (a route).
  - The first plateau lasts ∝ σ^{−(K−2)}, with a prefactor that can be computed for each
    initialization before training. The estimate from the initialization norm is only a lower bound
    and can be off by a factor of up to 3·10⁷.
  - The tree sets the later plateaus. The next component needs the k* cores of the smallest subtree
    that touches the sites where it differs, and the plateau is then absent, ∝ ln(1/σ), or
    ∝ σ^{−(k*−2)}. Hence locality beats weight, and the same teacher has different plateau laws on
    an MPS and on a binary tree.
  - Which component comes first is a race set by the initialization when the two components differ
    on three or more cores. When they differ on two cores, the order becomes greedy as σ → 0.
  - At larger σ, training becomes rich and then lazy. A lazy regime exists only when every bond
    reaches its cut dimension.
  - Optimizers:
    - GD with steps below 2/(K s_1^{2−2/K}) keeps the saddle sequence.
    - SGD lengthens later plateaus and can trap.
    - Adam keeps the order but removes the σ-dependence of the first escape.
    - ALS, DMRG and Riemannian GD remove the plateaus.
    - The reason is which invariance each method has; move the 5.6 "Interpretation" paragraph and
      the "landscape vs clock" paragraph (pp. 51–52) here as one sentence each.
- (c) In Table 1, replace the symbols by words or define them in the caption.
- (d) Cut each numbered finding to at most five lines: the claim, the scale of the evidence, and the
  tag. Move the numbers into the sections, and fold the "Novelty:" sentences into 1.3 (see the cut
  list).
- (e) Fill in or delete every placeholder.

### HI-6. Lab-notebook material in the main text

**Problem.** The main text contains material an outside reader cannot use:

- internal paths:
  - `notes/literature.md` and `notes/reviews/citation_check.md` (p. 9);
  - `notes/model.md` and `notes/decisions.md` (p. 15);
  - `notes/diagnostics.md` (pp. 18–19);
  - `notes/theory_regimes.md`, "R4 and R12" (p. 55);
  - `notes/real_data.md` (p. 76) and `notes/reviews/real_data_round1.md` (p. 77);
  - five note paths in Section 7.
- code names: `Recorder`, `merged_rows`, `runner.summary`, `spectral_events`, `classify_regime`,
  `project_out_gauge` (pp. 17–20).
- data files in table captions: `main_exponents.csv`, `sigma_fits.csv`, `stall_test_summary.csv`,
  `large_chi_slopes_tensor.csv`.
- review history:
  - "The research plan's first guess" (p. 30);
  - "(verification review)" (pp. 32, 44, 45);
  - "theory note" (p. 32);
  - "the pilot's code", "the pilot's conjecture" (pp. 48, 50);
  - "the review of this experiment" (p. 62);
  - "a 'slow crossover' ... is withdrawn" (p. 68);
  - "This repeats the theory's own check" (pp. 70, 73);
  - "Withdrawn:" (p. 74);
  - "Narrowed or withdrawn after review" (pp. 77–78);
  - "as an earlier cutoff of 10⁻¹⁰ did" (p. 20).

These distract from the claims and read as defensive.

**Fix.** Make one pass over the main text:
- Keep the claims and their scope.
- Move paths, code names and the correction history to App. B and C, for example as a short
  "Provenance and corrections" subsection in C.
- Shorten "Data: experiment X, sweep Y (Table 26)" in the captions to "(source: Table 26)".

### HI-7. Section order and repetition hide the main thread

**Problem.**
- Section 3 (the regime map, 6 pp.) comes before the saddle-to-saddle section that it relies on: the
  S2S signature, floors, stalls, and Props. A.46–A.48.
- Section 5 then returns to the same regimes, which duplicates material:
  - 5.1 "Lazy only with full bonds" repeats 3.2 "No lazy regime below the cut dimension";
  - 5.3 "Stalled runs" repeats 3.3 "Stalls on the saddle-to-saddle side" and 4.6;
  - 3.6 "Step size" repeats 5.5.
- Other repetitions:
  - Novelty is stated five times: the Table 1 column, the item "Novelty:" sentences, 1.3 "What is
    new", and the "What is known and what is new" paragraphs of 4.4 (0.6 p.) and 4.5 (0.4 p.).
  - Prop. 2.4 (aligned initialization, eqs. 12–14) is restated in 4.2.
  - 4.3 "Which component escapes first" anticipates 4.5.
  - The probe-kernel ratio 0.54–1.11 is given on both p. 18 and p. 21.
- Table 10 (2 pp.) summarizes Sections 3–6 but sits at the end of Section 4, before the reader has
  seen Sections 5–6. It also overlaps Tables 1 and 5.

**Fix.**
- Reorder the sections:
  1. Setup.
  2. Saddle-to-saddle, the main section.
  3. Regime map and other regimes: merge Section 3 with 5.1–5.4.
  4. Steps, noise and optimizers (5.5–5.6).
  5. Covariance and teachers.
  6. Synthesis and open questions: Table 10 plus the "landscape vs clock" paragraph.
- Merge the novelty paragraphs into 1.3.
- State Prop. 2.4 once.
- Merge 5.3 into 4.6, and delete 3.6.

## Medium priority

- **MED-1. Status tags clutter the prose.**
  - Example, p. 22: "The lazy boundary. [empirical] confirming [derived] Proposition A.47 [known](the
    criterion ...) [new](the TTN constants)".
  - Fix: one tag at the head of each paragraph. Put known/new and the references at the end of the
    paragraph. Use no tags in the narrative of the introduction.
- **MED-2. 4.4 is 9 pages long, with 4 figures, 1 table and the theorem.**
  - Fix: split it into three subsections:
    - the route rule with its two-component tests (Table 7, Figs. 7–8);
    - the theorem (HI-4);
    - several components and bond spectra (Figs. 9–10).
- **MED-3. 4.3 cannot be read without the appendix.**
  - Examples: "its norm blows up at the time (30)", "T(Z) = t̂_1(Z)/s_1 of Corollary A.5", "δ_K from
    (31)".
  - Fix: state the decomposition of eqs. (29)–(30) in the main text (about 0.1 p.). Proposed text:
    > Along θ̇ = ∇g, with dτ = ‖θ‖^{K−2}dt, the direction follows the spherical gradient flow of g,
    > and d ln‖θ‖/dτ = K g(θ̂). Hence ‖θ‖ blows up at
    > t* = ‖θ_0‖^{−(K−2)} ∫_0^∞ e^{−(K−2)K G(τ)} dτ, where G(τ) = ∫_0^τ g(θ̂).
    > With θ_0 = σZ this is σ^{−(K−2)} T(Z).
- **MED-4. Figures that send the reader into the text.**
  - Fig. 10: its legend lists 0-based sets of "sites below" each edge, and no tree is drawn. Add a
    small picture of each tree with the Steiner edges coloured.
  - Fig. 11: the legend uses m, "adj." and "LO race / fit". Spell these out in the caption.
  - Fig. 13: "route-end slices" is undefined (HI-1).
  - Fig. 16: η*_ms(B) and c_γ are not defined in the caption.
  - Fig. 20: the y-axis is γ_e[t_esc(ε) − t_esc(ε_min)] + ln(1/ε_min), and γ_e is defined only through
    M_e from A.5. Label it "escape time rescaled by the predicted slope".
  - Fig. 2(b): dots, bars, dashed lines and hatching are explained only in a 13-line caption. Add an
    in-panel legend and split the caption by panel.
  - Float pages 28, 37, 41 and 56 are half empty. Use [t] placement, and merge Figs. 7 and 8, which
    come from the same sweep.
- **MED-5. Anchor the DLN reader.**
  - In 2.2, say that binary4 is a three-factor matrix product of the 01|23 matricization, W = A B Cᵀ.
    All binary4 results are therefore checks against DLN theory, and the tensor-specific behaviour
    starts with MPS4, binary8 and HT4.
  - Say that tree depth is not network depth: binary4 has tree depth 1 but is called a "depth-3
    linear network".
  - At present this is only in Table 1's caption and Remark A.23.
- **MED-6. Motivate the centred model where it is introduced (2.4(b), 3.1).**
  - Proposed text: "At large σ the plain model starts with a large function f_{θ_0}. Centring removes
    it, so that σ changes the dynamics but not the target."
  - Also, "plain model" is never defined.
- **MED-7. Reword 1.1(i).**
  - Current text: "it can differ from what has been learned on a few sites only, carried by a few
    cores".
  - Proposed text: "a new component can differ from what has been learned on a few input positions
    (sites) only. Then only the cores that hold those sites, and the cores between them, need to
    change."
- **MED-8. Remove internal process from 1.3.**
  - Replace "audit ... notes/reviews/citation_check.md; about 270 searches" with "a systematic
    literature search (about 270 queries, arXiv to October 2026; App. C)".
  - Expand ALS, DMRG and HT at first use.
- **MED-9. Teacher labels need definitions where they are used.**
  - TB, N4 and E8 appear in the main text (4.4, 5.6) without their patterns, which are given only in
    App. B (Table 16). TA, B4, TC and TD have their patterns inline.
  - Fix: give the patterns at first use, or add a seven-row teacher table in 4.4.
- **MED-10. "star3" (a tree) clashes with "star teachers" (6.8, item 5).**
  - Rename the tree, e.g. "tripod6".
- **MED-11. Trim number-heavy passages that bury the message.** The following paragraphs report
  numbers at a precision the argument does not need:
  - 3.2 "Both boundaries scale like χ^{-1/2}" (p. 22: χ² values and four kinds of slopes);
  - 4.1 "Plateaus at the rank-constrained partial fits" (six deviation numbers, p. 27);
  - 4.5 "After a wrong first escape" (λ_min to five digits, p. 45);
  - 4.6, which is entirely of this kind (pp. 46–50).
  - Fix: one representative number per claim, and the rest in App. B.

## Low priority

- **LOW-1.** "Built-in topologies" (2.2) and the "custom" row of Table 2 are software language.
- **LOW-2.** Say "singular values (Schmidt values)" once, then use "singular values" throughout.
- **LOW-3.** Expand the abbreviations GF, LO, S2S, SP and NTK before their first use in a figure or
  table.
- **LOW-4.** 2.8, "What the results may depend on", is a list of limitations. Move it to the end of
  1.1 or to Section 7.
- **LOW-5.** "Head start" is first used in item 3 (p. 6) but defined on p. 34. Add "(the route
  already grew during the previous plateau)" at first use.
- **LOW-6.** Eq. (20), L_S, repeats eq. (10), L_k.
- **LOW-7.** Round main-text numbers to 2–3 significant digits (e.g. −0.18559 vs −0.18566 on p. 45).
- **LOW-8.** Define the "×/" notation for geometric error factors (3.1).
- **LOW-9.** Rephrase the defensive aside on p. 39 ("it is not a prediction success rate") as a
  plain statement of what the check tests.
- **LOW-10.** The abstract never says what the saddle order is. Add "(the power of the distance at
  which the loss first decreases)".

## Cut and move list (main text → appendices)

Each move keeps every result: either the material is a duplicate, or it goes to App. A/B with a
pointer. Savings are net pages of the main text, estimated from the current layout. Core moves
(**C**) give the best ratio of pages saved to effort; optional moves (**O**) come after.

| # | where (pp.) | cut or move | to | saved |
|---|---|---|---|---|
| X1 C | 1.2, items 1–10 (4–9) | Cut each item to ≤ 5 lines. Move the numbers to the sections and the "Novelty:" sentences to 1.3. | Sections 3–6, 1.3 | 2.2 |
| X2 C | 1.3, "What is new" (11) | Delete; it repeats Table 1. | — | 0.25 |
| X3 C | 1.2 placeholders (7, 8) | Fill with one sentence or delete. | — | 0.1 |
| — | Section 1 additions | Teaser figure and six bullets (HI-5). | — | −0.5 |
| X4 C | 2.3, "Closed form" bullets (14) | Move eqs. 8–9 and the loopy contraction; keep eqs. 7 and 10. | B.1 | 0.45 |
| X5 O | 2.4, proof of Prop. 2.2 (15) | Move the proof; keep the statement. | A.1 | 0.15 |
| X6 C | 2.4(c) and Remark 2.3 (15–16) | Move the storage of the two-copy model and the gradient mask; keep two lines. | B.1 | 0.4 |
| X7 C | 2.4, Prop. 2.4 + eqs. 12–14 (16) | State them once (4.2 already restates them); leave a one-line pointer here. | 4.2 | 0.5 |
| X8 C | 2.5, integrator paragraph and "Precision" (16–17) | Move them; B.1 already repeats most of this. | B.1 | 0.4 |
| X9 C | 2.6–2.7, Table 3, drift eq. 18, integrator check (18) | Move them; keep a one-line list of the recorded quantities. | B.1 | 0.5 |
| X10 C | 2.7, "Learning events" (18–19) | Move the rule variants per experiment and the bond-spectrum rules (a)–(c); keep one rule and a pointer to Table 17. | B.1 | 0.4 |
| X11 C | 2.7, Remark 2.6 and saddle-type implementation (19–20) | Move the cutoffs, history and fallback. | B.1 | 0.25 |
| — | Section 2 additions | Definitions (HI-1) and notation table (HI-2). The table could go in an appendix (−0.2). | — | −0.6 |
| X12 C | 3.1, Protocol (21) | Cut to three lines; the rest is in 2.4 and B.5. | B.5 | 0.35 |
| X13 C | 3.2, numbers (21–22) | Move the per-χ boundaries, χ² values, slopes, widths and extrapolations; keep one number per claim. | B.5 | 0.8 |
| X14 O | 3.3, "Beyond seven cores" and "Which scale stays fixed?" (24) | Cut to three sentences, and remove the placeholder. | B.5 | 0.5 |
| X15 O | 3.4, numbers (24–25) | Trim. | B.5 | 0.15 |
| X16 C | 3.5, threshold sensitivity (25) | Move it; keep "labels change for ≤ 7.1% of runs, always to a neighbouring regime". | B.5 | 0.35 |
| X17 C | 3.6 and 3.7 (26) | Delete 3.6 (it repeats 5.5); merge 3.7 into the introduction of Section 3. | — | 0.25 |
| X18 C | 4.1 (26–29) | Move the "Scope" run counts, the list of deviations, Table 6 and the seed-level saddle-type anecdotes; keep one sentence each. | B.2 | 1.4 |
| X19 C | 4.2 (29–30) | Move Fig. 4 and the "Test" paragraph, which validate the integrator. The trichotomy moves to the box at the start of Section 4 (HI-3). | B.2 | 0.8 |
| X20 C | 4.3 (32–33) | Cut "Test, seed by seed" to two sentences. Move "Dependence on d, χ and K" and Fig. 6, keeping three lines. Merge "Which component escapes first" into 4.5. Add the blow-up identity (MED-3, +0.1). | B.2, 4.5 | 1.4 |
| X21 C | 4.4 (33–42) | Fold "What is known and what is new" into 1.3. Trim the numbers in "head start" and "Saddle type". Move "Reading k from a single run" and "Independent cross-check". Merge Figs. 7 and 8, and fix the half-empty float pages 37 and 41. | 1.3, B.3 | 2.1 |
| X22 C | 4.5 (42–46) | Fold "What is known and what is new" into 1.3. Move Table 8. Cut the "tensor-like", "number of cores", "matrix-like", "after a wrong first escape" and "three and four components" paragraphs to their claims, with one number each. | 1.3, B.4 | 1.5 |
| X23 C | 4.6 (46–50) | Keep about 1.8 pp.: the definitions, Fig. 13 or Table 9, the mechanism in four lines, and three findings. Move "Stalled runs: slow where tested", "Effect on the exponents", "Relation to the conservation-law trap" and "Steep MPS routes". | B.2, B.3 | 2.2 |
| X24 O | 3.7 Table 5 (25); 4.7 Table 10 (50–52) | Drop Table 5, whose rows are in Table 10 and B.5; this saves 0.5 p. in Section 3. Move Table 10 and its two paragraphs from 4.7 to the final synthesis, with no net change in length. | Section 7 | 0.5 |
| — | Section 4 additions | The "How to read every plateau law" box and the route figure (HI-3). | — | −0.6 |
| X25 O | 5.1, "Lazy only with full bonds" (52) | Merge with the 3.2 paragraph that says the same. | — | 0.3 |
| X26 O | 5.2, "Transients" and "What sets the crossover" (53) | Move the event-rule motivation to 2.7 or B; cut the repeat of 3.2. | 2.7, B.5 | 0.35 |
| X27 C | 5.3, Stalled runs (54–55) | Merge into 4.6 and drop the duplicates; move Table 11. | 4.6, B.5 | 1.0 |
| X28 O | 5.4, Table 12 and the whitespace on p. 56 | Move Table 12; the numbers are in the text and Fig. 14. | B.5 | 0.5 |
| X29 C | 5.5 (57–62) | Keep about 2 pp.: eq. 25, GD as GF on a shifted clock with Fig. 15, SGD never shortens a plateau and can trap (eq. 26), one sentence on finite samples, and "When does gradient flow describe practical training?". Move the edge-of-stability and catapult details, SGD stability and forgetting, Figs. 16–17 and the numbers. | B.7 (or a new App. D) | 3.0 |
| X30 C | 5.6 (62–66) | Keep Fig. 18, Table 13, "Interpretation", and short Adam and sweep paragraphs. Move Fig. 19, the ε-law test, "Later plateaus and minima under Adam", the re-gauged and Gram details, and "Finite samples". | B.8 | 2.0 |
| X31 C | 6.3–6.5 (68–72) | Merge into one subsection, "Any non-orthogonality turns the MPS escape logarithmic", with one table of predicted against measured slopes. Fig. 21 already shows all three cases. | — | 1.2 |
| X32 O | 6.2, Table 14 (69) | Move it; Fig. 20 carries the message. | B.6 | 0.5 |
| X33 C | 6.8 "Withdrawn", 6.9 "Narrowed or withdrawn after review" and "Deviations" (74, 76–78) | Move them. | B.9 / App. C | 1.0 |
| X34 O | 6.10 (78) | Trim to three lines; keep Table 15. | — | 0.15 |
| X35 O | 7 (78–82) | Merge the 14 items into about 8 (2+5, 7+8, 9+10, 11–13), remove the note paths, and cut "What would settle it" to one sentence each. | — | 1.3 |
| X36 O | all captions | Shorten "Data: experiment …, sweep … (Table 26)" to "(source: Table 26)". | Table 26 | 0.3 |

### Totals (net of additions)

| section | now (pp.) | core saving | optional saving | after |
|---|---|---|---|---|
| 1 Introduction | 8.3 | 2.05 | – | ≈ 6.3 |
| 2 Setup | 9.2 | 2.3 | 0.15 | ≈ 6.8 |
| 3 Regime map | 5.8 | 1.75 | 1.15 (incl. Table 5) | ≈ 2.9 |
| 4 Saddle-to-saddle | 25.9 | 8.8 | – (Table 10, 1.8 pp., moves to 7) | ≈ 15.3 |
| 5 Other regimes, steps, optimizers | 14.1 | 6.0 | 1.15 | ≈ 7.0 |
| 6 Covariance, teachers, real data | 12.3 | 2.2 | 0.65 | ≈ 9.5 |
| 7 Synthesis and open questions, plus captions | 3.4 | – | 1.6 | ≈ 3.6 (with Table 10) |
| **total, pp. 3–82** | **≈ 79** | **≈ 23** | **≈ 4.7** | **≈ 56 (core only) / ≈ 51 (all)** |

- The core set alone saves about 23 pages, inside the 15–25 page target. X1, X13, X18–X23, X29 and
  X30 alone give about 17 of them.
- Adding the optional set brings the saving to about 28 pages.
- The references (pp. 82–88) are untouched.
- A main text of about 52–56 pages would be organized as follows:
  - Introduction, about 6 pp.
  - Setup, about 7 pp., including the definitions and the notation table.
  - Saddle-to-saddle, about 15–16 pp.:
    - the box that explains every plateau law;
    - phenomenology;
    - first escape;
    - route rule;
    - the theorem;
    - several components;
    - order of learning;
    - failure modes.
  - Regimes and optimizers, about 10 pp.
  - Covariance and teachers, about 9.5 pp.
  - Synthesis and open questions, about 4 pp.
