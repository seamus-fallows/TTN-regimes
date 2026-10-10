# Verification: `notes/theory_saddle_order.md` (v1, commit 13d2a25) - proofs and counterexamples

Verifier: adversarial maths verifier, 2026-10-10. Scope: every step of sections 1-4, the corollaries,
the counterexamples (independent numerics), the dynamics claim of section 6 (light check), and an
attempt to break Theorem 1. The code is independent: it does not import `ttn/` or the note's scripts, and the note's numerics folder was not used.
Four scripts in the session scratchpad (not committed): `common.py` (MPS_N / HT4 contractions, torch float64),
`checks.py` (gradient, exact Hessian, $k^*$, explicit curves), `sphere.py` (sphere minima), `gf.py` (gradient-flow escape).
Total compute about 15 CPU-minutes.

## Verdict in brief

- **Theorem 1 is correct and proven.** Lemma 1, the cumulant algebra and Lemma 2 (a cumulant on $B$ needs new entries on every core of $\mathrm{Steiner}(B)$) all check out, as does the absorption of disconnected blocks by $\mathcal S$. Gauge and mixed old/new curves are handled. The constant is uniform. No counterexample was found, by analysis or by numerical stress tests.
- **Theorem 2 is correct (proven)** under its hypotheses. Add $s_r>0$ to them, and state "pattern teacher, index condition" in section 0.
- **Gap (main finding).** The exact orders $2\ell^*$ (S7; S8 "order 4/4/6"; Cor. 2c; Cor. 3) need a lower bound that the note does not state. Theorem 1 does not apply there, because $\lambda_{\max}(\mathcal S)>s_q$. The fix is a level version, Theorem 1' (section 6): if $\lambda_{\max}(\mathcal S\vert_{\mathcal L_{\ell-1}})<s_q$ then the order is $\ge\min(k^*,2\ell)$. The proof is the same; I verified it, and with it all three exact orders are proven.
- **Counterexamples reproduced independently.** MPS6 pair-star: $k^*=5$, $\nabla L=0$, Hessian PSD (eigenvalues 0/1/6), explicit curve and sphere minimum both $-0.03500\,t^4$ (slope 4.000). MPS6 $110001$ (1.5): $-0.1562\,t^4$. HT4 $1111$ (1.5): $-t^6/432$. The pair-star refutes the multi-residual use of "order $=k^*$"; the two single-residual examples refute R2 literally.
- **Dynamics ($\sigma^{-2}$) confirmed** on two more seeds (exponents 2.04, 2.02 and 1.96, 1.98). It stays **derived**.
- **Factual error.** Section 7.5's "$N\le4$: nothing changes ($k^*\le4$)" is wrong for ht4 ($K=7$). There are also minor fixes: S4 range, the gauge sentence in section 5, the "iff" wording proposed for CLAUDE.md, and the scope of "counterexample to R2".

## 1. Definitions (note section 1)

| item | verdict | comment / fix |
|---|---|---|
| Sectors $\Pi_B$, $T(B)$, $T(B_1)\otimes T(B_2)\cong T(B_1\sqcup B_2)$, $R_\emptyset=0$ | correct | The identification is canonical (tensor factors are labelled by cores), so the product of 3.1 is commutative and associative. Cores without sites (HT internal nodes, binary root) have $D_v=1$, $q_v^\perp=0$, so $T(B)=0$ whenever $B$ contains one; everything below stays consistent. |
| Criticality iff $R_{\{v\}}=0$ for all $v$ | correct | Re-derived: new entries meet bond index 0 of every neighbour, so they do not move $W$ at first order; an old entry of $v$ moves $W$ inside $T(\emptyset)\oplus T(\{v\})$, and $R_\emptyset=0$. |
| $k^*$; pattern-teacher reduction $R_B=\sum_{T_r=B}s_rU_r$ | correct | Distinct patterns on a core are orthogonal basis vectors, so $\Pi_BU_r=U_r$ iff $B=T_r$. |
| degree, low sets | correct, cosmetic | $\deg C=\lvert\mathrm{Steiner}(C)\rvert$ for every $C$ (the singleton clause is redundant). "two adjacent cores, or one core plus a neighbour" names one case twice. |
| $\mathcal S$ | correct with fix | Symmetric because $*$ is commutative; zero diagonal blocks, so trace 0 and $\lambda_{\max}\ge0$. In the pattern basis it is the weighted adjacency matrix and $\lambda_{\max}=\rho$ by Perron-Frobenius **only if all $s_r\ge0$**. Fix: state $s_r>0$ for pattern teachers. With signed weights $\lambda_{\max}\le\rho(\lvert A\rvert)$, and Theorem 2's $x_n^{1/j}$ would need sign bookkeeping. |
| saddle order | correct with fix | (a) "largest $p$": the set of admissible $p$ is closed at the top for a polynomial $L$. The curve equivalence needs the curve-selection lemma ($L$ is polynomial, so semialgebraic); cite it. (b) Gauge: the order defined with $\lVert\theta-\theta_S\rVert$ equals the order defined with $\mathrm{dist}(\theta,\text{orbit})$. The reason: every orbit point $g\cdot\theta_S$ with $g$ near the identity is a saddle of the same type in a transformed bond basis, so Theorem 1 holds there with a locally uniform constant ($\lVert g^{-1}\rVert^{k^*}$). Section 5's justification ("distances to $\theta_S$ and to its gauge orbit agree up to constants") is **false as written**: a point on the orbit has distance 0 to the orbit. Replace it by the argument above. |

## 2. Lemma 1 (exact identity): correct

Re-derived line by line. Sector orthogonality gives $\lVert W-W^*\rVert^2=\sum_B\lVert\Pi_B(W-W^*)\rVert^2$. Then
$\Pi_\emptyset W=NU_q$, $\Pi_\emptyset W^*=s_qU_q$, $\Pi_BW^*=R_B$ ($B\ne\emptyset$), and
$L_S=\tfrac12\sum_{B\ne\emptyset}\lVert R_B\rVert^2$. The identity needs only $N\ne0$, which holds near
$\theta_S$. **proven.**

## 3. Section 3.1 (algebra, cumulants): correct

- $\mathcal A$ is a commutative unital $\mathbb Q$-algebra, and $f_\emptyset=0$ implies $f^{*(K+1)}=0$.
  So $\exp$ and $\log(1+\cdot)$ are finite sums, and $\log(fg)=\log f+\log g$ holds (a formal power-series identity in commuting nilpotents).
- $(\exp\kappa)_B=\sum_n\frac1{n!}\sum_{\text{ordered }(B_1..B_n)}\otimes\kappa_{B_i}=\sum_{\pi}\otimes_{C\in\pi}\kappa_C$, and the inverse is Möbius inversion on the partition lattice. Both formulas are right.
- $\kappa_{\{v\}}=m_{\{v\}}$ vanishes at $\theta_S$ (criticality is not needed for this: $\Pi_{\{v\}}W_S=0$). $\kappa_B$ is rational in $\theta$ with denominator $N^{\le\lvert B\rvert}$, so it is analytic near $\theta_S$.
- Remark: $m_{\{v\}}$ gets contributions both from old rotations of $v$ and from products of new entries (for example a new-index pair on an edge at $v$). The proof only uses $\lVert\kappa_{\{v\}}\rVert\le C_0\varepsilon$, so this is harmless.

## 4. Lemma 2 (locality of cumulants): correct. Status: **proven**

- (a) Factorization. If $\nu_x=0$, every nonzero entry of $G_x$ has all bond indices 0, so
  $W=g_x\otimes\bigotimes_bW_b$ exactly. The projectors are products over cores, so
  $m=m^{(x)}*\prod_bm^{(b)}$ in $\mathcal A$. Each factor has $\emptyset$-component 1 (it is normalized by $q_x^\top g_x$ and $N_b$, both near their nonzero $\theta_S$ values). The support of each factor's
  $\log$ lies in the subalgebra of families supported on subsets of $\{x\}$ or of one $V_b$; these subalgebras are closed under $*$. The topological step is also right: if $x\in\mathrm{Steiner}(B)\setminus B$ and $B\subseteq V_b$, then $\mathrm{Steiner}(B)\subseteq V_b$ because $V_b$ is a subtree, which is a contradiction. So $\kappa_B=0$ on $\{\nu_x=0\}$. This is the precise sense in which a cumulant on $B$ needs new entries on every core of $\mathrm{Steiner}(B)$; the claim is **correct**.
- (b) Bound. $\kappa_B$ is analytic on a complex polydisc where $N\ne0$. It vanishes on the real slice
  $\{\nu_x=0\}$ near $\theta_S$, so the restricted series is identically 0. Hence every monomial with a nonzero coefficient contains a $\nu_x$ variable for each $x\in\mathrm{Steiner}(B)$. The variable sets $\nu_x$ are disjoint, so one variable can be factored out per $x$. The remaining sum is bounded on radius $r_0$ by $r_0^{-\lvert\mathrm{Steiner}(B)\rvert}\sum_\alpha\lvert c_\alpha\rvert r_0^{\lvert\alpha\rvert}<\infty$ (absolute convergence on radius $2r_0$). Correct.
- Edge cases checked: a core of $\mathrm{Steiner}(B)$ with no new entries ($\chi=1$ on all its edges) gives
  $\kappa_B\equiv0$, which is consistent. New entries of $x$ on edges outside $\mathrm{Steiner}(B)$ count in $\nu_x$, so the bound is weaker but still true.
- Relation to F2 of `theory_saddles.md`: the remark is right. Disconnected $J$ appear only as products of lower cumulants, which section 3.4 handles.

## 5. Theorem 1 (proof in 3.4): correct. Status: **proven**

Every inequality was re-derived.

- Setup. For $k^*\ge3$, high sets have $\lvert C\rvert\ge2$, so Lemma 2 gives $y_C\le C_0\varepsilon^{\lvert\mathrm{Steiner}(C)\rvert}\le C_0\varepsilon^{k^*/2}$ for $\varepsilon\le1$. For $k^*=2$ the statement is the trivial $-C\varepsilon^2$.
- (i) GN, low sectors. Correct: $C\subseteq B$ implies $\mathrm{Steiner}(C)\subseteq\mathrm{Steiner}(B)$, so every block of a partition of a low $B$ is low. Factors beyond the first are bounded by $C_0\varepsilon$, and powers $(C_0\varepsilon)^{n-2}$ go into $M$. The inequality $\lVert u+v\rVert^2\ge(1-\varepsilon)\lVert u\rVert^2-\varepsilon^{-1}\lVert v\rVert^2$ is right. Dropping the high-sector GN terms is legitimate (coefficient $N^2/2>0$).
- (ii) Gain. No low $B$ has $R_B\ne0$ (that needs $\lvert\mathrm{Steiner}(B)\rvert\ge k^*$), which is consistent with the split. The low-low two-block terms equal $\tfrac12\langle\kappa_{\mathcal L},\mathcal S\kappa_{\mathcal L}\rangle$ **exactly**: each unordered pair of disjoint low sets occurs once, for $B=C_1\sqcup C_2$. The low-high term uses $y_{C_2}^2\le C_0^2\varepsilon^{k^*}$, high-high gives $\varepsilon^{k^*}$, and $\lvert\pi\rvert\ge3$ gives either $\varepsilon\times$ a quadratic form or at least two high blocks. All correct.
- (iii) Correct. Set $g=s_q-\lambda_{\max}$ and $\gamma=g/(4M)$, and use $N\in[s_q-C_3\varepsilon,2s_q]$.
- **Uniformity of $C$.** $C=2s_qC_1(g/4M)$ comes from Lemma 2's coefficients on a fixed polydisc and from combinatorial factors that depend only on $K$. It holds on a ball of radius $r_1(g)>0$, so it is uniform near $\theta_S$. It blows up and $r_1\to0$ as $g\to0$, as the note says.
- **Gauge directions.** They are not needed. The bound holds for every $\theta$ in a ball, including gauge-orbit points, where $L-L_S=0$. For the order with respect to the orbit, see section 1.
- **Mixed old/new curves.** These are covered: old rotations enter only through the singleton cumulants. Mixed polymer-plus-rotation structures (the pair-star) are exactly the off-diagonal blocks of $\mathcal S$ between multi-core low sets and singletons. Nothing in the proof assumes old and new entries separate.
- Route clause (order exactly $k^*$): correct. With new entries only on the internal edges of $P=\mathrm{Steiner}(B)$, a contraction term that uses any new entry must use the new entries of all of $P$, so $W=W_S+t^{k^*}\Phi$ exactly and $L-L_S=-t^{k^*}\langle R,\Phi\rangle+\tfrac12t^{2k^*}\lVert\Phi\rVert^2$.

## 6. GAP: the exact orders $2\ell^*$ (S7, S8) use a lower bound the note never states

Theorem 1 gives a lower bound only when $\lambda_{\max}(\mathcal S)<s_q$ on **all** of $\mathcal L$.
That hypothesis fails in every situation where Theorem 2 applies, so as written the note proves only
**upper** bounds there: pair-star order $\le4$, MPS6 $110001$ (1.5) order $\le4$, HT4 $1111$ (1.5) order $\le6$, and S7's "order $=\min(k^*,2\ell^*)$". A PSD Hessian adds order $\ge3$, nothing more.
So the claims "order 4" (S8, Cor. 2c, Cor. 3) and S7's "proven" are not proven by the text.

**Fix (state and prove this; it is a two-line modification).** *Theorem 1' (level version).* Let
$\ell\ge1$, $\mathcal L_{\ell-1}=\{C:\deg C\le\ell-1\}$ ($\mathcal L_0=\emptyset$). If
$\lambda_{\max}(\mathcal S\vert_{\mathcal L_{\ell-1}})<s_q$, then
$L-L_S\ge-C\lVert\theta-\theta_S\rVert^{\min(k^*,2\ell)}$ near $\theta_S$.
*Proof.* Run 3.4 with "low" $:=\mathcal L_{\ell-1}$ and "high" $:=\{\deg\ge\ell\}$. Lemma 2 gives
$y_{\rm high}\le C_0\varepsilon^{\ell}$, so every term bounded by $\varepsilon^{k^*}$ through two high
factors (low-high after Young, high-high, $\lvert\pi\rvert\ge3$ with at most one low block) is now
$O(\varepsilon^{2\ell})$. The single-cumulant term stays $O(\varepsilon^{k^*})$. Low $B$ still have $R_B=0$, since $\deg B\le\ell-1<k^*$. Theorem 1 is the case $2\ell\ge k^*$. $\square$
(Verified step by step; I recommend tagging it **proven**.)

Consequences (with Theorem 2 for the matching upper bound):

| case | level used | lower bound | upper bound | order |
|---|---|---|---|---|
| MPS6 pair-star (0.8, 0.8), $k^*=5$ | $\mathcal S\vert_{\mathcal L_1}=S_1=0$ | $\ge4$ | $\le4$ (Thm 2, $\ell=2$, $\chi_{01}\ge2$) | **4, proven** |
| MPS6 $110001$ (1.5), $k^*=6$ | $S_1=0$ | $\ge4$ | $\le4$ | **4, proven** |
| HT4 $1111$ (1.5), $k^*=7$ | $\mathcal S\vert_{\mathcal L_2}=0$ (two leaves are never adjacent) | $\ge6$ | $\le6$ (two vertex-disjoint degree-3 polymers) | **6, proven** |

S7 then holds as stated if $\lambda_{\max}(\mathcal S\vert_{\mathcal L_{\ell^*-1}})<s_q$ strictly (that is what "generic" must mean), with $\ell^*$ the smallest level where $\lambda_{\max}$ exceeds $s_q$, plus the index condition for the upper bound. Note that $\lambda_{\max}(\mathcal S\vert_{\mathcal L_\ell})$ is nondecreasing in $\ell$, because each level is a principal compression of the next.

## 7. Theorem 2 (sharpness): correct under its hypotheses. Status: **proven** (pattern teacher, $s_r>0$, index condition)

- Polymer representation. Correct. With distinct index values per edge, the pair (edge, value) determines the node. So both endpoints of a nonzero-index edge use that node's entries, and by connectivity the nonzero-index component is exactly $P_n$, with every core of $P_n$ on node $n$'s entries. Active polymers are therefore pairwise vertex-disjoint.
- Every item (polymer $a^{j}\eta x_n$, rotation $a\eta x_{v,w}$) is **exactly linear** in $\eta$. So $m^{(1)}$ consists of single items and $m^{(2)}$ of compatible pairs, with no other $\eta^2$ terms. $N=s_q$ exactly, because every non-empty configuration has a non-$q$ factor.
- Compatibility of edges: correct. A shared vertex would make $\mathrm{Steiner}(C_1)\cup\mathrm{Steiner}(C_2)$ a connected set of $\le2\ell-1<k^*$ cores containing $T_r$. Compatible non-edges contribute 0. Gain $=\tfrac12x^\top A_Xx=\rho/2$, GN $=\lVert x\rVert^2=1$, which gives $-\tfrac{s_q}2(\rho-s_q)\eta^2+O(\eta^3)$. The index condition is sufficient. For a single residual the two parts have vertex-disjoint Steiner trees, so it reduces to $\chi_e\ge2$ on their edges.
- Hidden assumption: $s_r>0$. The Perron vector must be $\ge0$ for $(\eta x_n)^{1/j}$ to be real. With signed weights one can move signs onto one entry per item, but the note does not say so. Fix: add "$s_r>0$" to the hypotheses.
- Wording fix: section 0 states Theorem 2 without "pattern teacher" and "index condition". Add both there.
- The leading constants of the examples (Cor. 3 and group C of section 5) come from optimizing the reduced quadratic form, not from the Perron-vector curve. They are upper bounds on $m(t)/t^{2\ell}$, and my sphere minima reach them (section 9). That they are the exact leading constants is **empirical**, which is the right tag.

## 8. Corollaries

| claim | verdict | fix / status |
|---|---|---|
| Cor. 1 ($k^*\le4$) | correct with fix | Restrict to $k^*\in\{3,4\}$: for $k^*=2$ the order is 2 whatever $\rho(S)$ is, so S4's "$k^*\le4$: order $=k^*$ iff $\rho<s_q$" is false at $k^*=2$. "Iff" also needs the route ($\chi_e\ge2$ on a minimal Steiner tree). Equality: $k^*=3$ has order 3 (PSD Hessian, so $\ge-C\varepsilon^3$ by Taylor; the route gives $\le$), while the Corollary 1 text calls the boundary open without excepting $k^*=3$. Align it with S4. "Old block PD modulo old gauge iff $\lambda_{\max}(S_1)<s_q$" re-derived: the rotation block is $s_q(s_q-S_1)$ in relative variables, the radial block has rank one with $K-1$ gauge zeros, and the cross terms vanish because $R_\emptyset=R_{\{v\}}=0$. **Proven.** |
| Cor. 2 (one residual) | correct with fix | Each node has at most one partner, so $\rho\in\{0,s_r\}$. (a) and (b) follow from Theorem 1: **proven**, given the route. (c): the upper bound $2\ell^*$ is proven by Theorem 2; **exact** order $2\ell^*$ needs Theorem 1' (section 6). |
| Cor. 3 (where R2 fails) | correct with fix | The necessary conditions are right **given a PD old block** ($\lambda_{\max}(S_1)<s_q$); state that assumption. "First failures need $k^*\ge5$" holds off the boundary only, because $k^*=4$ with $\rho(S)=s_q$ is not excluded. The mechanism formula is exact on the curve family: $2(L-L_S)=y^2+z_4^2+z_5^2-2y(s_1z_5+s_2z_4)+z_4^2z_5^2+y^2(z_4^2+z_5^2+z_4^2z_5^2)$. |

## 9. Counterexamples: independent verification (numerics, own code)

Setup: $d=\chi=2$, $s_q=1$, $a=1$; exact gradient and Hessian by torch autograd (float64); $L-L_S$ in the
cancellation-free form $\tfrac12\lVert\Delta\rVert^2-\langle R,\Delta\rangle$. The sphere minimum
$m(t)=\min_{\lVert\delta\rVert=t}$ uses L-BFGS with 8 Gaussian starts plus continuation, objective scaled by $t^{-p}$.

**Main counterexample, MPS6, $q=0^6$, $r_1=110001$, $r_2=110010$ (0.8 each).**
- $k^*=5$: Steiner sizes 6 ($r_1$) and 5 ($r_2$). By hand: generalized graph = star centred on $(\{0,1\},11)$ with leaves $(\{5\},1)$, $(\{4\},1)$, $\rho=0.8\sqrt2=1.131$. The splits $\{0\}\vert\{1,5\}$, $\{1\}\vert\{0,5\}$, $\{0\}\vert\{1,4\}$, $\{1\}\vert\{0,4\}$ are not low.
- $\nabla L(\theta_S)=0$ exactly. Hessian (40 x 40) eigenvalues: $0$ (x33, min $-5.7\cdot10^{-17}$), $1$ (x6, rotations, $S_1=0$), $6$ (x1, radial $Ka^{2(K-1)}$). **PSD: confirmed.**
- Explicit curve $G_0[1,\alpha{=}1]=G_1[\alpha{=}1,1,0]=b$, $G_4[0,1,0]=G_5[0,1,0]=0.8b^2$:
  $(L-L_S)/t^4=-0.034039,-0.034772,-0.034944,-0.034986,-0.034996$ for $t=0.20\to0.0125$, local slopes
  3.970, 3.993, 3.998, 4.000. Analytic: $-\tfrac18(1.28-1)=-0.035$. **Order-4 descent: confirmed.**
- Sphere minimum: $m/t^4=-0.034988,-0.035006,-0.035002,-0.035001$ ($t=0.2\to0.025$), slopes 3.999, 4.000, 4.000.
  Nothing faster was found, which is consistent with Theorem 1' (order $\ge4$).
- Control (0.6, 0.6), $\rho=0.85$: slopes 5.004, 5.001, 5.000; $m/t^5\to-0.010734$ vs single route of $r_2$:
  $0.6\cdot5^{-5/2}=0.010733$. **Order $k^*=5$, as Theorem 1 says.**

**Other counterexamples.**
- MPS6, $110001$ (1.5): $k^*=6$, Hessian PSD (same spectrum), curve $\to-0.15625\,t^4$, sphere
  $m/t^4=-0.15344,-0.15554,-0.15607,-0.15621$, slopes 3.980, 3.995, 3.999. **Order 4: confirmed.**
- HT4, $1111$ (1.5), $K=7$: gradient 0, Hessian PSD (eigenvalues 0 x31, 1 x4, 7 x1), $k^*=7$. Sphere
  $m/t^6=-0.002341,-0.002321,-0.002316,-0.002315$ ($t=0.4\to0.05$), slopes 6.012, 6.003, 6.001, against
  $-1/432=-0.0023148$. By hand: $2(L-L_S)=y_1^2+y_2^2-3y_1y_2+y_1^2y_2^2$ with $y_i=b_i^3$, so
  $-\tfrac12y^2$ at $y_1=y_2=(t^2/6)^{3/2}$, and the symmetric split is a local minimum of the allocation. **Order 6: confirmed.**

**Scope remark on "counterexample to R2".** R2 in `theory_saddles.md` is literally about **one** new
component $r$. The pair-star has two residuals, so it refutes the multi-residual use of "order $=k^*$"
(Prop. 1, 3.3, 3.4, 5.1, CLAUDE.md Q3 line), not R2's literal statement. The two single-residual
wrong-order examples refute R2 itself together with 4.1's sentence "$m\ge3$ ... piggyback is not an escape
route". They do not refute the 4.3 table, which only says "non-strict"; that remains true. Fix: in section 0 and S8, say which statement each example refutes.

## 10. Dynamics claim (note section 6): correct as **derived**; independent check agrees

- Reduced model re-derived. Eliminating the $O(1)$-curvature rotations ($z=s\,y$) leaves
  $L_{\rm eff}=-\tfrac c2y^2$ with $c=s_1^2+s_2^2-s_q^2$ (units $s_q=1$). Then $\dot b_v=c\,b_v\prod_{u\ne v}b_u^2$, so $t_{\rm esc}\propto\sigma^{-(2\ell-2)}$.
  Unlike a route, the sign of $y$ does not matter ($L_{\rm eff}$ is even in $y$), so no sign-correction delay is expected. For several multi-core
  nodes (HT4) the reduced dynamics is a coupled polymer system, and the same exponent requires the polymers to align in sign; not checked.
- Check (own code, own RNG, GF from $\theta_S+\sigma\xi$, $\xi\sim N(0,I)$, LSODA rtol $10^{-10}$, escape
  $=L-L_S<-10^{-3}$):

| teacher | seed | $t_{\rm esc}$ at $\sigma=0.02,0.01,0.005$ | local exponents | state at escape |
|---|---|---|---|---|
| pair-star (0.8, 0.8) | 0 | 40024, 164041, 664943 | 2.035, 2.019 | $110000$: $-0.085$; rotations $000001$, $000010$: $-0.065$ each |
| pair-star (0.8, 0.8) | 1 | 21203, 82652, 326276 | 1.963, 1.981 | same |
| control (0.6, 0.6) | 0 | none by $t=10^7$ (all three $\sigma$) | - | all residual amplitudes $\approx0$ |
| control (0.6, 0.6) | 1 | $3.78\cdot10^6$, none, none | - | $110010$ route ($+0.002$) |
| control (0.6, 0.6) | 2 | $9.39\cdot10^5$, none, none | - | $110010$ route |

  Pair-star exponent 2: **confirmed** on two more seeds, and the escape direction is the non-residual $110000$ plus rotations, as claimed. The
  control's $\sigma^{-3}$ could not be resolved within the $10^7$ cap, but it is consistent with it. On the same init direction (seed 1, $\sigma=0.02$)
  the control is 178x slower than the pair-star. Seed 0's control never escapes, and pair-star seed 2 was not run. Status: **derived + checked (now 5 seeds in total)**. The comparison with the route law
  rests on the note's own two control seeds.

## 11. Note section 7 (proposed edits to `theory_saddles.md`) and other factual points

| item | verdict | fix |
|---|---|---|
| 7.5: "For $N\le4$ (all current experiments with two components) nothing changes ($k^*\le4$)" | **wrong** | ht4 has $N=4$ but $K=7$ (4 leaf cores + 3 internal), so $k^*\le7$. The note's own HT4 $1111$ example is $N=4$. Replace with: "For $K\le4$ (mps4, binary4) nothing changes. On ht4, correct-order two-component teachers are unaffected (Cor. 2a). In the wrong order, $D=\{0,1,2,3\}$ has order 6, not 7. Three-site $D$ have $k^*=6$ and no low split ($\{l_0,l_1\}$ has degree $3=k^*/2$), so order 6 $=k^*$. Two-site $D$ have $m=2$ (classical piggyback)." Any wrong-order runs on ht4 with $D$ = all sites should be re-read. |
| 7.1, 7.4, 7.8 | correct with fix | "order $=k^*$ proven" is right for the stated cases (with the route). Statements of "order $2\ell^*$" need Theorem 1'. 7.8: comparing $k^*$ with $2\ell^*$ is **derived** (a leading-order race), not proven. |
| 7.9 (CLAUDE.md line) "proven **iff** the generalized piggyback condition holds" | overstated | Necessity is proven only for pattern teachers with $s_r>0$, strict inequality and the index condition. Suggested line: "saddle order $=k^*$ **proven** when $\lambda_{\max}(\mathcal S)<s_q$ (always for $k^*\in\{3,4\}$ with a PD old block, and for two-component teachers in the correct order); otherwise order $2\ell^*<k^*$ (proven for pattern teachers with a spare bond index; first case $k^*=5$, MPS6 pair-star)". |
| Section 5, gauge sentence | wrong justification, right conclusion | See section 1 of this review. |
| S4 range $k^*\le4$ | wrong at $k^*=2$ | Use $k^*\in\{3,4\}$. |
| Section 0 summary of Theorem 2 | incomplete | Add "pattern teacher, $s_r>0$, index condition". |

## 12. Attempts to break Theorem 1: none succeeded

1. **Line by line** (sections 4-5 above). Every inequality re-derived, no gap. In particular the
   low/high split never lets an unbounded quadratic term escape. Low-low two-block terms are exactly
   $\mathcal S$. Every other gain term is either $\varepsilon\times$ (quadratic in $\kappa_{\mathcal L}$), or Young-split, or $O(\varepsilon^{k^*})$.
2. **Structural edge cases**, all consistent:
   - cores without sites ($T(B)=0$);
   - $\chi=1$ edges ($\kappa\equiv0$ through cores without new entries);
   - multi-site cores (sectors are per core, and patterns may differ on a subset of a core's sites);
   - unbalanced $\theta_S$ (gauge-equivalent);
   - gauge-orbit points;
   - curves mixing old and new entries;
   - new-entry contributions to singleton sectors and to $N$.
3. **Numerical stress tests** (sphere minima, own code):
   - MPS5 $10101$ with $s_r=2s_q$: wrong order, three isolated terminals, large cubic piggyback coupling, $\mathcal S=0$. Theorem 1 predicts order 5 for any $s_r$. Measured slopes 5.000, 5.000, 5.000, $m/t^5=-0.035777=-2\cdot5^{-5/2}$ (single route) to 6 digits.
   - MPS6 $110001$ at 0.7: generalized edge just below $s_q$. Predicted 6; measured slopes 6.000 (x3), $m/t^6=-0.003241=-0.7/216$.
   - Pair-star control at $\rho=0.85$: predicted 5; measured 5.000.
4. **Not tested** (light compute): general non-pattern $R$ beyond the note's three cases; $\lambda_{\max}(\mathcal S)$
   within a few percent of $s_q$, where the asymptotic regime starts at very small $t$; trees with $K>7$.
   The only real defect found is in the converse direction (section 6), not in Theorem 1.

## 13. Recommended status tags

| # | claim | note's tag | recommended |
|---|---|---|---|
| S1 | Lemma 1 | proven | **proven** |
| S2 | Lemma 2 (locality of cumulants) | proven | **proven** |
| S3 | Theorem 1 (lower bound $k^*$ under $\lambda_{\max}(\mathcal S)<s_q$; equality with the route) | proven | **proven** (state the route condition $\chi_e\ge2$ on one minimal Steiner tree in S3 too) |
| new | Theorem 1' (level version, section 6) | absent | add it, **proven** |
| S4 | Cor. 1 | proven | **proven for $k^*\in\{3,4\}$** (not $\le4$); $k^*=3$ includes equality; $k^*=4$ at equality open |
| S5 | Cor. 2a | proven | **proven** (with the route) |
| S6 | Theorem 2 | proven | **proven** (pattern teacher, $s_r>0$, index condition) |
| S7 | order $=\min(k^*,2\ell^*)$ | proven (pattern, index) | **proven once Theorem 1' is added** (strict inequality at level $\ell^*-1$, pattern teacher, index condition); general $R$: lower bound proven (Thm 1'), upper bound empirical |
| S8 | R2 fails from $k^*=5$ | proven + numerics | **proven** as an upper bound by Theorem 2 alone (enough to refute "order $=k^*$"); exact orders 4/4/6 **proven with Theorem 1'**; numerics reproduced independently. Fix the scope wording (section 9) |
| S9 | escape $\propto\sigma^{-(2\ell-2)}$ | derived, one teacher | **derived**, checked on one teacher (now 5 seeds) |
| S10 | 32 numerical instances | empirical | **empirical** (6 instances reproduced independently) |
| 7.5 | "$N\le4$: nothing changes" | - | **wrong** for ht4; fix as in section 11 |

