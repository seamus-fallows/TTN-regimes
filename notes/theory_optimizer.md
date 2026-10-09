# Theory: learning rate, edge of stability and SGD noise (Q8, H6)

Theory note for Q8 / H6 (`PLAN.md`), **v2** (revised after adversarial verification; the issues and
responses are listed in section 7). Status tags: **proven**, **derived** (under the stated assumptions),
**heuristic**, **conjecture**, **empirical**; numerical checks are tagged N# (first version) or R#
(revision) and listed in section 8. Scripts live outside the repo, in `.../scratchpad/theory2/optimizer/`
(N1–N15, helpers in `common.py`) and `.../scratchpad/theory2/optimizer_rev/` (R1–R11, helpers in `rc.py`).
They use only `ttn.tree`, `ttn.network`, `ttn.teachers` and `ttn.losses`, in float64. The verifier's
scripts are in `.../scratchpad/theory2/optimizer_verify/` (V#).

## 0. Results at a glance

| # | result | status |
|---|---|---|
| A1 | At the balanced odeco minimum the Hessian spectrum follows a "pattern rule" (2.2) and $\lambda_{\max}=Ks_1^{2-2/K}$ | proven; checked to $10^{-15}$ on 5 trees (N1, V1). Binary4 is a depth-3 DLN, whose spectrum is known (Ghosh et al. 2025, Lemma 1); new: general trees |
| A2 | Every other global minimum (any gauge, $\chi_e=R$) is at least as sharp. At imbalanced aligned minima $\lambda_{\max}$ is the largest pattern block, which can be a weaker, strongly imbalanced component | proven (diagonal), derived (gauge); checked (N1, R6) |
| A3 | GD changes the conserved quantities by exactly $\eta^2(g_v^\top g_v-g_pg_p^\top)$ per step; above $2/\lambda(\text{GF minimum})$ GD selects a flatter, more balanced minimum with sharpness just below $2/\eta$ | identity: TTN form of a known DLN identity, checked to $2\cdot10^{-16}$; selection known for DLNs; checked (N11, V2) |
| B1 | Sharpness along the saddle-to-saddle path: $O(\sigma^{K-2})$ on the first plateau, then $K\max_{r\,\text{learned}}s_r^{2-2/K}$; finite init adds $O(\sigma^4)$ | derived; checked (N2, V2b) |
| B2 | For $\eta_c<\eta<2/\lambda_2$ the instability stays in component 1's radial mode, which is the scalar depth-$K$ map; explicit period-2 amplitude and EoS loss | derived; checked (N4, N5, V2, V3) |
| C1 | GD escape times shift by $\frac\eta2\ln\frac{\lvert\nabla L(t_r)\rvert}{\lvert\nabla L(0)\rvert}+O(\eta^2)$ along the path; asymptotically $\eta[\frac{K-1}2\ln\frac1\sigma+c_r]$. No saddle is skipped | derived; checked (N3, V3); the $\sigma$-slope is asymptotic |
| D1 | Online SGD, first plateau: noise vanishes with the drift at the origin; plateau exponent unchanged; mean delay and spread of $t_1$ given by a 1D SDE | derived; checked with 160–184 seeds per condition (R3, R4) |
| D2 | SGD, exact mean-square threshold at the minimum: $\eta^*_{\rm ms}(B)=2/\lambda_{\max}(G_B)$, a #patterns-dimensional matrix (5.3). Typical (almost-sure) divergence threshold $\eta_{\rm as}\ge\eta^*_{\rm ms}$: $\approx1.9$–$2.0\,\eta^*_{\rm ms}$ for $N=4$, $B\le16$; $\approx8$–$21\,\eta^*_{\rm ms}$ for binary8 ($d=\chi=3$ or 2) at $B\le16$ | MS formula derived exactly (checked to $10^{-15}$, R1); $\eta_{\rm as}$ computed (R2, R8, V7) |
| D3 | SGD on a later degenerate ($k=3$) plateau: noise **lengthens** it. $T_{\rm SGD}/T_{\rm GF}=-\ln(1-x)/x$, $x=c_\gamma(\eta/B)T_{\rm GF}$, and the network is **trapped** for $x\ge1$ (stochastic collapse of the route) | mechanism known (stochastic collapse); law heuristic; checked on binary4 at 4 noise levels (R5) |
| D4 | Near the SGD stability limit, heavy-tailed kicks erase learned components (forget-and-relearn) or diverge; this breaks the saddle sequence rather than compressing it | empirical (V12, V13, R5, R10) |

**Withdrawn from v1:** "noise-driven, $b_0$-independent escapes up to $\sim10^3\times$ faster than GF"
(N9, N13 and N15 events were forget-and-relearn events or blow-ups, not escapes); "$\eta_{\rm ms}$ is the
divergence threshold" and "$\eta\propto3^{-N}$ in practice"; "the SGD mean escape time is unbiased";
"$O(\sigma^2)$ excess sharpness".

## 1. Setting and notation

- $N$ sites, physical dimension $d$, tree with $K$ cores $G_v$, bond dimension $\chi_e$ on edge $e$.
  $W(\theta)$ is the contracted TTN and $\theta$ the vector of all $n_\theta$ core entries.
- Whitened product-Gaussian inputs $x^{(i)}\sim N(0,I_d)$, features $\Phi(x)=\otimes_ix^{(i)}$,
  $f_\theta(x)=\langle W(\theta),\Phi(x)\rangle$, population loss $L=\tfrac12\lVert W-W^*\rVert_F^2$.
- Odeco teacher $W^*=\sum_{r=1}^Rs_rU_r$, $U_r=\otimes_iu_r^{(i)}$, $s_1>s_2>\dots>s_R>0$, orthonormal
  $\{u^{(i)}_r\}_r$ at every site. $Z_r(x):=\langle U_r,\Phi(x)\rangle=\prod_iz_{r,i}$ with
  $z_{r,i}=u_r^{(i)}\cdot x^{(i)}$ iid $N(0,1)$.
- Jacobian $J=\partial W/\partial\theta$, residual $D=W-W^*$, Hessian $H=J^\top J+\langle D,\partial^2W\rangle$.
- **Aligned point**: core $v$ is $\sum_ra_{v,r}(\text{teacher vectors on its physical legs})\otimes(e_r\text{ on its bond legs})$
  (`ttn.network.aligned_init`); it represents $\sum_r(\prod_va_{v,r})U_r$. The **balanced minimum** has
  $a_{v,r}=q_r:=s_r^{1/K}$.
- Conserved quantities $Q_v=G_v^\top G_v-G_{p,(v)}G_{p,(v)}^\top$ (`notes/model.md` §5); at an aligned point
  $(Q_v)_{rr}=a_{v,r}^2-a_{p,r}^2$.
- GD: $\theta_{n+1}=\theta_n-\eta\nabla L$, time $t=\eta n$. Online SGD: a fresh batch of $B$ samples per step.
- Step-size thresholds used throughout: $\eta_c:=2/(Ks_1^{2-2/K})$ (GD at the balanced minimum);
  $\eta_{\rm ms}^{\rm rad}(B):=\eta_cB/(B+3^N-1)$ (radial mean-square bound of v1);
  $\eta^*_{\rm ms}(B)$ (exact mean-square threshold, 5.3); $\eta_{\rm as}(B)$ (almost-sure threshold, 5.3).
- Overlap $a_r:=\langle W,U_r\rangle/s_r$ (`component_overlaps`); event $r$ = first time $a_r>1/2$.

## 2. Sharpness at global minima

### 2.1 Hessian = tangent kernel (proven)

At a global minimum $D=0$, so $H=J^\top J$; its nonzero eigenvalues are those of the population NTK
$JJ^\top$ on tensor space, and $\lambda_{\max}(H)=\lVert J\rVert_{op}^2$.

### 2.2 Full spectrum at aligned minima: the pattern rule (proven; checked N1, V1)

Use an orthonormal basis of each core's entries: teacher vectors $u_\ell^{(i)}$ ($\ell\le R$) and a
complement basis on each physical leg, one-hot vectors on bond legs. An elementary perturbation $e$ of
core $w$ produces $\delta W=\mathrm{gain}(e)\,T(e)$, where the **pattern** $T(e)$ is a unit rank-one tensor
with: the chosen vectors at $w$'s own sites; $u_{\ell_c}$ at every site in the subtree of child $c$ (bond
label $\ell_c$); $u_{\ell_p}$ at every site outside $w$'s subtree (parent label $\ell_p$). The gain is
$\prod_{u\neq w}a_{u,\ell(u)}$, with $\ell(u)$ the label of the branch of $w$ containing $u$; labels $>R$ give gain 0.
Distinct patterns are orthonormal, so $J^\top J$ is block diagonal over patterns with rank-one blocks:

$$\mathrm{spec}(H)=\Big\{\lambda_P:=\textstyle\sum_{w\in\mathcal C(P)}\mathrm{gain}_w(P)^2\ :\ P\text{ a pattern}\Big\}\cup\{0,\dots\},$$

where $\mathcal C(P)$ is the set of cores that can produce $P$. Each block has one unit eigenvector
$e_P\propto J^\top T(P)$, and $T(P)$ is an eigenvector of $JJ^\top$ (used in 5.3). Special cases:

- **Radial mode of component $r$** (pattern $U_r$, produced by all $K$ cores): $\lambda=s_r^2\sum_va_{v,r}^{-2}$,
  $=Ks_r^{2-2/K}$ at balance.
- **Single-core modes** (a non-teacher vector at one site of a multi-site core): $s_r^{2-2/K}$ at balance.
- **Split modes** ($r$ below an edge, $r'$ above): in binary4 the root, $A$ and $B$ all produce $(1,1,2,2)$,
  $\lambda=q_1^4+q_1^2q_2^2+q_2^4$; in an MPS the leaf edge gives $q_1^{2(K-2)}(q_1^2+q_2^2)$.

**Top eigenvalue (proven).** Each block has $\lambda\le\lvert\mathcal C(P)\rvert q_1^{2(K-1)}\le Ks_1^{2-2/K}$, with
equality only for $P=U_1$, so $\lambda_{\max}=Ks_1^{2-2/K}$ with a nondegenerate eigenvector when $s_2<s_1$.

**Known vs new.** Binary4 is exactly a depth-3 deep linear network (`theory_saddles.md` §4.2), and its
pattern spectrum is the known balanced-DLN spectrum (Ghosh et al., ICLR 2025, Lemma 1: $L\sigma_i^{2-2/L}$,
$\sigma_i^{2-2/L}$ and $\sum_{\ell=0}^{L-1}(\sigma_i^{1-1/L-\ell/L}\sigma_j^{\ell/L})^2$; with $L=3$ the last is
$q_i^4+q_i^2q_j^2+q_j^4$). New here: the rule for general trees (MPS, HT, deeper binary), where the set
of producers of a pattern depends on the topology.

Numbers (N1, reproduced by V1; $q_r=s_r^{1/K}$):

| tree | $d,\chi$ | $K$ | $s$ | $\lambda_1$ | $\lambda_2$ (what it is) | $\lambda_2/\lambda_1$ |
|---|---|---|---|---|---|---|
| binary4 | 3,3 | 3 | (1,.6,.3) | 3.0000 | 2.2174 (split $(1,1,2,2)$, 3 producers) | 0.739 |
| mps4 | 3,3 | 4 | (1,.6,.3) | 4.0000 | 1.8590 (radial-2, $4s_2^{3/2}$) | 0.465 |
| binary8 | 2,2 | 7 | (1,.5) | 7.0000 | 2.1333 (radial-2) | 0.305 |
| mps6 | 2,2 | 6 | (1,.5) | 6.0000 | 1.8899 | 0.315 |
| ht4 | 2,2 | 7 | (1,.7) | 7.0000 | 3.7980 | 0.543 |
| binary4 | 3,3 | 3 | (3,2,1) | 12.980 | 10.15 | 0.782 |
| binary8 | 3,3 | 7 | (3,2,1) | 46.03 | 22.97 | 0.499 |
| mps8 | 3,3 | 8 | (3,2,1) | 54.71 | 26.91 | 0.492 |

In the 5 trees of N1, autograd and the pattern rule agree on every eigenvalue to $\le6.3\cdot10^{-15}$ (R1
re-checks binary4, mps4, binary8 to $3\cdot10^{-15}$); the last three rows (the experiment defaults
$d=\chi=3$, $s=(3,2,1)$) are from the pattern rule (R7).

### 2.3 Imbalance and the minimum manifold (proven / derived; checked N1, R6)

- **Diagonal imbalance (proven).** On aligned minima $\prod_va_{v,r}=s_r$, and the pattern rule holds with the
  actual amplitudes $a_{v,r}$. The radial eigenvalue of component $r$ is $s_r^2\sum_va_{v,r}^{-2}\ge Ks_r^{2-2/K}$
  (AM–GM; equality iff $a_{v,r}=q_r$). Gradient flow fixes $a_{v,r}^2-a_{p,r}^2=(Q_v)_{rr}(0)$, so the minimum
  it reaches and that minimum's sharpness are functions of the initial conserved quantities.
- **Which mode is the top one.** $\lambda_{\max}$ is the largest pattern block, $\max_P\lambda_P$. At balance this
  is radial-1. With imbalance, a weaker component with strong imbalance can dominate (verifier's example,
  confirmed by autograd in R6): binary4, $s=(1,.6)$, component 1 balanced, component 2 with leaves $a^2=0.05$
  and root $a=12$ gives $\lambda_{\max}=14.40$ (radial-2) and $\lambda_2=8.25$ (a split mode), against 3.00 from
  the radial-1 formula. Example where radial-1 dominates (N1): component 1 with $(Q)_{11}=-2.24$ on both leaf
  edges gives $\lambda=2/x+1/(x+2.24)=3.72$ with $x^2(x+2.24)=1$.
- **Any gauge (derived; $\chi_e=R$, where the minimum set is the gauge orbit of the balanced minimum).**
  With $g_v=\partial\langle W,U_1\rangle/\partial G_v$, $\lambda_{\max}\ge\sum_v\lVert g_v\rVert^2$; every edge contributes
  $\lVert A_v^{-1}e_1\rVert\lVert A_v^\top e_1\rVert\ge1$, so $\prod_v\lVert g_v\rVert\ge q_1^{K(K-1)}$ and, by AM–GM,
  $\lambda_{\max}\ge Ks_1^{2-2/K}$. Orthogonal gauges are isometries, so the spectrum depends only on the
  conserved quantities. Random $GL$ gauges: $\min\lambda_{\max}/(Ks_1^{2-2/K})=1.43$ (binary4) and $1.93$ (mps4)
  over 30 gauges (N1); $\ge2.34$ and $\ge1.67$ over 20 (V1).
- **Consequence.** For $\eta>\eta_c$ no global minimum is linearly stable for GD (derived for $\chi_e=R$;
  conjecture for $\chi_e>R$).
- **Literature.** Balanced minima are the flattest DLN minima (Mulayoff & Michaeli, ICML 2020); exact sharpness at
  arbitrary DLN minima: Kamber & Parhi (arXiv 2509.25783, 2025).
- **Anisotropic inputs (derived, not checked).** For $\Sigma_i$ aligned with the teacher ($u_r^{(i)}$ an
  eigenvector with eigenvalue $\lambda_{r,i}$, $\Lambda_r=\prod_i\lambda_{r,i}$), each pattern eigenvalue is weighted
  by the product of the metric eigenvalues of its site vectors; radial-$r$ is $K\Lambda_rs_r^{2-2/K}$ at
  balance, so the unstable component need not be the one with the largest $s_r$.

### 2.4 GD breaks the conservation laws and balances (identity proven; selection known; checked N11, V2)

For one GD step the linear terms cancel by the gauge identity $G_v^\top g_v=g_{p,(v)}G_{p,(v)}^\top$, so

$$Q_v^{n+1}-Q_v^{n}=\eta^2\big(g_v^\top g_v-g_{p,(v)}g_{p,(v)}^\top\big),\qquad g=\nabla L\ (\text{matricized as for }Q),$$

checked to $1.8\cdot10^{-16}$ (N11) and $2.4\cdot10^{-16}$ (V2). This is the TTN (per-bond) form of the DLN
identity behind approximate balancedness under GD (Du, Hu & Lee, NeurIPS 2018); for one aligned component it
reads $Q^{n+1}=Q^n(1-\eta^2\rho^2\Pi^2/(a_v^2a_p^2))$ (residual $\rho=s-\Pi$, $\Pi=\prod_ua_u$), the scalar
contraction law of Hofmann (2026). It is negligible on plateaus ($\rho\Pi\approx0$) and large during
oscillations.

**Selection (known; checked N11, reproduced by V2).** From an imbalanced start whose GF minimum is too sharp
($2/\eta<\lambda_{GF}$), GD oscillates, the imbalance shrinks and it converges to a flatter minimum with
sharpness just below $2/\eta$ (Kreisler et al., ICML 2023; Ghosh et al. 2025, Prop. 2; Wang et al., ICLR 2022).
N11 (binary4, $\lambda_{GF}=3.72$):

| $\eta$ | 0.30 | 0.53 | 0.56 | 0.60 | 0.64 | 0.68 |
|---|---|---|---|---|---|---|
| final sharpness | 3.692 | 3.668 | 3.543 | 3.311 | 3.103 | oscillates |
| $2/\eta$ | 6.67 | 3.77 | 3.571 | 3.333 | 3.125 | 2.94 $<3$ |
| final $(Q_{\text{leaf}})_{11}$ (start $-2.24$) | $-2.17$ | $-2.12$ | $-1.82$ | $-1.25$ | $-0.67$ | $-0.02$ |

## 3. Sharpness along the saddle-to-saddle trajectory

### 3.1 Plateau sharpness (derived; checked N2, V2b)

- **First plateau.** $W$ is a degree-$K$ polynomial in $\theta$, so $H=O(s_1\lvert\theta\rvert^{K-2})$, indefinite and
  $O(\sigma^{K-2})$. Binary4 at $\sigma=0.05$: $\lambda_{\max}\approx0.05$–$0.06$, $\lambda_{\min}\approx-0.05$ (N2, V2b).
- **During an event** (one aligned balanced component, per-core amplitude $a$): the radial eigenvalue is
  $\lambda(a)=(2K-1)a^{2K-2}-(K-1)s\,a^{K-2}$; it rises monotonically once $a^K>\frac{(K-2)s}{2(2K-1)}$ and reaches
  $Ks^{2-2/K}$ at $a=s^{1/K}$ without overshoot. The $K-1$ transverse eigenvalues $(s-a^K)a^{K-2}$ peak at
  $\frac{sK}{2K-2}\big(\frac{s(K-2)}{2K-2}\big)^{(K-2)/K}<Ks^{2-2/K}$.
- **Later plateaus.** At a balanced partial fit that has learned the set $S$, the radial mode of the largest
  learned component is an exact eigenvector with eigenvalue $K\max_{r\in S}s_r^{2-2/K}$, unchanged by the
  residual ($\langle D,\partial^2W[\text{radial},\cdot]\rangle=0$ for $K\ge3$); residual-induced eigenvalues are
  $O(s_{\rm new}q_1^{K-2})$.
- **Finite init: the excess is $O(\sigma^4)$, not $O(\sigma^2)$ (derived; corrected after V2b).** Balance minimizes
  the radial eigenvalue, so the excess is quadratic in the imbalance. Write $a_v^2=q^2e^{u_v}$ with
  $\sum_vu_v=0$ (product fixed). Then
  $$\lambda_{\rm rad}=s^2\sum_va_v^{-2}=Ks^{2-2/K}\Big(1+\frac1{2K}\sum_vu_v^2+O(u^3)\Big),\qquad u_v-u_p=\frac{(Q_v)_{11}}{q^2}+O(Q^2),$$
  and fan-in init gives $Q=O(\sigma^2)$, so the excess is $O(\sigma^4)$. V2b (binary4, GF to convergence):
  excess$/\sigma^4=0.68,0.90,1.25$ and $0.44,0.43,0.41$ at $\sigma=0.15,0.3,0.6$ (two seeds), while excess$/\sigma^2$
  grows from 0.010–0.015 to 0.15–0.45. N2 (mps4, $\sigma=0.25$): excess $0.0015$–$0.0032$, i.e. $0.4$–$0.8\,\sigma^4$.
- N2 trajectory (binary4, $s=(1,.6,.3)$, $\sigma=0.05$): $\lambda_{\max}$ goes from $0.06$ to $3.0000$ at the first
  event and stays at $3.0000$ through both later plateaus (reproduced by V2b).

### 3.2 Where GD first becomes unstable (derived; checked N5)

With a fixed $\eta$ a plateau is linearly unstable iff it has learned a component with
$s_r>s_c(\eta):=\big(2/(\eta K)\big)^{K/(2K-2)}$. So the instability first appears at the learning event of the
first component with $s_r>s_c(\eta)$: at the first event under greedy order (only if $\eta>\eta_c$), at
component 1's event under non-greedy order. Earlier plateaus are stable for any practical $\eta$ (sharpness
$O(\sigma^{K-2})$). The next mode to go unstable is $\lambda_2$ (table in 2.2).

### 3.3 Edge-of-stability dynamics = the scalar depth-$K$ map (derived; checked N4, N5, V2, V3)

For $\eta_c<\eta<2/\lambda_2$ only component 1's radial mode is unstable. On it, GD is exactly (aligned) or
approximately (generic) the balanced scalar map

$$x\mapsto x+h\,x^{K-1}(1-x^K),\qquad x=a/s_1^{1/K},\quad h=\eta\,s_1^{2-2/K},$$

the balanced-DLN singular-value map; Ghosh et al. (2025) report its period-doubling route to chaos. The
thresholds in $hK=2\eta/\eta_c$ (N4, V3; the accumulation point from the Feigenbaum ratio, corrected after V3):

| $K$ | period 2 | period 4 | period 8 | period 16 | chaos (accumulation) | divergence (balanced map) |
|---|---|---|---|---|---|---|
| 3 | 2.000 | 2.489 | 2.601 | 2.625 | 2.632 | 4.31 |
| 4 | 2.000 | 2.494 | 2.609 | 2.633 | 2.639 | 4.82 |
| 7 | 2.000 | 2.500 | 2.616 | 2.641 | 2.647 | $>5$ |

**Period-2 amplitude (derived; a small addition to Ghosh et al. 2025, whose Theorem 1 gives the orbit as two
roots of a polynomial).** $F''(1)=-3hK(K-1)$ and $F'''(1)=-hK(K-1)(7K-11)$, so the orbit is $x=1\pm\delta$ with

$$\delta^2\simeq\frac{3\,(hK-2)}{4(K-1)(5K-4)},\qquad
L_{\rm EoS}\simeq\tfrac12s_1^2K^2\delta^2=\frac{3K^2s_1^2}{4(K-1)(5K-4)}\Big(\frac{\eta}{\eta_c}-1\Big).$$

At $hK=2.1$: $\delta=0.0575,0.039,0.0199$ for $K=3,4,7$ (map), against $0.0584,0.0395,0.0201$ (formula).

**Full TTN (N5; reproduced by V2):** binary4, $s=(1,.6,.3)$, generic init $\sigma=0.05$, $\eta_c=0.667$, $2/\lambda_2=0.902$:

| $\eta$ | 0.6 | 0.7 | 0.8 | 0.85 | 0.95 | 1.2, 1.5 |
|---|---|---|---|---|---|---|
| late loss (mean) | $10^{-16}$ | $1.45\cdot10^{-2}$ | $5.0\cdot10^{-2}$ | $6.1\cdot10^{-2}$ | $8.4\cdot10^{-2}$ | diverged |
| oscillating overlaps | none | $a_1$ (range .34) | $a_1$ (.62) | $a_1$ (.80) | $a_1$ (1.23), $a_2$ (.43) | — |
| events $t_1,t_2,t_3$ | 75.5, 73.5, 282.4 | 75.9, 73.9, 282.8 | 76.3, 74.1, 283.2 | 76.6, 74.3, 283.4 | 77.0, 74.6, 283.8 | — |

At $\eta=0.7$ the late loss ($1.45\cdot10^{-2}$; V2: $1.454\cdot10^{-2}$) equals the scalar map's ($1.452\cdot10^{-2}$);
the normal form gives $1.53\cdot10^{-2}$. Components 2 and 3 converge exactly while component 1 oscillates, and
**later events still occur and still lower the loss**: across event 3 the mean loss drops from 0.0595 to
0.0146 at $\eta=0.7$ and from 0.094 to 0.049 at $\eta=0.8$ (V2). The full network diverges between $1.5$ and
$1.65\,\eta_c$ (V2; N5 bracket $1.43$–$1.8$), before the balanced map ($2.16\,\eta_c$), so other modes drive it.
(This seed's component 2 crosses $1/2$ first and is then displaced; GF does the same, at $t=71.8$ and $73.1$.)

## 4. Discrete steps on plateaus

### 4.1 Leading-order correction to escape times (derived; checked N3, V3)

To $O(\eta^2)$ GD follows the modified flow $\dot\theta=-\nabla\tilde L$, $\tilde L=L+\frac\eta4\lVert\nabla L\rVert^2$
(Barrett & Dherin, ICLR 2021), i.e. $\dot\theta=-(I+\frac\eta2H)\nabla L$. The component of the correction along
the path reparametrizes time: $dt_{\rm GD}=dt_{\rm GF}\,(1-\frac\eta2\nabla L^\top H\nabla L/\lVert\nabla L\rVert^2)$, and since
$\frac{d}{dt}\ln\lVert\nabla L\rVert=-\nabla L^\top H\nabla L/\lVert\nabla L\rVert^2$ along GF,

$$t_{\rm GD}(\text{event})-t_{\rm GF}(\text{event})=\frac\eta2\ln\frac{\lvert\nabla L(t_{\rm event})\rvert}{\lvert\nabla L(0)\rvert}+(\text{transverse part})+O(\eta^2).$$

(This path form was derived independently by the verifier.) The transverse part of the correction moves the
iterate to a neighbouring GF path and is negligible when GF has no long alignment phase.

- **Aligned init (derived; exact path):** one component evolves as $\dot a=f(a)-\frac\eta2f'(a)f(a)$,
  $f(a)=a^{K-1}(s-a^K)$, giving $\Delta t=\frac\eta2\ln\frac{f(a_1)}{f(a_0)}$. N3 (aligned binary4, 6 cases) and V3 (9 cases):
  within 1.3%. Relative to the plateau $t_{GF}\approx a_0^{-(K-2)}/((K-2)s)$ the shift is $O(\eta s a_0^{K-2}\ln\frac1{a_0})\to0$:
  **plateau exponents and prefactors are those of gradient flow.**
- **Generic init (asymptotic in $\sigma\to0$; heuristic):** $\lvert\nabla L(0)\rvert\propto\sigma^{K-1}$, so
  $\Delta t_1/\eta=\frac{K-1}2\ln\frac1\sigma+c$, a slope of $\frac{K-1}2\ln2$ per halving of $\sigma$ (binary4 0.69, mps4 1.04,
  binary8 2.08). Checks: binary4 0.70 and 0.69 (N3, V3); mps4 seed 1: 1.05, 1.04 (V3). **Pre-asymptotic seeds
  exist:** mps4 seed 0 gives $\Delta t/\eta=4.40,5.00,5.47$ at $\sigma=0.2,0.1,0.05$ (slopes 0.61, 0.47), because its
  GF trajectory first spends a long alignment phase in which $\lvert\nabla L\rvert$ drops $5\times$; the path formula
  then misses by $1.5$–$2\,\eta$ (V3). The path formula matches within $0.15\,\eta$ when there is no long
  alignment phase (V3). Average slopes over seeds.
- **Strict-saddle ($k=2$) escapes (derived):** growth per step $1+\eta\lvert\lambda_-\rvert$ instead of
  $e^{\eta\lvert\lambda_-\rvert}$, so $t_{GD}/t_{GF}=\eta\lvert\lambda_-\rvert/\ln(1+\eta\lvert\lambda_-\rvert)\in[1,1.82]$ for
  $\eta\lvert\lambda_-\rvert<2$ (e.g. $\eta<\eta_c$ at the TTN strict saddles, where
  $\lvert\lambda_-\rvert=s_rs_q^{(K-2)/K}\le\lambda_{\max}$; `theory_saddles.md` R3).

### 4.2 Can large steps skip or reorder saddles?

- **Aligned init (proven, with a separation condition).** GD keeps the diagonal structure, so the components
  evolve as independent scalar maps: no component's instability affects another's learning, and no saddle is
  skipped. Event $r$ moves by $\Delta t_r=\frac\eta2\ln\frac{f_r(a_1)}{f_r(a_0)}+O(\eta^2)$, so the GF order of events
  $r,r'$ is kept whenever $\lvert t_r^{GF}-t_{r'}^{GF}\rvert>\lvert\Delta t_r-\Delta t_{r'}\rvert+O(\eta^2)$; with
  $\Delta t_r=O(\eta\ln\frac1\sigma)$ this fails only for (nearly) simultaneous GF events.
- **Generic init (empirical).** On the first plateau the GD map is near-identity (curvature $O(\sigma^{K-2})$).
  N5b (binary4, 12 seeds, $\eta\in\{0.03,0.1,0.3\}$ vs GF): same event sequence in 12/12, $\Delta t_r/\eta\approx3$–$3.5$
  (not re-checked by the verifier). N5: with $\eta$ in the EoS window, events 2 and 3 stay within 2.2 time units of their
  $\eta=0.3$ times.
- **Corollary for schedules (derived).** The stability limit on a plateau is set by the sharpness learned so far,
  so a step larger than $\eta_c$ is stable until the first relevant event.

## 5. SGD noise

### 5.1 SDE and noise covariance (known framework; TTN-specific moments derived)

Online SGD with batch $B$: $\theta_{n+1}=\theta_n-\eta\hat g_n$, $\mathbb E\hat g=\nabla L$. The per-sample gradient is
$g_x=r(x)J^\top\Phi(x)$ with $r(x)=\langle D,\Phi(x)\rangle$ (noiseless realizable labels), so
$\mathrm{Cov}(\hat g)=C(\theta)/B$ with

$$C(\theta)=J^\top\big(\mathcal M[D]-DD^\top\big)J,\qquad\mathcal M[D]:=\mathbb E\big[\langle D,\Phi\rangle^2\Phi\Phi^\top\big],$$

a sum over per-site Wick pairings ($\mathbb E[x_ax_bx_cx_e]=\delta_{ab}\delta_{ce}+\delta_{ac}\delta_{be}+\delta_{ae}\delta_{bc}$ at every
site). To weak order 1, $d\theta=-\nabla L\,dt+\sqrt{\eta/B}\,C^{1/2}dW_t$ with $t=\eta n$ (Li, Tai & E 2017/2019).
For odeco teachers: $\mathbb E[r^2Z_r^2]=s_r^23^N+\sum_{r'\ne r}s_{r'}^2$ at the origin, and the feature kurtosis
along a component is $\mathbb EZ_r^4=3^N$ (Gaussian sites) or $\prod_i\mathbb E[(u_r^{(i)}\cdot\phi_i)^4]$ for other
orthonormal features ($=1$ for $\pm1$ features in the teacher's basis). **Population dynamics depend only on
second moments (`notes/model.md`); SGD noise and SGD stability depend on these fourth moments.**

### 5.2 First plateau: noise does not change the plateau law (derived; checked R3, R4)

At small $\theta$, $J=O(\lvert\theta\rvert^{K-1})$ and $D\approx-W^*$, so $C=O(s_1^23^N\lvert\theta\rvert^{2K-2})$ while the drift is
$O(s_1\lvert\theta\rvert^{K-1})$: **the noise vanishes at the origin at the same order as the drift** (a Type-II saddle,
Ziyin et al. 2023). Unlike the spherical single-index setting (Ben Arous, Gheissari & Jagannath, JMLR 2021),
where the noise stays $O(1)$ near the equator, the noise amplitude in the plateau clock $d\tau=\lvert\theta\rvert^{K-2}dt$ is
$O(\sqrt{\eta3^N/B}\,\lvert\theta\rvert^{(K-2)/2})\to0$, so noise cannot drive the first escape.

**1D SDE (derived).** One aligned balanced component (per-core amplitude $a$, all $K$ cores get the same
per-sample update, so balance is kept on the aligned manifold):
$da=s_1a^{K-1}dt+\sqrt{\eta/B}\,\sigma_na^{K-1}dW$, $\sigma_n^2=s_1^2(3^N-1)+\sum_{r\ge2}s_r^2=:s_1^2\kappa$. With
$y=a^{-(K-2)}$ (a drifted Brownian motion plus an Itô term $\propto a^{K-2}$) the hitting time is inverse Gaussian
to leading order:

$$\mathbb E\,t_1^{\rm SGD}-t_1^{\rm GF}\simeq\frac\eta2(K-1)\Big(1+\frac\kappa B\Big)\ln\frac{a_{1/2}}{a_0}>0,\qquad
\mathrm{sd}(t_1^{\rm SGD})\simeq\sqrt{\kappa\,\frac\eta B\,t_1^{\rm GF}},$$

equivalently $\mathrm{sd}/t_1^{\rm GF}\simeq\sqrt{(K-2)\kappa s_1(\eta/B)a_0^{K-2}}$, with $a_{1/2}=(s_1/2)^{1/K}$ and $a_0$ read
off from $t_1^{\rm GF}\approx a_0^{-(K-2)}/((K-2)s_1)$. The "1" is the GD correction of 4.1. **The mean delay is positive**
(v1 called the mean "unbiased"; that contradicted this formula and is withdrawn). Both corrections vanish
relative to $t_1^{\rm GF}\propto a_0^{-(K-2)}$ as $\sigma\to0$: **the plateau exponent $K-2$ is unchanged.**

**Checks** (binary4, $s=(1,.6,.3)$, $\kappa=80.45$; aligned init $a_0=0.0167$, GF $t_1=59.03$; generic init
$\sigma=0.05$ seed 0, GF $t_1=59.99$; formula values in brackets):

| condition | source | $n$ | mean shift | sd | sd / formula |
|---|---|---|---|---|---|
| aligned-manifold projection, $B=16$, $\eta=0.05$ | R3 | 2000 | $+1.03\pm0.08$ [1.16] | 3.60 [3.84] | 0.94 |
| full network, aligned, $B=16$, $\eta=0.05$ | R4 + V10 | 160 + 24 | $+0.91\pm0.30$ [1.16] | $4.02\pm0.21$ | 1.04 |
| full network, aligned, $B=4$, $\eta=0.05$ | R4 + V10 | 70 + 24 | $+3.70\pm0.73$ [4.06] | $7.06\pm0.68$ | 0.92 |
| full network, generic, $B=16$, $\eta=0.05$ | R4 + V10 | 160 + 24 | $+0.39\pm0.35$ [1.16] | $4.68\pm0.24$ | 1.21 |
| full network, generic, $B=64$, $\eta=0.05$ | V10 | 20 | $+1.06\pm0.39$ [0.43] | 1.76 | 0.91 |

So for the aligned dynamics the 1D formula gives the mean delay and the spread to within errors (10–15%).
For generic init the spread is about $1.2\times$ the formula and the mean delay is smaller; the formula ignores
the noise off the aligned manifold and the random alignment phase. **Rebuttal of the verifier's "shift $1.2$–$2.4\times$,
sd $0.8\times$":** with 24 seeds the sample sd scatters widely (my seven disjoint 24-seed blocks give sd
$4.2$–$5.6$ for generic and $3.0$–$4.8$ for aligned); the verifier's 24-seed sets (sd 2.95, 2.98) lie at the
low end, and their mean shifts at the high end. Pooled over all seeds the aligned case agrees with the
formula. Use $\ge100$ seeds (or interquartile ranges) for these statistics.

This agrees with Corlouer et al. (2026; aligned DLN SDE) that SGD keeps the first saddle-to-saddle step; in our
model their outward Itô drift is subleading near the origin by a factor $a^{K-2}$.

### 5.3 SGD stability at the minimum (exact mean-square formula derived; almost-sure threshold computed)

At the realizable minimum every per-sample residual vanishes, so the minimum is a fixed point of SGD and the
linearization is the random product $\delta_{n+1}=(I-\eta H_n)\delta_n$, $H_n=\frac1B\sum_bg_bg_b^\top$, $g_b=J^\top\Phi(x_b)$.

**Pattern-basis reduction (derived exactly from 2.2).** In the eigenbasis $\{e_P\}$ of $H$ (one unit vector per
pattern $P$ with $\lambda_P>0$), $g\cdot e_P=\Phi^\top Je_P=\sqrt{\lambda_P}\,Z_P(x)$ with
$Z_P(x)=\prod_i\big(t_{P,i}\cdot x^{(i)}\big)$, where $t_{P,i}$ is the site-$i$ basis vector of pattern $P$ (a teacher
vector or a complement vector). Null-space components never move. So linearized SGD is exactly

$$\delta_P\leftarrow\delta_P-\frac\eta B\sum_b\sqrt{\lambda_P}Z_P(x_b)\sum_Q\sqrt{\lambda_Q}Z_Q(x_b)\,\delta_Q,$$

a process in #patterns dimensions (45 patterns for binary4 with $d=\chi=3$, against $n_\theta=63$ parameters;
117 for binary8, against 171); for Gaussian sites the $t_{P,i}\cdot x^{(i)}$ are iid $N(0,1)$.

**Exact mean-square threshold (derived; checked R1).** The second-moment operator
$\mathcal S(C)=\mathbb E[(I-\eta H_n)C(I-\eta H_n)]$ maps matrices that are diagonal in the pattern basis to diagonal
ones and off-diagonal ones to off-diagonal ones, because $\mathbb E[Z_R^2Z_PZ_Q]=0$ for $P\ne Q$ (at a site where
$t_{P,i}\ne t_{Q,i}$, all three vectors come from one orthonormal basis, and $\mathbb E[(r\cdot x)^2(p\cdot x)(q\cdot x)]=2(r\cdot p)(r\cdot q)=0$
because $r$ cannot equal both). $\mathcal S$ preserves the PSD cone, so its Perron eigenvector is PSD and has
a nonzero diagonal, so $\rho(\mathcal S)$ is the spectral radius of the diagonal block
$M=I-2\eta\Lambda+\eta^2[(1-\tfrac1B)\Lambda^2+\tfrac1B\Lambda\mathcal K\Lambda]$, with $\Lambda=\mathrm{diag}(\lambda_P)$ and the
**site-agreement kernel** $\mathcal K_{PQ}=\mathbb E[Z_P^2Z_Q^2]=\prod_i m(t_{P,i},t_{Q,i})$, $m=3$ if the two site vectors
coincide and $m=1$ if they are orthogonal (Gaussian). Therefore

$$\boxed{\eta^*_{\rm ms}(B)=\frac{2}{\lambda_{\max}(G_B)},\qquad G_B=\Big(1-\frac1B\Big)\Lambda+\frac1B\Lambda^{1/2}\mathcal K\Lambda^{1/2}}$$

(mean-square stable iff $\eta<\eta^*_{\rm ms}$). Restricting $G_B$ to the radial-1 entry gives v1's bound
$\eta^{\rm rad}_{\rm ms}=\eta_cB/(B+3^N-1)$ (the 1D criterion of Wu, Ma & E, NeurIPS 2018), so
$\eta^*_{\rm ms}\le\eta^{\rm rad}_{\rm ms}$, with equality as $B\to\infty$. The exact criterion for general models is
Mulayoff & Michaeli (COLT 2024); the new part is the reduction to a symmetric #patterns-dimensional matrix built
from the pattern rule. The radial mode couples strongly to patterns that differ from $U_1$ at few sites
($\mathcal K=3^{N-1}$ for one differing site), which is why the exact threshold is lower. **Check (R1):** the formula
equals my dense $n_\theta^2$-dimensional operator to $\le1.4\cdot10^{-15}$ (4 small cases) and the verifier's dense
operator (V7b) to 4 digits.

**Typical (almost-sure) divergence threshold (known criterion; values computed).** SGD can converge to the
minimum iff the top Lyapunov exponent $\gamma$ of the random product is negative (Chemnitz & Engel, JMLR 2025,
Thm B; also the "probabilistic stability" of Ziyin et al. 2023). By Jensen $\gamma\le\frac12\ln\rho(\mathcal S)$ per
step, so $\eta_{\rm as}\ge\eta^*_{\rm ms}$ (proven). Between the two thresholds SGD converges almost surely but the
second moment diverges. I compute $\gamma$ by simulating the pattern-basis product (64 chains; R2, R8); the
verifier's dense simulation (V7c) gives the same numbers.

| config ($d,\chi$; $s$) | $B$ | $\eta^{\rm rad}_{\rm ms}$ | $\eta^*_{\rm ms}$ (R1, R7) | $\eta_{\rm as}$ (R2, R8, V7c) | $\eta_{\rm as}/\eta^*_{\rm ms}$ | $2/\lambda_1$ |
|---|---|---|---|---|---|---|
| binary4 (3,3; 1,.6,.3) | 1 | 0.00823 | 0.00474 | 0.0094 | 1.99 | 0.667 |
| | 4 | 0.0317 | 0.0187 | 0.0370 | 1.98 | |
| | 16 | 0.111 | 0.0710 | $\approx0.137$ | $\approx1.93$ | |
| binary4 (3,3; 3,2,1) | 1 | 0.00190 | 0.00107 | 0.00207 | 1.93 | 0.154 |
| | 16 | 0.0257 | 0.0161 | 0.0300 | 1.87 | |
| | 256 | 0.117 | 0.112 | 0.167 | 1.49 | |
| binary8 (2,2; 1,.5) | 1 | $4.36\cdot10^{-5}$ | $3.73\cdot10^{-5}$ | $7.8\cdot10^{-4}$ | $\approx21$ | 0.286 |
| binary8 (3,3; 3,2,1) | 1 | $6.62\cdot10^{-6}$ | $4.82\cdot10^{-6}$ | $4.0\cdot10^{-5}$ | $\approx8.3$ | 0.0435 |
| | 16 | $1.06\cdot10^{-4}$ | $7.70\cdot10^{-5}$ | $6.7\cdot10^{-4}$ | $\approx8.8$ | |
| | 256 | $1.63\cdot10^{-3}$ | $1.21\cdot10^{-3}$ | $\approx9\cdot10^{-3}$ | $\approx8$ | |

Full nonlinear SGD from $10^{-5}$ off the minimum agrees with $\eta_{\rm as}$, not with $\eta^{\rm rad}_{\rm ms}$: binary4
$B=1$ converges at $\eta=0.008$ and diverges at $0.011$ (3/3 each; V7, which also re-ran N7); binary8 $B=1$
converges at $5.1$ and $14.9\,\eta^{\rm rad}_{\rm ms}$ (V7d). v1's statement that the divergence threshold is
$\eta^{\rm rad}_{\rm ms}$ is withdrawn: N7's bracketing of $\eta^{\rm rad}_{\rm ms}$ was a coincidence at $N=4$
($\eta_{\rm as}\approx1.15\,\eta^{\rm rad}_{\rm ms}$ there). So was its explanation ("heavy-tailed kicks carry the iterate into
the nonlinear regime"): the 1D radial product is a.s. stable up to $\eta\lambda_1\approx2$ (N6), but the
multi-dimensional linearization is already unstable at $\eta_{\rm as}$. The ratio $\eta_{\rm as}/\eta^*_{\rm ms}$ grows with $N$
(heavier tails of $Z_P$), so the practical limit at $N=8$ is far above $\eta^*_{\rm ms}\propto3^{-N}B$; the v1 headline
"$\eta\propto3^{-N}$" holds for mean-square stability only. At large $B$, $\eta_{\rm as}$ can exceed the GD limit
$2/\lambda_1$ (binary4, $B=256$: $0.167>0.154$), as for probabilistic stability in Ziyin et al. (2023).

**$\pm1$ features (proven).** For $\pm1$ site features with the teacher in the standard basis, $Z_P^2\equiv1$, so
$\mathcal K\equiv1$ and $\lVert g\rVert^2=\mathrm{tr}H$ for every sample. At $B=1$, $G_1=\lambda^{1/2}\lambda^{1/2\top}$ gives
$\eta^*_{\rm ms}=2/\mathrm{tr}H$, and each step is non-expansive iff $\eta\,\mathrm{tr}H\le2$ (it multiplies the component
along $g$ by $1-\eta\,\mathrm{tr}H$), so $\eta_{\rm as}=\eta^*_{\rm ms}=2/\mathrm{tr}H$ exactly. Binary4 ($d=3$, $\chi=2$,
$s=(1,.6)$): $\mathrm{tr}H=30.04$, threshold $0.0666$. Checks: pattern-basis $\gamma=\mp0.019$ at $0.98$ and
$1.02\times$ (R2); full SGD from $10^{-5}$ (R11): loss decays at $0.90$ and $0.96\times$, grows at $1.04\times$ (rate
$0.07$ per unit time, i.e. $2\gamma$) and $1.13\times$ (3 seeds each). v1 compared N14 with this bound, but N14 ran
at the saddle, where the Gauss–Newton block has $\mathrm{tr}H=19$ ($2/\mathrm{tr}H=0.105$) and the residual adds noise;
N14 (0.05 stable, 0.1 and 0.2 divergent) is therefore not a check of the minimum's bound.

### 5.4 Later plateaus: noise lengthens them, and near the limit it erases learned components

At a saddle where the components in $S$ are learned and the residual is $O(1)$, the directions split into
(i) **learned-channel directions $\mathcal T$** ($J\ne0$; $O(1)$ additive noise from $r(x)=-\sum_{r\notin S}s_rZ_r(x)$;
positive curvature, softest along the piggybacks: $\lambda_-=s_1^{(K-2)/K}(s_1-s_{\rm new})$ for an $m=2$ block);
(ii) **gauge directions** (flat; per-sample gradients are orthogonal to the orbit); (iii) **route directions
$\mathcal N$** (new-channel entries; $J=0$ and population $H=0$ at a $k\ge3$ saddle; per-sample gradients vanish at
$\mathcal N=0$, a Type-II saddle). Learning the next component needs the route to grow, and under GF a balanced
$k=3$ route of amplitude $b$ obeys $\dot b=s_{\rm new}b^2$, i.e. $T_{\rm GF}=1/(s_{\rm new}b_0)$.

**Diagnostic.** The fixed-gauge "new-channel norm" used in v1 is not an escape diagnostic: noise drives it up
along gauge and quartic-flat directions without any route growth (the linearized new-channel block of
binary4 has a positive top Lyapunov exponent, $+0.35$ to $+0.8\,\eta/B$, R9), and after a forget event the
relearned component can sit in the new bond index. Use gauge-invariant quantities: the overlaps $a_r$ and the
second Schmidt value $\sigma_2$ of the bond matricization (`bond_spectra`); for a balanced route $\sigma_2\approx b^3$.

**(a) OU cloud of the learned channel (derived; checked N12, V12, R5).** The piggyback amplitudes fluctuate with

$$\mathrm{Var}(\varepsilon)\simeq\frac\eta B\,\frac{\kappa_w\,s_{\rm new}^2s_1^2}{2(s_1^2-s_{\rm new}^2)},$$

with $\kappa_w$ the feature kurtosis over the piggyback core's sites ($3^{n_w}$ Gaussian). Binary4, $s=(1,.6)$,
Gaussian: $\sqrt{2.53\,\eta/B}$. Measured rms: $0.023$–$0.026$ vs $0.028$ ($\eta/B=3.1\cdot10^{-4}$), $0.036$–$0.037$ vs
$0.040$ ($6.25\cdot10^{-4}$), $0.055$–$0.066$ vs $0.056$ ($1.25\cdot10^{-3}$), $0.068$–$0.094$ vs $0.080$ ($2.5\cdot10^{-3}$,
$B=16$; $0.081$–$0.099$ at $B=4$). **Validity:** within $\sim25\%$ up to $\eta\approx0.5\,\eta^*_{\rm ms}$ of the learned
block ($\eta^*_{\rm ms}=0.0211,0.079,0.248$ at $B=4,16,64$ for component 1 of binary4, pattern formula with
component 2 absent); it fails near $\eta^*_{\rm ms}$ ($0.13$–$0.40$ vs $0.11$ at $B=4$, $\eta=0.02$; V12), where the
variance becomes infinite (5.3), and when $s_{\rm new}\to s_1$ the cloud is $O(1)$ (rms $0.36$–$0.46$ at
$s=(1,.9)$, $\pm1$ features; N15), so the plateau is not localized at all (R10: component 1 is erased within
10–25 time units and $a_1$ wanders over $[-0.1,1.1]$ for 1000 time units).

**(b) Stochastic collapse of the route lengthens the plateau (D3; mechanism known, law heuristic, checked R5).**
Chen, Kunin, Yamamura & Ganguli (NeurIPS 2023) show that an invariant set becomes attractive under SGD when
$L''(0)+D''(0)/2>0$. At a degenerate ($k\ge3$) saddle $L''(0)=0$ on the route, so any noise term makes
$\mathcal N=0$ attractive. Concretely (binary4), the new-channel entry $\alpha=u_2^\top A_1u_2$ is coupled to the mixed
root entry $R_{10}$ by a zero-mean per-sample coefficient (here $q_1=1$)
$c(x)=(f-y)(x)\,z_{2,0}z_{2,1}\,(x^{(2)\top}B_0x^{(3)})=-s_{\rm new}Z_{\rm new}(x)\,z_{2,0}z_{2,1}z_{1,2}z_{1,3}$ with
$\mathbb E\,c^2=3^{n_v}s_{\rm new}^2$ ($n_v=2$ sites on $A$; other unlearned components add $\sum_rs_r^2$);
likewise $\beta$ with $R_{01}$, while $\rho=R_{11}$ has no linear coupling. For one such commuting pair the
log-amplitude drifts at $-\frac12\mathbb Ec^2\,\eta/B$ (the Itô term of a zero-mean multiplicative noise), so the
coherent route $b=(\alpha\beta\rho)^{1/3}$ decays at $\gamma_c\approx\frac23\cdot\frac12\cdot3^{n_v}s_{\rm new}^2\,\frac\eta B=1.08\,\frac\eta B$
(heuristic; for $q_1\ne1$ multiply by $q_1^{2}$, in general by $s_1^{2(K-2)/K}$, the learned amplitudes on the
coupling path). Adding this to GF gives $\dot b=s_{\rm new}b^2-\gamma_cb$, hence

$$\boxed{\frac{T_{\rm SGD}}{T_{\rm GF}}=\frac{-\ln(1-x)}{x},\qquad x=\gamma_cT_{\rm GF}=c_\gamma\frac\eta B\,T_{\rm GF},\qquad\text{trapped (no escape) for }x\ge1.}$$

"Trapped" means the coherent route decays exponentially; escape then needs a rare upward fluctuation of the
log-amplitude or a reset (5.4d), and none was seen. For a $k$-core route, $\dot b=s_{\rm new}b^{k-1}-\gamma_cb$: escape
requires $s_{\rm new}b_0^{k-2}>\gamma_c$, i.e. roughly $(k-2)\gamma_cT_{\rm GF}<1$. Because $b_0$ scales with the init, **at fixed $\eta/B$ SGD gets stuck on a later
degenerate plateau once $\sigma$ is small enough**, instead of reproducing $T\propto b_0^{-(k-2)}$. Strict ($k=2$)
saddles are not affected this way (their escape is exponential at rate $\lvert\lambda_-\rvert$ and only needs
$\lvert\lambda_-\rvert>\gamma_c$).

R5 (binary4, $d=3$, $\chi=2$, $s=(1,.6)$, Gaussian; start on the aligned route: component 1 learned, component 2
with per-core amplitude $b_0$, so GF stays aligned and $T_{\rm GF}$ is exact; event $a_2>1/2$):

| $\eta/B$ ($B,\eta$) | $b_0$ ($T_{\rm GF}$) | SGD escape times | $T/T_{\rm GF}$ | $c_\gamma$ from the law |
|---|---|---|---|---|
| $3.1\cdot10^{-4}$ (64, 0.02) | $10^{-2}$ (165) | 168, 170 | 1.02, 1.03 | 0.7, 1.1 |
| | $3\cdot10^{-3}$ (554) | 626 | 1.13 | 1.28 |
| $6.25\cdot10^{-4}$ (64, 0.04) | $3\cdot10^{-3}$ (554) | 786 | 1.42 | 1.52 |
| $1.25\cdot10^{-3}$ (16, 0.02; 64, 0.08) | $10^{-2}$ (165) | 186, 190, 194; 190, 192, 194 | 1.13–1.18 | 1.05–1.37 |
| | $3\cdot10^{-3}$ (554) | none by 1200; $\sigma_2^{1/3}$: $3.0\to0.94\cdot10^{-3}$ | $>2.2$ (trapped) | $\ge1.44$; decay 1.6–2.0 |
| | $10^{-3}$ (1665) | none by 800 (2 runs); $\sigma_2^{1/3}$: $10^{-3}\to1.9\cdot10^{-4}$ | trapped | $\ge0.48$; decay 1.6–2.5 |
| $2.5\cdot10^{-3}$ (16, 0.04; 4, 0.01) | $10^{-2}$ (165) | 250 ($B=16$); 258, 326 ($B=4$); 262 after a dip; none by 500 after a dip; 246 at a forget | 1.5–2.0 | 1.44–1.92 |
| | $10^{-3}$ (1665) | none by 400 (2 runs, dips at $t=44$, 92); route decays | trapped | |

"$c_\gamma$ from the law" solves $T/T_{\rm GF}=-\ln(1-x)/x$ for $x$; "decay" is
$(-\frac{d}{dt}\ln\sigma_2^{1/3}+s_{\rm new}b)/(\eta/B)$ along trapped runs. No run escaped faster than GF. The same
$\eta/B$ at different $B$ gives the same delays (16 vs 64 at $1.25\cdot10^{-3}$), as the SDE limit requires. The
delays give $c_\gamma=1.0$–$1.9$ (median 1.35), the decay rates of trapped routes $1.6$–$2.5$, the pair
estimate 1.08. Status: the qualitative law (lengthening, trapping at $x\ge1$) is checked on one model;
$c_\gamma$ is empirical for binary4 and must be measured for other trees and teachers.

**(c) Noise-opened negative curvature: real coupling, no net growth (derived for a commuting pair; empirical).**
The OU cloud gives the pair $(\alpha,R_{10})$ a fluctuating population coupling $\varepsilon_A-s_{\rm new}\varepsilon_B$ (v1:
$\partial^2L/\partial R_{01}\partial\beta=-s_{\rm new}\varepsilon_A$, re-derived by the verifier). Its sign changes on the time scale $1/\lambda_-$,
and for a commuting pair a zero-mean coupling only makes the log-amplitude diffuse, with no drift; the white
part (b) adds a negative drift. No escape faster than GF was seen in any run (R5: 21 runs, 4 noise levels). The
v1 "noise-driven escapes" were forget-and-relearn events (next item).

**(d) Near the stability limit: excursions, forget-and-relearn, divergence (D4; empirical, mechanism heuristic).**
At a plateau the learned block obeys an affine random recursion (additive noise from the $O(1)$ residual, heavy-
tailed multiplicative noise), whose stationary law has power-law tails with index $\alpha$ solving $h(\alpha)=1$,
$\alpha>2$ iff $\eta<\eta^*_{\rm ms}$ (Kesten–Goldie; for SGD, Gürbüzbalaban, Şimşekli & Zhu, ICML 2021, Thm 2).
Large excursions then knock out learned components:

- $\eta\le0.25$–$0.32\,\eta^*_{\rm ms}$ ($\eta/B\le1.25\cdot10^{-3}$ at $B=16,64$): no excursion with $a_1<0.5$ in $\approx6000$
  time units over 13 runs (R5); $\min a_1\ge0.6$.
- $\eta\approx0.5\,\eta^*_{\rm ms}$ ($\eta/B=2.5\cdot10^{-3}$): excursions $a_1<0.5$ in 3 of 4 runs at $B=16$ within 250 time
  units (one to $a_1=0$; rate $\approx2\cdot10^{-3}$ per unit time); $\min a_1=0.56$–$0.69$ at $B=4$ (R5).
- $\eta\approx0.7$–$1.2\,\eta^*_{\rm ms}$ ($B=4$, $\eta=0.015$–$0.025$; V12, V13 reproducing N13): component 1 is
  erased (loss back to $\approx L(0)$) and relearned, often through the new bond index, after which component 2 is
  learned within 40–80 time units (the reset leaves an $O(1)$ random state, where plateaus are short); or the run
  diverges (2 of 4 at $0.71\,\eta^*_{\rm ms}$). Event times depend on $b_0$ (245 vs 550 for the same noise
  seed); the "same time at both $b_0$" coincidences of N13 were all divergences or blow-ups, which are driven by
  the learned-channel noise and do not depend on $b_0$.

So near the limit **SGD breaks the saddle sequence (forgets and relearns) rather than compressing it**, and the
v1 claims "independent of $b_0$" and "up to $\sim10^3\times$ faster than GF" are withdrawn.

**Corrected conjecture D3/D4 (to test at scale).** Under online SGD with Gaussian features: the first plateau
keeps its GF law (5.2). Later degenerate plateaus are lengthened by the factor $-\ln(1-x)/x$ and become
infinite when $x=c_\gamma(\eta/B)T_{\rm GF}\ge1$ (for $k=3$). Learned components survive a plateau of length $T$
only if $\eta\lesssim0.25\,\eta^*_{\rm ms}(B)$; near $\eta^*_{\rm ms}$ they are repeatedly erased. Binary4 is a depth-3 DLN,
so the same should hold for deep linear networks; the aligned per-mode SDE of Corlouer et al. (2026) has no
route block and cannot show it.

## 6. Open problems

- **Collapse coefficient.** A derivation of $c_\gamma$ beyond the commuting-pair estimate: the top Lyapunov
  exponent of the full route block (route entries, mixed root entries, OU-driven couplings) projected on the
  coherent route, for general trees, route degrees $k$ and feature kurtosis.
- **Excursion rate.** A formula for the rate of learned-component erasures as a function of $\eta/\eta^*_{\rm ms}$ and $B$
  (Kesten tail constant of the learned block's affine recursion).
- **Almost-sure threshold.** A formula, or a bound sharper than $\eta_{\rm as}\ge\eta^*_{\rm ms}$, for the top Lyapunov
  exponent in the pattern basis, and its growth with $N$ ($\eta_{\rm as}/\eta^*_{\rm ms}\approx2$ at $N=4$; for binary8 $\approx8$
  with $d=3$ and $\approx21$ with $d=2$).
- **Mean-square threshold at saddles.** The learned block at a plateau adds the residual Hessian
  $r(x)\nabla^2f_x$ to the multiplicative noise; its exact second-moment threshold (not only the Gauss–Newton
  block's) would sharpen the plateau-safety rule.
- Whether EoS oscillation of a learned component can pump energy into a dormant route (parametric resonance).
  Not seen (N5), one tree tried.
- Minimal sharpness over minima with $\chi_e>R$ (conjecture: still $Ks_1^{2-2/K}$).
- SGD noise and the conservation-law trap of `theory_saddles.md` §8: noise moves flat gauge directions, so SGD
  might rescue stalled HT runs, while route collapse (5.4b) works the other way.

## 7. Revision log: verifier issues and responses

| # | issue (severity) | verdict | change |
|---|---|---|---|
| 1 | N13 "escapes" are forget-and-relearn events (major) | agree | withdrawn; 5.4 rewritten with gauge-invariant diagnostics; new route-start runs (R5) show that stable-regime SGD **lengthens** later plateaus and traps below a threshold; N15's "escape" was also not a saddle escape (R10) |
| 2 | $\eta_{\rm ms}$ is neither the MS nor the divergence threshold (major) | agree | exact MS threshold derived in closed form via the pattern basis (reproduces the verifier's 0.58–0.86 ratios to $10^{-4}$); almost-sure thresholds computed and tabulated (5.3); attribution to Wu–Ma–E, Mulayoff–Michaeli, Chemnitz–Engel, Ziyin et al. |
| 3 | P4 wording (moderate) | agree | component 1's error stops converging; later events still lower the loss; floor at $L_{\rm EoS}$ (3.3, P5) |
| 4 | P7 mean biased, sd $0.8\times$ (moderate) | partly agree | "unbiased" was wrong (the formula itself predicts a positive delay). With 160 new seeds per condition (R4) the aligned-init delay and spread agree with the formula; the verifier's 24-seed sets lie at the extremes of the sampling scatter. Generic init: spread $1.2\times$, delay smaller. The 1D SDE is now called leading order |
| 5 | excess sharpness is $O(\sigma^4)$ (moderate) | agree | derivation added (3.1) |
| 6 | P2 incomplete (moderate) | agree | $\lambda_{\max}=\max_P\lambda_P$ at any aligned minimum; counterexample confirmed by autograd (R6) |
| 7 | P6 pre-asymptotic for some seeds (minor) | agree | path formula added; slope tagged asymptotic; average over seeds (4.1) |
| 8 | chaos onset values (minor) | agree | 2.632 / 2.639 / 2.647 |
| 9 | "no reordering (proven)" needs a separation condition (minor) | agree | condition stated (4.2) |
| 10 | attribution (minor) | agree | Du–Hu–Lee 2018 and Hofmann 2026 (2.4); Ghosh et al. Lemma 1 for binary4's spectrum (2.2); Ghosh Thm 1 orbit is implicit (3.3) |
| 11 | validity ranges and consistency (minor) | agree | OU validity stated ($\lesssim0.5\,\eta^*_{\rm ms}$, 5.4a); "5 trees"; one saddle-safety rule (5.4d); N10's $B=64$ sd replaced by 20-seed data; P3's window stated as an N11 observation; $\pm1$ threshold now exact at the minimum ($2/\mathrm{tr}H$, checked R11) and N14 no longer cited as its check |

## 8. Numerical checks (float64; core library only)

v1 checks (`scratchpad/theory2/optimizer/`; N1–N9 and N11 on the library before the 19:44 site-basis change,
the rest after it; basis-independent quantities agree across versions):

| # | what | result | status in v2 |
|---|---|---|---|
| N1 | autograd Hessian at the balanced minimum vs pattern rule, 5 trees, $K=3$–$7$; $GL$ gauges | agree to $\le6\cdot10^{-15}$; gauged minima $\ge1.43\times$ (binary4), $1.93\times$ (mps4) sharper | kept (V1 reproduces) |
| N2 | sharpness along GF | $0.06\to3.0000$ at event 1, flat after; mps4 $4.0015$–$4.0032$ | kept; excess reinterpreted as $O(\sigma^4)$ |
| N3 | GD vs GF event times | aligned within 1%; generic slope 0.70 per halving | kept; slope asymptotic (V3) |
| N4 | balanced scalar map, $K=3,4,7$ | thresholds, period-2 amplitude within 2% | kept; chaos values corrected |
| N5, N5b | full-TTN GD, $\eta\in[0.3,1.5]$; 12-seed order check | EoS confined to component 1 below $2/\lambda_2$; same event order 12/12 | kept |
| N6 | 1D SGD thresholds | 1D a.s. threshold $\eta\lambda\approx2$ | kept, but irrelevant (multi-D is unstable much earlier) |
| N7 | online SGD at the minimum, $B=1,4,16$ | brackets $\eta^{\rm rad}_{\rm ms}$ | reinterpreted: brackets $\eta_{\rm as}\approx1.15\,\eta^{\rm rad}_{\rm ms}$ |
| N8, N13a | GF escape from the component-1 saddle | $T_{\rm GF}b_0=2.40$ (earlier library), 6.15 (current) | kept |
| N9, N13 | online SGD from that saddle | "noise-driven escapes" | **withdrawn**: forget-and-relearn / blow-ups (V12, V13) |
| N10 | SGD first escape, 5 batch sizes | means and sds from 3–8 seeds | superseded by R4 |
| N11 | GD conservation identity; EoS selection | identity to $1.8\cdot10^{-16}$; selected sharpness $2/\eta-0.02$ | kept (V2 reproduces) |
| N12 | SGD at the saddle, $\eta/B\le2.5\cdot10^{-3}$ | OU rms; no escape by 2500 | kept (consistent with route collapse) |
| N14, N15 | $\pm1$ features at the saddle | OU rms; "escape" for $s=(1,.9)$ | N15 escape withdrawn (R10); N14 not a check of the minimum's bound |

v2 checks (`scratchpad/theory2/optimizer_rev/`):

| # | what | result |
|---|---|---|
| R1 | exact MS threshold: pattern formula vs dense $n_\theta^2$-dimensional operator (4 small cases) and vs V7b | $\le1.4\cdot10^{-15}$ relative; 0.0047396 / 0.018708 / 0.070988 vs the verifier's 0.00475 / 0.01871 / 0.07099; mps4 0.656 and binary8 0.856 of radial, as the verifier found |
| R2, R8 | top Lyapunov exponent of linearized SGD in the pattern basis | binary4 $\eta_{\rm as}=1.99,1.98,1.93\,\eta^*_{\rm ms}$ ($B=1,4,16$), (3,2,1): 1.93, 1.87, 1.49 ($B=1,16,256$); binary8 ($d=2$) $\approx21\times$ ($B=1$), ($d=3$; 3,2,1) $\approx8.3$, $8.8$, $8$ ($B=1,16,256$; the last uncertain by about 1); matches V7c where compared |
| R3 | SGD restricted to the aligned manifold (2000 chains) | delay $+1.03\pm0.08$ (formula 1.16), sd 3.60 (3.84) |
| R4 | full-network SGD first escape, 160 + 160 + 70 seeds | table in 5.2 |
| R5 | SGD from the aligned route start, 4 noise levels, 3 route amplitudes | escapes delayed $1.02$–$2\times$, trapped for $x\ge1$; no run faster than GF |
| R6 | pattern rule vs autograd at imbalanced minima | $\lambda_{\max}=14.4025$ (radial-2) as predicted |
| R7 | exact MS thresholds for the experiment configurations | tables in 5.3 and 9 (P8) |
| R9 | linearized route block at the saddle | top Lyapunov exponent of the new-channel block $+0.35$ to $+0.8\,\eta/B$ (norm grows, route does not) |
| R10 | N15 re-run with $a_1$ tracking | component 1 erased within 10–25 time units; the "escape" at 1065 comes from that state |
| R11 | full SGD at the minimum, $\pm1$ features | stable at $0.90,0.96\times2/\mathrm{tr}H$, unstable at $1.04,1.13\times$ (3/3 each) |

## 9. Predictions for experiments

Setting unless stated: whitened Gaussian inputs, odeco teacher $s_1>s_2>\dots$, $q_r=s_r^{1/K}$,
$\eta_c=2/(Ks_1^{2-2/K})$, time $t=\eta\cdot$steps, event $r$ = first $a_r>1/2$. "Default" = the `optim-samples`
configurations $d=\chi=3$, $s=(3,2,1)$. Constants for them (pattern rule, R7):

| tree | $K$ | $\lambda_1=Ks_1^{2-2/K}$ | $\lambda_2$ | $\eta_c$ | $2/\lambda_2$ | $L_{\rm EoS}$ slope | $\kappa=3^N-1+\sum_{r\ge2}s_r^2/s_1^2$ |
|---|---|---|---|---|---|---|---|
| binary4 | 3 | 12.98 | 10.15 | 0.1541 | 0.1971 | 2.761 | 80.56 |
| mps4 | 4 | 20.78 | 11.31 | 0.0962 | 0.1768 | 2.250 | 80.56 |
| binary8 | 7 | 46.03 | 22.97 | 0.0435 | 0.0871 | 1.778 | 6560.6 |
| mps8 | 8 | 54.71 | 26.91 | 0.0366 | 0.0743 | 1.714 | 6560.6 |

**P1 (sharpness at the GF minimum).** From small fan-in init, $\lambda_{\max}=Ks_1^{2-2/K}\,(1+c\,\sigma^4)$ with
$c=O(1)$ (binary4, $s_1=1$: excess$/\sigma^4\in[0.4,1.3]$ for $\sigma\in[0.15,0.6]$); in general
$\lambda_{\rm rad}=Ks_1^{2-2/K}(1+\frac1{2K}\sum_vu_v^2)$ with $u_v-u_p=(Q_v)_{11}/q_1^2$ from the conserved quantities. The second
eigenvalue is $\lambda_2$ of the table (other configurations: binary4 $s=(1,.6,.3)$: 2.2174; mps4: 1.8590; binary8
$d=\chi=2$, $s=(1,.5)$: 2.1333). The whole spectrum follows the pattern rule (2.2).

**P2 (imbalanced aligned minima).** $\lambda_{\max}=\max_P\lambda_P$ with the actual amplitudes; radial-$r$ is
$s_r^2\sum_va_{v,r}^{-2}$. Test case: binary4, $s=(1,.6)$, component 2 with leaf $a^2=0.05$, root $a=12$:
$\lambda_{\max}=14.40$, $\lambda_2=8.25$. Root-vs-rest imbalanced fan-in inits give $\lambda_{\max}>Ks_1^{2-2/K}$, growing with
the imbalance.

**P3 (EoS selection).** For $2/\lambda_{GF}<\eta<\eta_c$ GD converges to a minimum with sharpness slightly below
$2/\eta$ (binary4 observation: deficit 0.022–0.028) and less imbalance than GF's; per step the aligned imbalance
contracts by $1-\eta^2\rho^2\Pi^2/(a_v^2a_p^2)$ exactly ($\Pi=\prod_ua_u$, $\rho=s-\Pi$).

**P4 (GD above $\eta_c$).** Plateaus before component 1's event are unaffected. At component 1's event its error
stops converging (period-2 oscillation of $a_1$); later events still occur within $O(\eta)$ of their GF times
and still lower the loss; once all components are learned the time-averaged loss floors at $L_{\rm EoS}$ (P5).
Divergence of the full network: binary4 ($s=(1,.6,.3)$) between $1.5$ and $1.65\,\eta_c$; the balanced map alone at
$2.16\,\eta_c$ ($K=3$).

**P5 (EoS loss and route to chaos).** For $\eta_c<\eta<\min(2/\lambda_2,1.245\,\eta_c)$ only component 1 oscillates, with
$L_{\rm EoS}\approx\frac{3K^2s_1^2}{4(K-1)(5K-4)}(\eta/\eta_c-1)$ (slopes in the table; binary4 $s_1=1$: 0.307, measured
loss $1.45\cdot10^{-2}$ at $\eta=1.05\,\eta_c$ vs 0.0153). Period 4 at $\eta/\eta_c=1.245$ ($K=3$), $1.247$ ($K=4$), $1.250$
($K=7$); chaos at $1.316$, $1.320$, $1.324$.

**P6 (GD vs GF event times).** For $\eta<\eta_c$: $t_r^{\rm GD}-t_r^{\rm GF}=\frac\eta2\ln\frac{\lvert\nabla L(t_r)\rvert}{\lvert\nabla L(0)\rvert}+O(\eta^2)$
along the GF path (computable from the GF run; accurate to $0.15\,\eta$ without a long alignment phase).
Asymptotically in $\sigma\to0$, $\Delta t_r/\eta=\frac{K-1}2\ln\frac1\sigma+c_r$: slope per halving of $\sigma$ = 0.69 (binary4),
1.04 (mps4), 2.08 (binary8), 2.43 (mps8); single seeds can be pre-asymptotic (mps4: 0.47–0.61 at
$\sigma\in[0.05,0.2]$), so average $\ge5$ seeds. Same event order as GF (aligned: whenever GF gaps exceed the
shift differences). Plateau exponents unchanged.

**P7 (SGD, first plateau).** Exponent $K-2$ unchanged. Mean delay
$\mathbb Et_1^{\rm SGD}-t_1^{\rm GF}\approx\frac\eta2(K-1)(1+\kappa/B)\ln(a_{1/2}/a_0)>0$ and
$\mathrm{sd}(t_1)\approx\sqrt{\kappa(\eta/B)t_1^{\rm GF}}$, with $a_0$ from $t_1^{\rm GF}=a_0^{-(K-2)}/((K-2)s_1)$. Accuracy:
$\approx10$–$15\%$ for aligned init; generic init: spread $\approx1.2\times$, delay smaller. Example (binary4,
$s=(1,.6,.3)$, $t_1^{\rm GF}=59$, $B=16$, $\eta=0.05$): delay $+1.16$ (measured $+0.91\pm0.30$), sd 3.85 (measured
4.0). Needs $\ge100$ seeds.

**P8 (SGD stability at the minimum).** Mean-square stable iff $\eta<\eta^*_{\rm ms}(B)=2/\lambda_{\max}(G_B)$ (5.3);
converges a.s. iff $\eta<\eta_{\rm as}(B)$, with $\eta^*_{\rm ms}\le\eta_{\rm as}$. Values (R7, R2, R8):

| tree (default) | $B=1$ | $B=4$ | $B=16$ | $B=64$ | $B=256$ | $\eta_{\rm as}/\eta^*_{\rm ms}$ |
|---|---|---|---|---|---|---|
| binary4 $\eta^*_{\rm ms}$ | 0.00107 | 0.00423 | 0.0161 | 0.0528 | 0.112 | 1.93 ($B=1$), 1.87 (16), 1.49 (256) |
| mps4 $\eta^*_{\rm ms}$ | 0.000765 | 0.00301 | 0.0113 | 0.0357 | 0.0708 | not computed |
| binary8 $\eta^*_{\rm ms}$ | $4.82\cdot10^{-6}$ | $1.93\cdot10^{-5}$ | $7.70\cdot10^{-5}$ | $3.07\cdot10^{-4}$ | $1.21\cdot10^{-3}$ | 8.3 ($B=1$), 8.8 (16), $\approx8$ (256) |
| mps8 $\eta^*_{\rm ms}$ | $4.13\cdot10^{-6}$ | $1.65\cdot10^{-5}$ | $6.59\cdot10^{-5}$ | $2.63\cdot10^{-4}$ | $1.03\cdot10^{-3}$ | not computed |

Test: online SGD from $10^{-5}$ off the balanced minimum diverges for $\eta>\eta_{\rm as}$ and converges below
(binary4 $s=(1,.6,.3)$, $B=1$: $\eta_{\rm as}=0.0094$; converged at 0.008, diverged at 0.011). The loss growth rate just
above $\eta_{\rm as}$ is $2\gamma$ (pattern-basis Lyapunov exponent). For $\pm1$ site features with the teacher in the
standard basis and $B=1$: $\eta^*_{\rm ms}=\eta_{\rm as}=2/\mathrm{tr}H$ exactly (binary4 $d=3$, $\chi=2$, $s=(1,.6)$: 0.0666).

**P9 (SGD on later degenerate plateaus: lengthening and trapping).** For a later plateau whose escape needs a
$k=3$ route with GF duration $T_{\rm GF}$:
$T_{\rm SGD}/T_{\rm GF}=-\ln(1-x)/x$ with $x=c_\gamma(\eta/B)T_{\rm GF}$, and **trapped** (the route decays; no escape seen) when
$x\ge1$; for a $k$-core route
roughly $x_k=(k-2)c_\gamma(\eta/B)T_{\rm GF}$. Binary4 ($s=(1,.6)$, Gaussian): $c_\gamma\approx1.35$ (range 1.0–2.5).
Heuristic scaling (commuting-pair estimate, 5.4b):
$c_\gamma\approx\frac mk\cdot\frac12\,s_1^{2(K-2)/K}\big(3^{n_v}s_{\rm new}^2+\sum_{r\notin S,\,r\ne{\rm new}}s_r^2\big)$, with $m$ the number of
route cores coupled linearly to a mixed bond entry and $n_v$ the number of sites on such a core ($m=2$, $k=3$,
$n_v=2$ for binary4): 1.08 for $s=(1,.6)$, and $\approx26$ for the default binary4 second plateau ($s=(3,2,1)$), so
trapping should set in at $\sim20\times$ smaller $\eta/B\cdot T_{\rm GF}$ there. Consequences to test: (i) at fixed $\eta/B$ the second-plateau length grows faster than
$\sigma^{-(k-2)}$ and diverges at a finite $\sigma_c$ where $x=1$; (ii) $T_{\rm SGD}$ depends on $\eta/B$ only (not on $\eta$
and $B$ separately) while $\eta\lesssim0.25\,\eta^*_{\rm ms}$; (iii) the gauge-invariant route amplitude $\sigma_2^{1/3}$
($\sigma_2$ = second Schmidt value across the route's bond, `bond_spectra`) decays at $\approx c_\gamma\eta/B-s_{\rm new}b$ per
unit time on trapped runs; (iv) SGD never
escapes a later degenerate plateau faster than GF in the stable regime. The first plateau is exempt (P7).

**P10 (plateau-crossing safety).** The learned-channel OU cloud has
$\mathrm{rms}(\varepsilon)=\sqrt{(\eta/B)\kappa_ws_{\rm new}^2s_1^2/(2(s_1^2-s_{\rm new}^2))}$ (binary4 Gaussian $s=(1,.6)$:
$\sqrt{2.53\,\eta/B}$), within $\sim25\%$ for $\eta\le0.5\,\eta^*_{\rm ms}(B)$. Excursions erasing a learned component
($a_r<1/2$): none at $\eta\le0.25\,\eta^*_{\rm ms}$, rate $\approx2\cdot10^{-3}$ per unit time at $0.5\,\eta^*_{\rm ms}$ (binary4), and
forget-and-relearn or divergence within a few hundred time units at $0.7$–$1.2\,\eta^*_{\rm ms}$. When
$s_{\rm new}/s_1\to1$ the cloud is $O(1)$ (rms $\ge0.3$) and there is no localized plateau. **Recommendation:** runs that
must cross later plateaus should use $\eta\le0.25\,\eta^*_{\rm ms}(B)$ of the learned block (table P8, or the pattern
formula with only the learned components) and classify events with $a_r$, loss jumps and $\sigma_2$, never with
fixed-gauge channel norms.
