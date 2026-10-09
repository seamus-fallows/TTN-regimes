# Theory: regime boundaries and large-bond-dimension limits (Q5, H4)

Theory note for the `regime-map` agent (brief `notes/briefs/regime-map.md`). Model, loss and
diagnostics are those of `notes/model.md` and `notes/diagnostics.md`.

Status tags: **proven**; **derived** (under the stated assumptions); **heuristic**; **conjecture**;
**empirical** (numerical checks in §7, exact population gradient flow unless stated otherwise).
Literature was checked by fetching arXiv pages directly. The shared web-search budget ran out
during this session and `notes/literature.md` did not exist yet, so a citation marked
*[unverified]* comes from memory.

## 0. Results at a glance

| # | result | status |
|---|---|---|
| R1 | Per-core tangent kernel $J_vJ_v^\top=(\otimes_c\Phi_c\Phi_c^\top)\otimes I\otimes\Psi_v\Psi_v^\top$. For every $\chi$: $\mathbb E\,\Theta=\frac{\alpha^2}{D}\sum_v\eta_vF_v/s_v^2\;I_D$ and $\mathbb E[W_0\otimes W_0]=\frac{\alpha^2}{D}I_D$ | proven; empirical |
| R2 | $\operatorname{rank}\Theta$ equals the dimension of the fixed-bond-rank TTN manifold. The lazy (linearized) fit reaches zero loss iff **every bond is full** | proven; empirical |
| R3 | $\Theta\approx\bar\lambda I$ only when every bond is strongly overparameterized ($\chi_e\gg D_e$). At capped full bonds $\lambda_{\min}/\bar\lambda\sim10^{-4}$ | derived; empirical |
| R4 | Centred model: $\delta_K=c_1x+O(x^2)$ with $x=\Vert W^\star\Vert/\alpha$. Closed form for $c_1$ and $c_\theta$ at large $\chi$ ($N=4$) | derived; empirical |
| R5 | Two-copy model: the first-order kernel change cancels, so $\delta_K=c_2x^2$, and $\delta_\theta$ is half the centred value | derived; empirical |
| R6 | With a deficient bond the centred model is **unrealizable** at every $\sigma$. The two-copy model is then lazy-then-rich | derived; empirical |
| R7 | Exact S2S boundary for aligned odeco dynamics. The per-core boundary $a_{0c}(K)\to1$; the function-space boundary $u_{0c}\to s\,q(1-q)/39.85$ | proven (given the aligned reduction) |
| R8 | Random init behaves like aligned init with per-core amplitude $\kappa_K\sigma$, with $\kappa_3=0.27$ and $\kappa_5=0.50$. So $\sigma_c=0.24$ ($K=3$) and $0.65$ ($K=5$); $\sigma_c\to1$ iff $\kappa_K\to1$ | heuristic; empirical |
| R9 | TTN muP: fan-in init everywhere except the root, which is scaled by $(\sum\text{child dims})^{-1/2}$; $\eta_v=\chi_v/F_v$. The spectral condition must use the **injective norm** | derived; empirical |
| R10 | $\chi\to\infty$: SP gives $\delta_K\propto\chi^{-3}$, $\delta_\theta\propto\chi^{-3/2}$; NTK gives $\chi^{-1}$, $\chi^{-1/2}$; muP gives $\Theta(1)$ for both | derived ($N=4$); empirical |
| R11 | Root imbalance at fixed $\alpha$: a small root makes training lazy ($\delta_K\propto\lambda^2$). A large root freezes the root, leaving a degree-$(K-1)$ model that is not lazy | derived (aligned); empirical |

## 1. Notation

- Tree with $K$ cores. Core $v$ has fan-in $F_v$ (the product of its child-leg dimensions, physical $d_i$
  and child bonds $\chi_c$) and fan-out $\chi_v$ (its parent bond; $\chi_{\rm root}=1$). $D=\prod_i d_i$
  is the dimension of tensor space.
- Init $G_v=s_vZ_v/\sqrt{F_v}$ with $Z_v$ iid $N(0,1)$; default $s_v=\sigma$. In §4 we write
  $s_v=\sigma m_v$. Function-space scale $\alpha:=\prod_vs_v$, so $\mathbb E\Vert W_0\Vert_F^2=\alpha^2$
  exactly.
- Gradient flow (GF) $\dot G_v=-\eta_v\,\partial L/\partial G_v$ with $L=\tfrac12\Vert W-W_{\rm target}\Vert_F^2$
  (whitened inputs).
- $J_v=\partial W/\partial G_v$ (a $D\times F_v\chi_v$ matrix). The tensor-space tangent kernel is
  $\Theta=\sum_v\eta_vJ_vJ_v^\top$ ($D\times D$). The probe kernel of `notes/diagnostics.md` is
  $K=X\Theta X^\top$, where the rows of $X$ are $x_1\otimes\cdots\otimes x_N$. With per-core learning
  rates, the $\eta$-weighted $\Theta$ is the kernel that governs $\dot f$.
- For the edge between $v$ and its parent, $\Phi_v\in\mathbb R^{D_v\times\chi_v}$ is the subtree tensor
  below $v$ ($D_v=\prod_{i\in{\rm sub}(v)}d_i$) and $\Psi_v\in\mathbb R^{(D/D_v)\times\chi_v}$ is the
  environment, so the matricization of $W$ across this edge is $\Phi_v\Psi_v^\top$.
- $T$ is what the network has to add. $T=W^\star$ for the centred model (target $W^\star+W_0$,
  `ShiftedTeacher`) and for the two-copy model ($W_0=0$, `antisymmetric_double` + mask).
  Define $x:=\Vert T\Vert_F/\alpha$.
- Kernel scale $\bar\lambda:=\operatorname{tr}\Theta(0)/D$ and rescaled time $\tau:=\bar\lambda t$.
- Diagnostics: $\delta_K=\Vert\Theta_T-\Theta_0\Vert_F/\Vert\Theta_0\Vert_F$ at the end $T$ of
  training, and $\delta_\theta=\Vert\theta_T-\theta_0\Vert/\Vert\theta_0\Vert$.
- **Reference case for numbers:** binary $N=4$ ($K=3$), $d=3$ ($D=81$), odeco teacher with weights
  $(3,2,1)$, $\Vert T\Vert=\sqrt{14}=3.742$ (the regime-map teacher).
- For $N=4$ binary, the TTN is the 3-factor matrix factorization $W=ABC^\top$ of a $d^2\times d^2$
  matrix. $A,C$ ($d^2\times\chi$) are the bottom cores and $B$ ($\chi\times\chi$) is the root. Both bonds
  cut $\{12|34\}$.

## 2. (a) Lazy regime: tangent kernel at init and the lazy boundary

### 2.1 Structure, scale, rank

**R1a (proven).** $W$ is linear in $G_v$: $W=(\Phi_{c_1}\otimes\Phi_{c_2}\otimes I_{{\rm phys}(v)}\otimes\Psi_v)\,{\rm vec}\,G_v$.
Hence
$$J_vJ_v^\top=\Phi_{c_1}\Phi_{c_1}^\top\otimes\Phi_{c_2}\Phi_{c_2}^\top\otimes I_{{\rm phys}(v)}\otimes\Psi_v\Psi_v^\top$$
(legs permuted to site order). For $N=4$:
$\Theta=I\otimes CB^\top BC^\top+ABB^\top A^\top\otimes I+AA^\top\otimes CC^\top$ (for $\eta=1$).

**R1b (proven).** For any bond dimensions,
$$\mathbb E\,\Theta=\frac{\alpha^2}{D}\sum_v\frac{\eta_vF_v}{s_v^2}\;I_D,\qquad \mathbb E[W_0\otimes W_0]=\frac{\alpha^2}{D}I_D .$$
Proof: the cores are independent, and $\Phi_c$ and $\Psi_v$ are built from disjoint sets of cores.
The init is invariant under $O(d_i)$ on every physical leg, and $\otimes_iO(d_i)$ acts irreducibly, so
$\mathbb E\,\Phi_c\Phi_c^\top\propto I$ and $\mathbb E\,\Psi_v\Psi_v^\top\propto I$. Their traces are
$\mathbb E\Vert\Phi_c\Vert_F^2=\chi_c\prod_{u\in{\rm sub}(c)}s_u^2$ and
$\mathbb E\Vert\Psi_v\Vert_F^2=\prod_{u\notin{\rm sub}(v)}s_u^2$ (induction from the leaves and from the
root). Consequences:

- **Kernel scale.** With $s_v=\sigma$ and $\eta=1$:
  $$\bar\lambda=\sigma^{2(K-1)}\frac{\sum_vF_v}{D}.$$
  The factor $\sum_vF_v/D$ matters when comparing trees and values of $\chi$, so rescale time by
  $\bar\lambda$, not by $\sigma^{2(K-1)}$ alone. For the two-copy model, $\Theta_{2}(0)=2\Theta(\theta_0)$.
- **Prior function covariance (NNGP).** For every $\chi$,
  $\mathbb E f(x)f(x')=\frac{\alpha^2}{D}\prod_i\langle x_i,x_i'\rangle$, the product linear kernel. It
  becomes a Gaussian process as $\chi\to\infty$; Guo & Draper (2021, arXiv:2101.02333) prove GP
  convergence of tensor networks at infinite bond dimension. The mean NTK is the same kernel times
  $\bar\lambda$.
- **Kernel shares.** Core $v$ carries the fraction $\frac{\eta_vF_v/s_v^2}{\sum_u\eta_uF_u/s_u^2}$ of
  $\operatorname{tr}\Theta$. With fan-in init and $\eta=1$, the cores with the largest fan-in dominate:
  binary roots and internal cores, which have $F=\chi^2$.

Empirical ($N=4$, $d=3$, 5 seeds, $\sigma=1$): mean eigenvalue $0.297\pm0.106$ (predicted 0.333) at
$\chi=3$, $1.30\pm0.44$ (1.222) at $\chi=9$, $51.6\pm3.5$ (50.8) at $\chi=64$. Root share at $\chi=16$:
0.933 and 0.943 (predicted 0.934).

**R2 (proven).** $\operatorname{rank}\Theta=\dim T_{W_0}\mathcal M_\chi$, where $\mathcal M_\chi$ is the
set of tensors with bond ranks $\le\chi_e$. For generic init with every $\chi_e\le\min(D_e,D/D_e)$,
$$\operatorname{rank}\Theta=\sum_vF_v\chi_v-\sum_e\chi_e^2,$$
the dimension of the fixed-rank TTN manifold (same count as Uschmajew & Vandereycken 2013 for HT).
$\operatorname{rank}\Theta=D$ iff every bond is full ($\chi_e\ge\min(D_e,D/D_e)$). If a bond is
deficient, $\mathcal M_\chi$ is a proper subvariety, so $\operatorname{rank}J<D$ everywhere. If all
bonds are full, the image contains an open set, so $J$ is generically surjective.

- **Consequence.** The lazy (linearized) solution reaches zero loss iff all bonds are full. Otherwise
  it stops at $\tfrac12\Vert P_\perp T\Vert^2$, where $P_\perp$ projects onto the normal space of
  $\mathcal M_\chi$ at $W_0$.
- **$N=4$ (derived).** The tangent space is $\{X:P_A^\perp XP_C^\perp=0\}$, and the column spaces of
  $A$ and $C$ are Haar-random, so
  $$\mathbb E\Vert P_\perp T\Vert^2/\Vert T\Vert^2=(1-\chi/d^2)^2.$$
  The same holds for $N=8$ binary when only the top cut is deficient, with $d^2\to d^4$.
- **Analogy (known).** Lazy training of overparameterized CP needs $\Omega(d^{l-1})$ components
  (Wang, Wu, Lee, Ma & Ge 2020, arXiv:2010.11356). This is the same obstruction: the tangent space must
  be everything.

Empirical: ranks 17, 45, 72, 81 at $\chi=1,3,6,9$ equal the formula. The linearized residual fraction
is $0.41\pm0.13$ at $\chi=3$ (predicted 0.444) and $0.124\pm0.08$ at $\chi=6$ (predicted 0.111).

**R3 (derived + empirical).** $\Theta\approx\bar\lambda I$ requires every factor $\Phi_c\Phi_c^\top$
and $\Psi_v\Psi_v^\top$ to be near-isotropic. That needs $\chi_e\gg D_e$ (Wishart: relative spread
$\approx2\sqrt{D_e/\chi_e}$), which is possible only for small trees. At a capped full bond
($\chi=d^2$) the bottom cores are square Gaussian matrices and $\Theta$ is badly conditioned.

Measured spectrum ($N=4$, $d=3$, 5 seeds):

| $\chi$ | $\lambda_{\min}/\bar\lambda$ | $\lambda_{\max}/\bar\lambda$ |
|---|---|---|
| 9 (capped) | $1.6\times10^{-4}$ | 8.8 |
| 16 | 0.015 | 6.1 |
| 32 | 0.10 | 4.2 |
| 64 | 0.24 | 2.9 |

The lazy fit converges in $\tau\approx(\bar\lambda/\lambda_{\min})\ln(1/\text{tol})$. That is impractical
at capped full bonds, so **lazy-regime runs should use $\chi\ge2d^2$ with `cap=False`.**

### 2.2 Laziness criterion and constants

Chizat, Oyallon & Bach (2019, eq. 1) define the criterion
$\kappa=\Vert h(w_0)-y^\star\Vert\,\Vert D^2h(w_0)\Vert/\Vert Dh(w_0)\Vert^2$. For a $q$-homogeneous
model with $h(w_0)=0$ it falls like (scale)$^{-q}$. For the centred TTN, $\kappa\propto x=\Vert T\Vert/\alpha$.
This much is known; what is new are the TTN constants.

**R4 (derived).** Expanding in $x$ around the lazy (min-norm) displacement $\delta\theta=J^+T$:
$$\delta_K=c_1x+O(x^2),\qquad \delta_\theta=c_\theta x+O(x^2),$$
with
$$c_1=\frac{\Vert D\Theta[J^+T]\Vert_F}{\Vert\Theta\Vert_F\Vert T\Vert}\Big|_{\alpha=1},\qquad c_\theta=\frac{\Vert J^+T\Vert}{\Vert\theta_0\Vert\Vert T\Vert}\Big|_{\alpha=1}.$$
Both are properties of the init (tree, $d$, $\chi$, seed). Compute them at init by linear response
(central difference of $\Theta$ along $J^+T$).

In the isotropic limit ($N=4$, $\chi\gg d^2$, $\eta=1$) the calculation closes. Write
$S_L=TW_0^\top+W_0T^\top$, $S_R=T^\top W_0+W_0^\top T$, $p=\chi\sigma^2/d^2$. Using
$BB^\top\approx(\sigma^2/\chi)I$, one gets
$\delta\Theta\approx(2p+\sigma^2/\chi)(I\otimes S_R+S_L\otimes I)/\bar\lambda$, and hence
$$c_1=\frac{2(2\chi^2+d^2)\sqrt{d^2+1}\,d^3}{\chi(2d^2+\chi^2)^2}\ \xrightarrow{\chi\to\infty}\ \frac{4d^3\sqrt{d^2+1}}{\chi^3},\qquad c_\theta=\frac{d^2}{\sqrt{2d^2+\chi^2}\sqrt{2\chi+1}}\ \xrightarrow{\chi\to\infty}\ \frac{d^2}{\sqrt2\,\chi^{3/2}} .$$

Empirical ($d=3$; linear response over 5 seeds vs the formula):

| $\chi$ | $c_1$ (lin. resp.) | $c_1$ (formula) | $c_\theta$ (lin. resp.) | $c_\theta$ (formula) |
|---|---|---|---|---|
| 9 (capped) | $2.7\pm2.0$ | 0.33 | $2.3\pm1.4$ | 0.21 |
| 16 | $0.146\pm0.044$ | 0.074 | $0.21\pm0.05$ | 0.095 |
| 32 | $0.0115\pm0.0008$ | 0.0101 | $0.046\pm0.007$ | 0.035 |
| 64 | $0.00139\pm0.0003$ | 0.00129 | $0.0145\pm0.0012$ | 0.0124 |

The formula is reached as $\chi$ grows. At small $\chi$, the small eigenvalues of $\Theta$ inflate $c_1$
by up to $8\times$, with about 100% seed-to-seed spread. Full GF at $\chi=16$, $x=0.041$ gives
$\delta_K/x=0.155$ and $0.092$ for the two seeds; linear response at the same inits gives 0.161 and 0.090.
The probe-set (64 inputs) $\delta_K$ is 0.7–1.3 times the tensor-space value.

**R5 (derived).** In the two-copy model $W=W(\theta_+)-W(\theta_-)$, the linearized displacements are
antisymmetric ($\delta\theta_\pm=\pm J^\top(2\Theta)^{-1}T$). Therefore
$\Theta_2(t)=\Theta(\theta_0+\delta)+\Theta(\theta_0-\delta)+O(\delta^2)$ has **no first-order change**:
$$\delta_K^{(2)}=c_2x^2+O(x^3),\qquad \delta_\theta^{(2)}=\tfrac12\,\delta_\theta^{(\rm centred)}+O(x^2).$$
Pitfall: compute the two-copy kernel on the trainable (masked) parameters, $J\leftarrow J\cdot{\rm mask}$.
The unmasked doubled TTN has off-diagonal-block directions that are never trained.

Empirical:

- Ratio of $\delta_\theta$ (two-copy / centred) = 0.4995 at $\chi=32$, $x=0.14$.
- $\delta_K/x^2$ is constant: $4.1,4.2,4.3,4.1\times10^{-3}$ at $\chi=32$ for $x=2.17,0.91,0.35,0.14$.
- At $\chi=16$, $c_2\to0.17$ and $0.056$ for the two seeds.

**Lazy boundary (derived + empirical).** $\sigma_{\rm lazy}=(\Vert T\Vert/x_c)^{1/K}$, where
$\delta_K(x_c)=0.1$; at first order $x_c=0.1/c_1$. GF measurements ($N=4$, $d=3$, teacher $(3,2,1)$,
2 seeds):

| model | $\chi$ | $x_c$ | $\sigma_{\rm lazy}$ |
|---|---|---|---|
| centred | 16 | 0.6–0.85 | 1.65–1.8 |
| centred | 32 | ≈2.6 | ≈1.13 |
| two-copy | 16 | 1.3–1.9 | 1.25–1.4 |
| two-copy | 32 | ≈4.9 | ≈0.91 |

Threshold sensitivity: at threshold 0.05 or 0.2, $x_c$ scales by $\times\frac12$ or $\times2$ for the
centred model (linear law) and by $\times\frac1{\sqrt2}$ or $\times\sqrt2$ for the two-copy model
(quadratic law). So $\sigma_{\rm lazy}$ shifts by $2^{\pm1/K}$ and $2^{\pm1/(2K)}$ respectively; the
two-copy boundary is twice as sharp in $\log\sigma$.

**R6 (derived).** If some bond is deficient, $W^\star+W_0$ generically has bond rank
$\operatorname{rank}W^\star_{(e)}+\chi_e>\chi_e$. The **centred model then cannot reach zero loss at
any $\sigma$.**

- **Centred, large $\sigma$.** The minimum tends to $\tfrac12\Vert P_\perp W^\star\Vert^2$
  (perturbation theory for the trailing singular values of $W_0+W^\star$), so
  $L_{\rm end}/L_0\approx\Vert P_\perp T\Vert^2/\Vert T\Vert^2\approx(1-\chi/d^2)^2$ ($N=4$).
- **Centred, small $\sigma$.** $L_{\rm end}\to0$, because $W_0$ is negligible.
- **Two-copy.** It has bond $2\chi$ and is realizable if $2\chi\ge$ teacher rank. At large $\sigma$ it
  (i) fits the tangent part lazily, (ii) **plateaus at $L/L_0=\Vert P_\perp T\Vert^2/\Vert T\Vert^2$**,
  then (iii) escapes by rotating the bond subspaces. The escape is a 2-core-type escape with rate
  $\gamma\approx s_1(P_A^\perp W^\star P_C^\perp)\,s_1(B)$, $s_1(B)\approx2\sigma/\sqrt\chi$
  (heuristic). The plateau lasts $\tau_{\rm esc}\propto\sigma^3\ln\sigma$ in kernel time ($N=4$). This
  is the TTN version of the mixed lazy-then-active dynamics of Tu, Aranguri & Jacot (2024,
  arXiv:2405.17580).

"Large $\sigma$ ⇒ lazy" therefore holds only with full bonds. **Warning for the regime map.** With
$N=8$, $d=3$ the top cut has maximal rank 81. For $9\le\chi<81$ the centred model at large $\sigma$ ends at
$L/L_0\approx(1-\chi/81)^2$ (0.49 at $\chi=24$). For $\chi<9$ the bottom bonds are deficient too and the
residual is larger. By the diagnostics this is not "lazy" (no convergence). With $N=8$, $d=2$, full
bonds need $\chi\ge16$.

## 3. (b) Saddle-to-saddle boundary

### 3.1 Aligned (decoupled) odeco dynamics: exact

**R7 (proven, given the aligned reduction).** Use `aligned_init` with amplitude $a_0$ on every core and
component. Each component then follows its own balanced scalar dynamics, with $u=a^K$:
$$\dot u=K\,u^{2-2/K}(s-u),\qquad u(0)=u_0=a_0^K .$$
With $v=u/s$ and
$$I_K(v_0,v_1)=\frac1K\int_{v_0}^{v_1}\frac{v^{-2+2/K}}{1-v}\,dv,$$
the event time (overlap $=\tfrac12$) and the 10–90% width are
$$t_r=s_r^{-(2-2/K)}\,I_K(u_0/s_r,\tfrac12),\qquad w_r=s_r^{-(2-2/K)}\,\omega_K,\quad \omega_K=I_K(0.1,0.9).$$
Values: $\omega_3=2.009$, $\omega_4=1.800$, $\omega_5=1.612$, $\omega_7=1.318$, $\omega_{10}=1.025$,
$\omega_{15}=0.743$, and $K\omega_K\to80/9+\ln81=13.28$.

For consecutive weights $s_i>s_{i+1}$ with $q_i=s_{i+1}/s_i$, the separation ratio of `diagnostics.md` is
$$\rho_i=\frac{I_K(u_0/s_{i+1},\tfrac12)-q_i^{\,2-2/K}\,I_K(u_0/s_i,\tfrac12)}{\omega_K}.$$
For small $u_0$ it becomes
$$\rho_i\simeq\Big(\frac{u_0}{s_{i+1}}\Big)^{-(1-2/K)}\frac{1-q_i}{(K-2)\,\omega_K}\quad\Rightarrow\quad u_{0c}\simeq s_{i+1}\Big[\frac{1-q_i}{3(K-2)\omega_K}\Big]^{K/(K-2)} .$$

**As $K\to\infty$:** $u_{0c}\to s_iq_i(1-q_i)/39.85$ stays finite, while the per-core boundary
$a_{0c}=u_{0c}^{1/K}=1+\ln(u_{0c})/K+O(K^{-2})\to1$. This is **H4 for aligned init, made
quantitative.** The rich window in $\log\sigma$ between the S2S and lazy boundaries narrows like $1/K$;
this is the TTN form of "depth sharpens the transition" (Woodworth et al. 2020).

Two components $(1,q)$: per-core boundary $a_{0c}$ (exact), with the exact function-space boundary
$u_{0c}$ and its asymptotic estimate in brackets:

| $K$ | $q=0.5$ | $q=0.8$ | $q=0.9$ |
|---|---|---|---|
| 3 | 0.061 ($2.2\times10^{-4}$; $2.9\times10^{-4}$) | 0.030 | 0.016 |
| 4 | 0.178 | 0.128 | 0.093 |
| 5 | 0.282 | 0.229 | 0.186 |
| 7 | 0.435 ($2.9\times10^{-3}$) | 0.387 | 0.342 |
| 11 | 0.607 | 0.572 | 0.535 |
| 15 | 0.700 ($4.8\times10^{-3}$) | 0.673 | 0.643 |
| 31 | 0.846 | 0.832 | 0.815 |
| 101 | 0.951 ($6.26\times10^{-3}$; limit $6.27\times10^{-3}$) | 0.947 | 0.941 |

Teacher $(3,2,1)$, binding separation in brackets:

| $K$ | $u_{0c}$ | $a_{0c}$ |
|---|---|---|
| 3 | $2.85\times10^{-4}$ ($\rho_1$) | 0.066 |
| 5 | $3.56\times10^{-3}$ ($\rho_2$) | 0.324 |
| 7 | $5.85\times10^{-3}$ | 0.480 |
| 11 | $8.28\times10^{-3}$ | 0.647 |
| 15 | $9.50\times10^{-3}$ | 0.733 |
| 31 | $1.13\times10^{-2}$ | 0.865 |

As $K\to\infty$, $u_{0c}\to0.0126$.

Empirical: aligned-init TTN runs ($N=4$, $d=3$, $\chi=3$) reproduce the formula to three decimals. At the
predicted $a_{0c}=0.0658$ they give $\rho=(2.999,\,3.526)$; at $a_0=0.03$, $(6.79,\,8.04)$.

### 3.2 Random fan-in init

- **Exact small-$\sigma$ scaling (derived).** The near-origin vector field is homogeneous of degree
  $K-1$, so $t_r=\sigma^{-(K-2)}T_r(\hat\theta_0)+O(1)$ and the widths are $O(1)$. Hence
  $\rho_i=R_i\,\sigma^{-(K-2)}+O(1)$ with seed-dependent $R_i$.
- **Effective-amplitude heuristic.** Random init behaves like aligned init with per-component amplitude
  $$u_0^{\rm eff}=c_K\,\sigma^K/\sqrt D .$$
  Here $\sigma^K/\sqrt D$ is the RMS overlap of $W_0$ with a unit rank-one tensor, and $c_K$ absorbs the
  alignment phase. This gives
  $$\sigma_c(K)=\big(\sqrt D\,u_{0c}(K)/c_K\big)^{1/K}.$$
- **Per-core form.** Equivalently, random init acts like aligned init with per-core amplitude
  $\kappa_K\sigma$, where $\kappa_K:=(c_K/\sqrt D)^{1/K}$. Then
  $$\sigma_c(K)=a_{0c}(K)/\kappa_K .$$
- **Calibration (empirical, $d=3$, $\chi=3$, teacher $(3,2,1)$, binary trees).**
  - $K=3$: $c_3=0.155$–$0.19$, stable over $\sigma\in[0.04,0.24]$ (the transition widths equal the
    aligned ones). So $\kappa_3=0.27$ and $\sigma_c=0.066/0.27=0.24$. Measured $\rho_1=3.07$ and $2.98$
    at $\sigma=0.24$.
  - $K=5$ ($N=5$, $D=243$): $c_5=0.4$–$0.58$. So $\kappa_5=0.50$ and $\sigma_c=0.32/0.50\approx0.65$.
    Measured $\rho_1=3.23$ at $\sigma=0.65$.
  - Prediction for $K=7$ ($N=8$): $\sigma_c\approx(0.54/c_7)^{1/7}$, i.e. $0.86$–$1.0$ for
    $c_7\in[0.5,1.5]$.
- **Limit (conjecture).** $\kappa_K$ increases with $K$ (0.27 to 0.50; the escape picks the best
  direction in every core, and that advantage compounds with depth). $\sigma_c\to1/\lim_K\kappa_K$, so
  H4's $\sigma_c\to1$ holds for random init iff $\kappa_K\to1$. Test this at $K=7$ and $15$. The robust
  statements are the exact $\sigma^{-(K-2)}$ scaling of $\rho$, $a_{0c}(K)\to1$, and the finite
  function-space limit of $u_{0c}$.
- **Merged events (empirical, important for classification).** For one seed out of three ($K=3$), the
  first two components are learned in **one merged event** at every $\sigma$ from 0.04 to 0.24:
  $\rho_1=0.02$–$0.25$, with the width of the $s=2$ event between 5 and 30. A mixed direction escapes
  first and then rotates (the H1b correction). Because the escape direction converges as
  $\sigma\to0$, the fraction of such seeds does not go to zero. The S2S label should be reported
  per seed, with this fraction, or events should be defined on bond spectra.
- **Plain vs centred model.** They coincide while $\sigma^K\ll\Vert W^\star\Vert$. For large $K$ and
  $\sigma\gtrsim1$, the plain model has $\Vert W_0\Vert\gg\Vert W^\star\Vert$ and must first remove
  $W_0$, so the S2S boundary should be mapped in the centred model, as H4 states.

## 4. (c) Large bond dimension $\chi\to\infty$

### 4.1 What is known

- **GP limit.** Infinite bond dimension gives a GP (Guo & Draper 2021). R1b gives the exact covariance
  at every $\chi$.
- **NTK, muP, polynomial networks.** No NTK or muP analysis of TTNs or MPS was found; the search was
  limited (see header). For polynomial networks, the NTK changes by $O(1/n)$ during training (Dyer &
  Gur-Ari 2020; proven for polynomial activations by Aitken & Gur-Ari 2020, arXiv:2006.06687;
  *[order unverified]*). This is consistent with R10's $\chi^{-1}$ under NTK parameterization.
- **Spectral condition** (Yang, Simon & Bernstein 2023, arXiv:2310.17813, verified). Feature learning
  needs $\Vert W_\ell\Vert_*\asymp\Vert\Delta W_\ell\Vert_*\asymp\sqrt{n_\ell/n_{\ell-1}}$. Their
  parameterization is $\sigma_\ell\asymp n_{\ell-1}^{-1/2}\min(1,\sqrt{n_\ell/n_{\ell-1}})$ and
  $\eta_\ell\asymp n_\ell/n_{\ell-1}$ for SGD. They note that the alignment argument for the init term
  fails when fan-out < fan-in.
- **Tensor Programs.** Tensor Programs (Yang & Hu 2021) does not cover TTNs. A binary core is an
  $n\times n^2$ map acting on products, so its dimensions are not proportional.

### 4.2 What is special for TTNs: the injective-norm spectral condition (derived)

A core's input is always a product vector $h_{c_1}\otimes h_{c_2}$. Its gradient
$\mathbb E[r\,b_v\otimes h_{c_1}\otimes h_{c_2}]$ is a sum of rank-one product tensors, where $r$ is
the residual and $b_v=\partial f/\partial h_v$. So the norm that controls $\Vert G_v(\Delta h\otimes h)\Vert$
and $\Vert\Delta G_v(h\otimes h)\Vert$ is the **injective norm**
$\Vert G\Vert_{\rm inj}=\sup_{\Vert a\Vert=\Vert b\Vert=1}\Vert G(a\otimes b)\Vert$, not the spectral norm
of the $\chi_v\times F_v$ matricization.

- **TTN spectral condition.** $\Vert G_v\Vert_{\rm inj}\asymp\Vert\Delta G_v\Vert_{\rm inj}\asymp\sqrt{\chi_v/F_v}$.
- **Gaussian core.** With entry std $t_v$, $\Vert G_v\Vert_{\rm inj}\asymp t_v\sqrt{\chi_v+\sum_cm_c}$,
  where $m_c$ are the child-leg dimensions (Gaussian tensors: Tomioka & Suzuki 2014 *[unverified]*).
  Hence $t_v\asymp\sqrt{\chi_v/F_v}\,/\sqrt{\chi_v+\sum_cm_c}$.
- **Relative to fan-in**, this is a multiplier $m_v=\sqrt{\chi_v/(\chi_v+\sum_cm_c)}$:
  - every non-root core: $m_v=\Theta(1)$, so **fan-in init already satisfies the condition**;
  - the root: $m_v=(\sum_cm_c)^{-1/2}$, i.e. $(2\chi)^{-1/2}$ for binary and $(d+\chi)^{-1/2}$ for MPS.
- **Learning rates.** The update condition gives $\eta_v\asymp\chi_v/F_v$, as in YSB.
- **Check.** Fan-in binary core with std $1/\chi$: $\Vert G\Vert_{\rm inj}\sqrt\chi=2.43,\,2.81,\,2.78,\,2.78$
  at $\chi=8,16,32,64$, while the matrix spectral norm is $1.21,\,1.23,\,1.16,\,1.12$.

**Applying YSB's matrix rule naively is wrong for TTNs.** It gives $m_v=\chi^{-1/2}$ for binary internal
cores and $1/\chi$ for the binary root. From R1b with $s_v^2=\sigma^2m_v^2$, the whole kernel at init then
vanishes (derived, confirmed in §7): for $N=8$ binary,
$\mathbb E\Theta=3\sigma^{12}/(\chi^2D)\,I$. With the injective rule, $\mathbb E\Theta=4\sigma^{12}/D\,I$,
which is $O(1)$. The network therefore starts closer to a saddle under the matrix rule. Empirically
it fits $2.5$–$8.6\times$ more slowly than muP-TTN at $\chi=4$–$16$. Whether this slowdown keeps growing
with $\chi$ is not established ($t_{1/2}=12.3$ and $12.5$ at $\chi=8$ and $16$).

### 4.3 Parameterizations (gradient-flow form)

Init $G_v=\sigma m_vZ/\sqrt{F_v}$, rates $\eta_v$; rescale time by $\bar\lambda_\eta=\operatorname{tr}\Theta_\eta(0)/D$.
Each cell gives $(m_v,\ \eta_v)$:

| core type | $F_v$ | $\chi_v$ | SP | NTK | **muP-TTN** | YSB matrix rule (avoid) |
|---|---|---|---|---|---|---|
| bottom (binary, 2 phys) | $d^2$ | $\chi$ | $1,\ 1$ | $1,\ 1/d^2$ | $1,\ \chi/d^2$ | $1,\ \chi/d^2$ |
| internal binary | $\chi^2$ | $\chi$ | $1,\ 1$ | $1,\ 1/\chi^2$ | $1,\ 1/\chi$ | $\chi^{-1/2},\ 1/\chi$ |
| binary root | $\chi^2$ | 1 | $1,\ 1$ | $1,\ 1/\chi^2$ | $(2\chi)^{-1/2},\ 1/\chi^2$ | $1/\chi,\ 1/\chi^2$ |
| MPS first / HT leaf | $d$ | $\chi$ | $1,\ 1$ | $1,\ 1/d$ | $1,\ \chi/d$ | $1,\ \chi/d$ |
| MPS middle | $d\chi$ | $\chi$ | $1,\ 1$ | $1,\ 1/(d\chi)$ | $1,\ 1/d$ | $d^{-1/2},\ 1/d$ |
| MPS root | $d\chi$ | 1 | $1,\ 1$ | $1,\ 1/(d\chi)$ | $(d+\chi)^{-1/2},\ 1/(d\chi)$ | $(d\chi)^{-1/2},\ 1/(d\chi)$ |

Kernel shares at init (R1b; the share of core $v$ is $\propto\eta_vF_v/s_v^2$):

| parameterization | predicted shares | measured |
|---|---|---|
| SP | $\propto F_v$, root-dominated: $\chi^2/(2d^2+\chi^2)$ for $N=4$ | — |
| NTK | $1/K$ each | ≈0.33 each ($N=4$) |
| muP-TTN, $N=4$ | $(\tfrac14,\tfrac14,\tfrac12)$ | $(0.25,0.25,0.50)$ |
| muP-TTN, $N=8$ | bottoms $\tfrac18$ each, internals $\tfrac18$ each, root $\tfrac14$ | — |

Under muP-TTN, $f_0\sim\chi^{-1/2}\to0$ and every core's update is $O(1)$ from the start.

### 4.4 Predicted $\chi$-scaling

$N=4$ binary, centred model, end of training, fixed $\sigma$, $x=\Vert W^\star\Vert/\sigma^3$. Derived in
the isotropic limit; the exponents are the robust part.

- **SP.** Only the $\chi^2$-fan-in cores move (random-feature regression on
  $\Phi_{c_1}\otimes\Phi_{c_2}$).
  - $\delta_K\to c_1x$ with $c_1\approx4d^3\sqrt{d^2+1}/\chi^3$, so $\delta_K\propto\chi^{-3}$.
  - $\delta_\theta\approx xd^2/(\sqrt2\chi^{3/2})$.
  - Per-core Frobenius change: root $\approx xd^2/\chi$, bottom cores $\approx xd^3/\chi^{5/2}$.
- **NTK.** Lazy as $\chi\to\infty$.
  - $\delta_K\approx\tfrac23d\sqrt{d^2+1}\,x/\chi$ at first order, so $\propto\chi^{-1}$ (matching the
    $O(1/n)$ NTK change of wide polynomial nets).
  - $\delta_\theta\approx xd/(3\sqrt\chi)$.
  - Per-core: bottom $xd/(3\sqrt\chi)$, root $xd^2/(3\chi)$.
- **muP-TTN.**
  - $\delta_K$, $\delta_\theta$ and the bottom-core changes tend to $\chi$-independent nonzero limits
    (feature learning).
  - The root's Frobenius-relative change decays like $\chi^{-1/2}$, because $\Delta B$ has low rank. Its
    spectral (= injective) relative change is $\Theta(1)$.
- **Conjecture.** The muP-TTN limit is a two-population mean field with a pairwise (graphon-like)
  root coupling:
  $$W=\iint b(a,c)\,a\,c^\top\,d\mu_A(a)\,d\mu_C(c)$$
  for $N=4$, with nested versions for deeper trees.

Empirical (σ = 1.5, $d=3$, $x=1.11$, 2 seeds per $\chi$, $\chi=9$–$128$, `cap=False`; table in §7) confirms all
three: SP $\delta_K$ falls by $\times8$–$9$ per doubling at $\chi=64\to128$, NTK $\delta_K\approx(0.13,0.11)$ at
$\chi=128$ approaching $\chi^{-1}$, NTK per-core changes match the formulas to within 5% at $\chi=128$,
muP $\delta_K\to6.3$, $\delta_\theta\to0.57$. For $N=8$ (internal cores; §7 items 10–11), muP-TTN's init
kernel scale and time to halve the loss are $\chi$-independent for $\chi=4$–$16$, while the matrix
rule's init kernel falls like $\chi^{-2}$.

## 5. (d) Imbalance (root scale $\lambda\sigma$, other cores $\sigma$, fixed $\alpha=\lambda\sigma^K$)

**Conserved quantities (derived, exact in expectation).** $Q_v=G_v^\top G_v/\eta_v-G_{p,(v)}G_{p,(v)}^\top/\eta_p$.
With fan-in init,
$$\mathbb E[G_v^\top G_v]=s_v^2I,\qquad \mathbb E[G_{p,(v)}G_{p,(v)}^\top]=s_p^2(\chi_p/\chi_v)\,I,$$
so $\mathbb E Q_v=(s_v^2-s_p^2\chi_p/\chi_v)\,I$ for $\eta=1$.

- Internal edges ($\chi_p=\chi_v$, equal $s$) are balanced in expectation.
- Root edges ($\chi_p=1$) have $\mathbb EQ_v=\sigma^2(1-\lambda^2/\chi)\,I$, balanced at
  $\lambda_{\rm bal}=\sqrt\chi$. The default fan-in init is therefore "root-light" by $\sqrt\chi$.
- Kunin et al. (NeurIPS 2024, arXiv:2406.06158, verified) use $\delta=\eta_wa^2-\eta_a\Vert w\Vert^2$
  ($a$ downstream). This corresponds to $-\operatorname{tr}Q$ at the root edges: $\lambda>\sqrt\chi$ is
  their "upstream" init ($\delta>0$) and $\lambda<\sqrt\chi$ their "downstream" init.
- Check: $\operatorname{tr}Q/\chi$ changes sign between $\lambda=3$ (+0.43, predicted +0.47) and
  $\lambda=10$ (−2.70, predicted −2.55) at $\chi=16$.

**Aligned single mode, exact (derived).** All non-root amplitudes are equal to $x$, the root amplitude
is $b$, and $x^2-b^2=\Delta=a_0^2(1-\lambda^2)$ is conserved. The dynamics is
$$\dot b=(s-bx^{K-1})x^{K-1},\qquad \dot x=(s-bx^{K-1})\,b\,x^{K-2} .$$
Three regimes, for the centred model with target $s+\alpha$:

- **Small root, $\lambda\ll1$.** $x\approx a_0$ stays frozen and $b$ relaxes linearly. Training is lazy iff
  $b_\infty=(s+\alpha)/a_0^{K-1}\ll a_0$, i.e. $s\ll\alpha/\lambda$. The kernel change is
  $$\delta_K\approx2(K-1)\lambda^2s(s+2\alpha)/\alpha^2\ \propto\lambda^2 .$$
  In the plain model at small $\alpha$ there is **no plateau** once $\lambda<\alpha/s$.
- **Default, $\lambda=1$.** $\delta_K=(1+s/\alpha)^{2(K-1)/K}-1$.
- **Large root, $\lambda\gg1$.** $b$ is frozen and the other $K-1$ cores form a degree-$(K-1)$ model with
  the same $\alpha$, so
  $$\delta_K\approx(1+s/\alpha)^{2(K-2)/(K-1)}-1,$$
  which is not lazy. The plain-model plateau becomes $t\approx\alpha^{-(1-2/K)}\lambda^{-2/K}/((K-3)s)$
  for $K\ge4$, and an exponential (log-plateau) escape for $K=3$.

This differs from Kunin's two-layer linear result ("upstream ⇒ lazy") because a frozen large root still
leaves a nonlinear, degree $\ge2$ model when $K\ge3$.

**Random init, $N=4$, isotropic first order (derived).**
$$\delta_K(\lambda)\approx\frac{2\Vert T\Vert\lambda^2\sqrt{d^2+1}\,d^5(\lambda^2/\chi+2\chi/d^2)}{\alpha(2\lambda^2d^2+\chi^2)^2}$$
This is $\propto\lambda^2$ for $\lambda\ll1$ and saturates at $\Vert T\Vert d\sqrt{d^2+1}/(2\chi\alpha)$ for
$\lambda\gg\chi/d$. The root-kernel share is $\chi^2/(\chi^2+2d^2\lambda^2)$.

**Heuristic for general trees.** Chizat's $\kappa\approx\Vert T\Vert\min(1,\lambda)/\alpha$ at fixed
$\alpha$. In words: when the root is small, laziness is set by the scale of the non-root cores,
$\sigma^K=\alpha/\lambda$, not by $\alpha$.

Empirical ($N=4$, $d=3$, $\chi=16$, $\alpha=3.375$, seed 0):

| $\lambda$ | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | 30 |
|---|---|---|---|---|---|---|---|
| $\delta_K$ | 0.0014 | 0.012 | 0.053 | 0.196 | 0.48 | 0.55 | 0.51 |

The local slope approaches 2 as $\lambda\to0$ (1.8 between 0.03 and 0.1). At $\lambda=0.03$ the root moves
by 193% of its init while the kernel moves by 0.14%: a lazy kernel with a far-moving readout. The
isotropic formula captures the trend but underestimates $\delta_K$ at small $\lambda$ by about
$10\times$, because the root's relative change is not small there.

## 6. Testable predictions (for `regime-map`)

**Lazy regime and kernel (§2)**

- **P1. Kernel scale.** $\operatorname{tr}\Theta(0)/D=\sigma^{2(K-1)}\sum_vF_v/D$ in expectation, for
  every tree and $\chi$. Rescale time by it. Kernel shares are $\propto\eta_vF_v/s_v^2$.
- **P2. Rank.** $\operatorname{rank}\Theta(0)=\sum_vF_v\chi_v-\sum_e\chi_e^2$, or $D$ iff all bonds are
  full.
- **P3. Lazy residual (centred, large $\sigma$).** With deficient bonds, $L_{\rm end}/L_0\approx(1-\chi/d^2)^2$
  for $N=4$ and $(1-\chi/d^4)^2$ for $N=8$ when only the top cut is deficient. No lazy regime exists.
- **P4. Centred, full and overparameterized bonds.** $\delta_K=c_1x$ for $x\lesssim0.5$, with $c_1$
  from linear response at init, or from the closed form when $\chi\gg d^2$. Lazy boundary
  $\sigma_{\rm lazy}=(10c_1\Vert W^\star\Vert)^{1/K}$ at first order; measured values in the R4 table.
- **P5. Two-copy.** $\delta_K\propto x^2$ (log-log slope 2, versus 1 centred), and
  $\delta_\theta^{(2)}/\delta_\theta^{(c)}=1/2$. The boundary is sharper by a factor 2 in $\log\sigma$.
- **P6. Two-copy with deficient bonds at large $\sigma$.** Lazy plateau at $L/L_0=\Vert P_\perp T\Vert^2/\Vert T\Vert^2$
  (computable at init), then escape. The plateau length in kernel time grows like $\sigma^3\ln\sigma$
  ($N=4$), and $\delta_K(\text{end})=O(1)$.
- **P7. Conditioning.** At capped full bonds the lazy fit needs $\tau\sim10^4$ (in units $1/\bar\lambda$),
  and $c_1$ varies by about 100% across seeds. Use $\chi\ge2d^2$ uncapped.

**S2S boundary (§3)**

- **P8. Aligned odeco init.** $\rho_i$ is given by the exact formula of §3.1. The boundary
  $a_{0c}(K,q)$ follows the table and $\to1$. In function space $u_{0c}\to s_iq_i(1-q_i)/39.85$.
- **P9. Random fan-in init.** $\rho_i\propto\sigma^{-(K-2)}$ exactly as $\sigma\to0$, and
  $\sigma_c(K)=a_{0c}(K)/\kappa_K$. For $d=3$, $\chi=3$, teacher $(3,2,1)$: $\kappa_3=0.27$ and
  $\kappa_5=0.50$ give $\sigma_c\approx0.24$ ($K=3$) and $0.65$ ($K=5$); predicted 0.86–1.0 for $K=7$.
  H4's $\sigma_c\to1$ holds iff $\kappa_K\to1$.
- **P10. Merged first events.** A $\sigma$-independent fraction of seeds (1 of 3 here) has a merged first
  event ($\rho_1<1$) at all small $\sigma$.

**Large bond dimension (§4)**

- **P11. Exponents** ($N=4$, centred, fixed $\sigma$):

  | | $\delta_K$ | $\delta_\theta$ | root (Frobenius) | bottom cores |
  |---|---|---|---|---|
  | SP | $\chi^{-3}$ | $\chi^{-3/2}$ | $\chi^{-1}$ | $\chi^{-5/2}$ |
  | NTK | $\chi^{-1}$ | $\chi^{-1/2}$ | $\chi^{-1}$ | $\chi^{-1/2}$ |
  | muP-TTN | $\chi^0$ | $\chi^0$ | $\chi^{-1/2}$ (Frobenius), $\chi^0$ (spectral) | $\chi^0$ |

  For deeper binary trees, the same exponents should hold for SP and NTK (conjecture).
- **P12. Deep trees.** With the YSB matrix rule on internal cores, the init kernel vanishes like
  $\chi^{-2}$ ($N=8$) and the fit is slower than under muP-TTN; whether the gap grows with $\chi$ is
  open. Under muP-TTN the init kernel and the time to halve the loss are $\chi$-independent.

**Imbalance (§5)**

- **P13.** At fixed $\alpha$, $\delta_K\propto\lambda^2$ as $\lambda\to0$ (lazy). $\delta_K$ saturates
  for $\lambda\gtrsim\chi/d$, where the root is frozen: the effective degree is $K-1$ and training is not
  lazy. $\operatorname{tr}Q$ at the root edges changes sign at $\lambda=\sqrt\chi$.

## 7. Numerical checks

Scripts are in the scratchpad (`theory/regimes/`), not in the repo. All runs: exact population loss,
float64, `solve_ivp` (RK45/DOP853, rtol $10^{-8}$–$10^{-10}$), time rescaled by
$\operatorname{tr}\Theta(0)/D$, stopping at $L/L_0=10^{-6}$–$10^{-7}$. Teacher $(3,2,1)$ odeco with $d=3$
unless stated otherwise.

1. **Kernel at init** (`init_kernel.py`, $N=4$, $\chi=1$–$64$, 5 seeds). Mean eigenvalue, rank,
   conditioning, $c_1$ and $c_\theta$: numbers in R1–R4. Uncapped $\chi=16/32/64$:
   $c_1=0.146/0.0115/0.00139$ (formula $0.074/0.0101/0.00129$).
2. **Centred GF** (`gf_lazy.py`, $\chi=16,32$, $\sigma=1.2$–$4.5$, 2 seeds). At $\chi=32$:
   $\delta_K=0.065,0.013,0.0038,0.0015$ at $x=2.17,0.91,0.35,0.14$. First-order prediction at
   $x=0.14$: 0.0016. $\delta_\theta=0.0055$.
3. **Two-copy GF** (same settings). At $\chi=32$: $\delta_K=0.0192,0.0035,0.00053,0.000082$, so
   $\delta_K/x^2=0.0041$–$0.0043$. $\delta_\theta=0.002732$, half the centred value (ratio 0.4995).
4. **S2S separations.**
   - Aligned init ($K=3$): matches the exact formula.
   - Random init ($K=3$, `s2s_random.py`, $\sigma=0.04$–$0.24$, 3 seeds): $\rho_1\sigma\approx0.76$
     (clean seeds), $\rho_1=3$ at $\sigma=0.24$. One seed is merged at every $\sigma$
     ($\rho_1=0.02$–$0.25$).
   - $K=5$ ($N=5$): $\rho_1=0.74/0.89$ ($\sigma=1$), $1.58/1.44$ ($\sigma=0.8$), $3.23$ ($\sigma=0.65$).
   - $K=7$ ($N=8$), one run at $\sigma=0.6$: non-greedy order ($s=2$ learned first, $t=25.9$, then
     $s=3$ at $77.1$, $\rho_1=54$). The $s=1$ component was not learned by $t=843$, so $K=7$ is deep
     in S2S at $\sigma=0.6$.
5. **Parameterizations** (`param_chi.py`, $N=4$, $\sigma=1.5$, $\chi=9$–$128$ uncapped, 2 seeds each):

   | | $\chi$ | 16 | 32 | 64 | 128 |
   |---|---|---|---|---|---|
   | SP | $\delta_K$ | 0.20/0.15 | 0.019/0.025 | 0.0044/0.0029 | 0.00044/0.00035 |
   | SP | $\delta_\theta$ | 0.17/0.13 | 0.043 | 0.0153 | 0.0052/0.0057 |
   | SP | root change | 0.72/0.63 | 0.36 | 0.17/0.18 | 0.082/0.091 |
   | NTK | $\delta_K$ | 0.57 | 0.36/0.40 | 0.28/0.23 | 0.14/0.11 |
   | NTK | $\delta_\theta$ | 0.30/0.25 | 0.19/0.18 | 0.13 | 0.096/0.107 |
   | muP | $\delta_K$ | 4.9/4.5 | 5.6/5.5 | 6.2/6.1 | 6.3/6.3 |
   | muP | $\delta_\theta$ | 0.67/0.59 | 0.59/0.55 | 0.57/0.57 | 0.56/0.59 |
   | muP | root change | 0.64/0.67 | 0.53/0.50 | 0.32/0.36 | 0.24/0.25 |

   - SP at $\chi=128$: formula gives $\delta_\theta=0.0049$ and root change 0.078.
   - NTK at $\chi=128$: bottom-core changes 0.097/0.102 (formula 0.098), root 0.024/0.027 (formula
     0.026).
   - muP bottom-core changes stay at 0.55–0.6.
6. **Injective norm** (`injective.py`): see §4.2.
7. **Imbalance** (`imbalance.py`): see §5.
8. **Deficient bond at large $\sigma$** (`lazy_deficient.py`, $N=4$, $\chi=3$ capped, seed 0). The
   linearized residual at init is $\Vert P_\perp T\Vert^2/\Vert T\Vert^2=0.535$.
   - Centred, $\sigma=4$: $L/L_0=0.52,\,0.506,\,0.504,\,0.504$ at $\tau=37,\,105,\,295,\,2320$. It
     stalls at 0.504, the unrealizable centred problem's minimum (R6), with $\delta_K=0.16$.
   - Centred, $\sigma=8$: stalls at $L/L_0=0.5339$ (predicted 0.5346 as $\sigma\to\infty$), with
     $\delta_K=0.016$.
   - Two-copy, $\sigma=4$: follows the centred curve to 0.49 ($\tau=37$), then escapes: 0.40, 0.19,
     0.049, 0.032 at $\tau=105,\,295,\,827,\,2320$, with $\delta_K=0.49$ (rich). At this $\sigma$ the
     escape overlaps the slow lazy phase.
   - Two-copy, $\sigma=8$: **plateau at the lazy residual**: 0.54, 0.533, 0.529, 0.517 at
     $\tau=37,\,105,\,295,\,827$. The escape starts after that (0.484 at $\tau=2320$; $\delta_K=0.044$
     so far). The escape is delayed by more than $8\times$ in $\tau$ from $\sigma=4$ to $8$, consistent
     with $\tau_{\rm esc}\propto\sigma^3\ln\sigma$ (predicted $\times12$).
9. **$K=5$ at $\sigma=0.5$**: $\rho_1=7.76$ and $6.39$, consistent with $\rho\propto\sigma^{-3}$ from
   $\sigma=0.65$.
10. **Init kernel with internal cores** (`init_kernel_params.py`, $N=8$ binary, $d=2$, $\sigma=1$,
    2 seeds). Mean eigenvalue $\operatorname{tr}\Theta_\eta/D$:

    | | $\chi=4$ | $\chi=8$ | $\chi=16$ | predicted |
    |---|---|---|---|---|
    | NTK | 0.028/0.025 | 0.026/0.039 | 0.042/0.032 | 0.0273 |
    | muP-TTN | 0.0150/0.0151 | 0.0147/0.0237 | 0.024/0.018 | 0.0156 |
    | YSB matrix rule | $9.3\times10^{-4}$ | $2.9\times10^{-4}$ | $6.9\times10^{-5}$ | $9.8\times10^{-4}$, $2.1\times10^{-4}$, $5.0\times10^{-5}$ ($\to3/(\chi^2D)$) |

    muP shares by core type (bottom/internal/root): 0.36–0.50 / 0.24–0.27 / 0.24–0.37, predicted
    0.5 / 0.25 / 0.25. Seed spread is about 30% because the top cut ($D_e=16$) is not strongly
    overparameterized at these $\chi$.
11. **$N=8$ GF, plain model** (`param_plain.py`, $d=2$, teacher $(3,2)$, $\sigma=1$, 1 seed, absolute
    time):

    | | $\chi$ | $t(L=L_0/2)$ | $t(L=10^{-4}L_0)$ | $\delta_K$ | $\delta_\theta$ |
    |---|---|---|---|---|---|
    | muP-TTN | 4 / 8 / 16 | 2.04 / 2.17 / 1.45 | 28 / 53 / 51 | 36 / 47 / 50 | 0.78 / 0.63 / 0.49 |
    | YSB matrix rule | 4 / 8 / 16 | 4.99 / 12.3 / 12.5 | 49 / 36 / 43 | 510 / 2700 / 12700 | 0.96 / 0.76 / 0.59 |

    Both are rich. Under muP-TTN the early dynamics is $\chi$-independent; under the matrix rule it
    starts from a smaller kernel.

## 8. Notes for the experiment agents

- Lazy-regime runs need full, and preferably overparameterized, bonds. Check that $\chi_e\ge\min(D_e,D/D_e)$
  on every edge, otherwise the centred model is unrealizable (R6). It then never reaches a stop
  criterion such as $L<10^{-6}L_0$: this stalled my first $N=8$, $\chi=4$ muP run. For $\chi$-sweeps on
  $N=8$, use the plain model, or $\chi\ge16$ ($d=2$) / $\chi\ge81$ ($d=3$).
- Report $\delta_K$ for the $\eta$-weighted kernel when per-core learning rates are used. For the
  two-copy model, mask the Jacobian.
- The probe kernel with 64–128 inputs fluctuates around the tensor-space $\Theta$; the probe $\delta_K$
  was 0.7–1.3 times the tensor-space value. For $N\le8$, $d\le3$, compute the tensor-space $\Theta$
  directly with `jacrev` of `to_dense`.
- For large $D$, $\delta_K$ in Frobenius norm may be diluted, because the kernel change concentrates on
  directions related to $T$. Also report a directional measure such as
  $|y^\top(K_t-K_0)y|/(y^\top K_0y)$ (conjecture: needed for $N\ge8$).
- Separation ratios: classify per seed and report the fraction with merged first events (P10).
