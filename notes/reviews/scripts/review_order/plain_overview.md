# Plain-language frame (for the final answer)
- TTN: an N-index tensor W written as a contraction of small "core" tensors on a tree; each tree edge is a
  shared summed index (bond) of dimension chi. MPS (tensor train) = chain, one core per input index;
  binary tree N=4 = two leaf cores (2 inputs each) joined by a root core = W = A R B^T, i.e. a 3-layer
  linear network / matrix factorization.
- Teacher W* = s1 u1(x)...(x)u1 + s2 u2(x)...(x)u2 (two rank-one "components", orthogonal), rho = s2/s1.
- Gradient flow from a small random init (per-core scale sigma) learns components one at a time with long
  plateaus (saddle-to-saddle). Q4: which comes first?
- m = number of cores whose input indices differ between the two components.
  m >= 3: random, decided by the init like a race of independent clocks; P(larger first) does not depend on sigma.
  m = 2: matrix-like; larger first as sigma -> 0, P(wrong) ~ sigma^((1-rho)/rho) (slow).
- H1b: wrong order corrected through a strict saddle (negative-curvature direction) only for m = 2
  [reviewer: only for non-adjacent cores]; m >= 3 keeps the wrong order.
