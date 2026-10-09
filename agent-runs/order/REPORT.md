# order: which teacher component a TTN learns first, and how random the order is (Q4, H1, H1b)

Status: working - Branch: agent/order - Updated: 2026-10-09 18:18 UTC

## Summary
(in progress)

## Results
(in progress)

## What was done
1. Read brief, repo, model, diagnostics, plan; merged origin/main (report skeleton).

## Decisions
(in progress)

## Files left on the machine
- `/root/outputs/order/`: raw outputs (to be filled).

## Caveats and next steps
(in progress)

## Log
- 18:18 Started; read brief and repo; merged origin/main.
- 18:23 Derived theory (m-count rule, MPS exact race, m=2 wrong-order exponent, H1b Hessian) in notes/order.md; pilots: GF to first event costs 1-9 s/run; leading-order race on MPS gives P(greedy)~0.73-0.87 at rho=0.5
- 18:48 Per-seed pilot: full GF first event agrees 100% with the leading-order flow (MPS4, binary8; binary4 when the LO flow is stopped at radius 1/sigma). Independent-race approx: 100% MPS4, 98% binary8.
- 18:48 binary4 sigma-scan (1000 seeds x 5 rho x 7 sigma): P(wrong order) ~ sigma^gamma with gamma = 0.41, 0.24, 0.10, 0.04 for rho = 0.7, 0.8, 0.9, 0.95 vs predicted (1-rho)/rho = 0.43, 0.25, 0.11, 0.05.
- 18:48 Merged origin/main (literature v1, core fixes 57d5262). My pilots used basis='standard' (no random teacher draw), so the teacher/init stream bug did not affect them; final runs will use the fixed library anyway.
- 18:52 mps4 sigma-scan (400 seeds, sigma 0.003-0.3): P(greedy) identical for every sigma <= 0.1 (0.815/0.605/0.510 at rho 0.5/0.8/0.95): sigma-independent as derived for m >= 3.
- 18:52 Merged draft theory notes (theory_saddles.md agrees with my m-rule and H1b derivation). Launched main sweep (configs/main.yaml: 28 settings x 400 seeds x 6 rho, sigma=0.01, PID 7480, ETA ~75 min).
