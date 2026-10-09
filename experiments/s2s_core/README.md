# s2s_core: saddle-to-saddle phenomenology and escape-time scaling (Q1, Q2)

Agent `s2s-core`, brief `notes/briefs/s2s-core.md`. Findings and derivations: `notes/s2s_core.md`.

## Question

Does gradient flow on a TTN from small initialization show saddle-to-saddle (S2S) dynamics, which
saddles does it visit, and how do plateau durations scale with the init scale sigma, the tree (K
cores) and the bond dimension chi, against theory?

## Setup (common to all parts)

- Model as in `notes/model.md`: whitened Gaussian inputs, exact population loss, d = 4 at every
  site, odeco teachers (`ttn.teachers.odeco_teacher`, teacher seed 100), weights
  "separated" [3, 2, 1] and "close" [1, 0.85, 0.7].
- Trees: binary N = 4 (K = 3), binary N = 8 (K = 7), MPS N = 4 (K = 4), MPS N = 8 (K = 8),
  HT N = 4 (K = 7). Bond dimension chi = 3 (= number of components) and chi = 6 (bonds capped at
  the largest rank their bipartition allows, `TTNSpec.uniform(cap=True)`).
- Fan-in init with per-core scale sigma; seeds 0..9.
- Dynamics: gradient flow (details per part below, with tolerances).

## Parts

(in progress)

## How to run

(in progress)

## Raw outputs

Raw trajectories go to `/root/outputs/s2s_core/` (not in git).
