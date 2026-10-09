# real-data: do the teacher-student TTN phenomena survive on real images? (B5)

Status: working - Branch: agent/real-data - Updated: 2026-10-09 21:19 UTC

## Summary
In progress. Task: train TTNs (quadtree binary, snake MPS, scrambled binary) by gradient flow on
downsampled MNIST with cos/sin site features, and test plateaus, stepwise bond growth, the locality
rule and the regime map.

## Results
(pending)

## What was done
1. Reverted the launch commit b622158 (in-progress library edits), `uv sync`, pushed (98042d5).
2. Data download + downsampling script `experiments/real_data/download.py` (MNIST, Fashion-MNIST; 4x4 and 8x8).

## Decisions
(see below as they are made)

## Files left on the machine
- `/root/outputs/real_data/data/` (~25 MB): raw IDX files and downsampled npz (regenerate with download.py).

## Caveats and next steps
(pending)

## Log
- 21:19 reverted launch commit, read brief/notes/library, downloaded data, launched cost pilot
