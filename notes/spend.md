# Vast.ai spend log

Budget: $300 total (set by the user, 2026-10-09). Target: <= $200 for two rounds of agents, keeping
>= $100 in reserve for follow-ups and reproductions.

| date | agent | machine | $/h | hours | cost | notes |
|---|---|---|---|---|---|---|
| 2026-10-09 18:25 UTC (start) | s2s-core | 64 vCPU (Indiana) | 0.74 | running | | |
| 2026-10-09 18:25 UTC (start) | locality | 48 vCPU (Norway) | 0.35 | running | | |
| 2026-10-09 18:25 UTC (start) | order | 96 vCPU (Nevada) | 0.84 | running | | |
| 2026-10-09 18:25 UTC (start) | regime-map | 384 vCPU (British Columbia) | 1.04 | running | | |
| 2026-10-09 18:25 UTC (start) | data-teacher | 32 vCPU (Quebec) | 0.47 | running | | |
| 2026-10-09 18:25 UTC (start) | optim-samples | 32 vCPU (Netherlands) | 0.37 | running | | |
| 2026-10-09 19:10 UTC (start) | optimizers | 96 vCPU Zen 4 (US) | 1.40 | done (fix round pending) | | |
| 2026-10-09 21:45 UTC (start) | real-data | 128 vCPU EPYC (British Columbia) | 0.52 | running | | |

Burn rate with all seven running: $5.21/h (~$125/day; ~57 h of all seven within the $300 cap). Running total: see `crab-student list` (field `spent`); updated at each check-in.
