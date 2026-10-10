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
| 2026-10-09 18:25 UTC (start) | optim-samples | 32 vCPU (Netherlands) | 0.37 | DELETED 2026-10-10 (raw outputs in data/optim_samples, verified) | | $2.32 |
| 2026-10-09 19:10 UTC (start) | optimizers | 96 vCPU Zen 4 (US) | 1.40 | DELETED 2026-10-10 (raw outputs in data/optimizers, verified) | | $3.68 |
| 2026-10-09 21:45 UTC (start) | real-data | 128 vCPU EPYC (British Columbia) | 0.52 | running | | |

Burn rate with all seven running: $5.21/h (~$125/day; ~57 h of all seven within the $300 cap). Running total: see `crab-student list` (field `spent`); updated at each check-in.

Checkpoint 2026-10-10 ~00:30 UTC: listed agents $21.82 + deleted optimizers $3.68 = **$25.50** spent.

Checkpoint 2026-10-10 ~10:05 UTC: about **$30** spent (estimate: ~$29 at 08:52 UTC plus the current sessions).
`crab-student list` reports `spent` per machine session (it resets on restart), so session totals are now
recorded here at each stop or delete. Sessions that ended: locality $2.19 (6.25 h), regime-map $6.45 (6.23 h),
real-data $1.69 (3.25 h); all three stopped themselves and cannot restart yet (stopped disks ~$1.6/day together).
Running fix rounds: s2s-core ($0.74/h), order ($0.84/h), data-teacher ($0.47/h).

Sessions ended 2026-10-10 midday: s2s-core fix round $0.43 (0.58 h; DELETED after verified copy), real-data restart
$0.16 (0.32 h; DELETED after verified copy). regime-map restarted (queued start succeeded) at $1.04/h for the raw copy.
Running: order ($0.84/h, fix round), data-teacher ($0.47/h, errata pass). Estimated total so far: about $32.

2026-10-10 afternoon: data-teacher DELETED after verified copy (errata session $0.47, 1.02 h); regime-map restarted for
round 2 at $1.04/h (the earlier copy session cost < $0.5). Running: order ($0.84/h), regime-map ($1.04/h).
Estimated total so far: about $34.

2026-10-10 ~14:30 UTC: order DELETED after verified copy (fix-round session $2.16, 2.59 h). regime-map round 2 running
($1.04/h). locality still stopped (disk only, ~$0.71/day; cannot restart; raw copy verified earlier, re-check pending).
Estimated total so far: about $36.
