---
name: keep-agents-busy
description: "In research-manager mode, maximize parallelism - always keep at least 4-8 agents (local and/or remote) doing useful work until the project is complete"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ccfcd9a3-98b9-48c2-8218-49a3ab02b926
  modified: 2026-10-09T17:59:24.458Z
---

Always keep at least 4-8 agents doing useful work in parallel until the project looks complete. Local agents are fine for experiments too (not only remote Vast.ai agents); use both. Don't let the pipeline drain while waiting on one result: queue the next wave (theory, experiments, reproductions, report writing, reviews) before the current wave finishes.

**Why:** User said on 2026-10-09 (TTN regimes project): "It's fine to use local agents for this too. But you should make the most out of parallelization and should always have at least 4-8 agents doing useful work until the project is looking complete."

**How to apply:** At every check-in, count running agents; if fewer than 4 are doing useful work, launch more from the plan's backlog (independent questions, variations, reproductions, theory, report sections). Prefer independent tasks so they don't block each other. Related: [[delegate-coding-to-subagents]].
