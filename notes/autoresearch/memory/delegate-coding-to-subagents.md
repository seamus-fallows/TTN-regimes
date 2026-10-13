---
name: delegate-coding-to-subagents
description: "As research lead/manager, delegate all code writing and experiment running to agents (local subagents for small things, remote crab-student agents for heavy compute); lead only directs, briefs, reviews, writes"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ccfcd9a3-98b9-48c2-8218-49a3ab02b926
  modified: 2026-10-09T17:58:19.186Z
---

When acting as lead researcher / research manager, do not write code or run experiments yourself. Delegate implementation (library modules, experiment scripts, pilots) and all experiment running to agents. Small pilots and light computation can go to local subagents; compute-heavy experiments go to remote crab-student (Vast.ai) agents. The lead's job is direction, specs and briefs, critical review of results, and writing.

**Why:** On 2026-10-09 (TTN regimes project) the user said "You shouldn't be writing all the code yourself. this is what the subagents are for", then "The point is for you to use other agents to run experiments for you", and then clarified "you don't have to send small things to remote agents".

**How to apply:** For any non-trivial code or computation, write a brief and spawn an agent — local for small/light work, remote for heavy sweeps. When reporting, make clear which agent is doing what. Editing notes, plans, CLAUDE.md and the report remains the lead's job.
