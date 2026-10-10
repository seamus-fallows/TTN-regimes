export const meta = {
  name: 'verify-saddle-dyn',
  description: 'Verify the saddle-dyn experiment (pre-registration integrity and independent numerics) before it enters the report',
  phases: [
    { title: 'Verify', detail: 'integrity of the pre-registered analysis; independent recomputation' },
    { title: 'Synthesize', detail: 'safe-to-cite list for the report' },
  ],
}

const COMMON = `Repository: c:/Users/seamu/Iliad-ML/TTN_regimes (CLAUDE.md = project map). Python: \`uv run python ...\` from the repo root (float64, torch.set_num_threads(1) per process, at most 6 processes). Never kill processes by name. READ-ONLY for the repository: write only the output file named below; do not commit.
The experiment under review is saddle-dyn (merged to main as 6555568, UNVERIFIED): it tests the dynamical consequence of the saddle-order theorem (notes/theory_saddle_order.md v2; verifications notes/reviews/saddle_order_verification_proof.md and _numerics.md): the escape time from a learned rank-one saddle should scale as eps^-(p-2) (saddle start) and sigma^-(p-2) (small init), with p the theorem's saddle order, instead of the route law with k*.
Its pre-registration is notes/saddle_dyn.md Part I (commit 3377a9f); results in notes/saddle_dyn.md Part II, agent-runs/saddle-dyn/REPORT.md, experiments/saddle_dyn/ (code, results/*.csv, figures); raw JSON in data/saddle_dyn/ (being copied; usable once C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/tasks/bqkk1zuxy.output contains a line starting with "remote"; until then use the committed CSVs).
Keep each message under ~5k tokens; write your file in small pieces.`

const VERDICT = {
  type: 'object',
  properties: {
    verdicts: { type: 'array', items: { type: 'object', properties: { claim: { type: 'string' }, verdict: { type: 'string', enum: ['holds', 'holds with fix', 'not supported', 'wrong'] }, evidence: { type: 'string' }, fix: { type: 'string' } }, required: ['claim', 'verdict', 'evidence'] } },
    errata: { type: 'array', items: { type: 'string' } },
  },
  required: ['verdicts', 'errata'],
}

phase('Verify')
const [integrity, numerics] = await parallel([
  () => agent(`${COMMON}

LENS: PRE-REGISTRATION AND ANALYSIS INTEGRITY. Check: (1) the registration commit 3377a9f precedes every run (git history vs raw file times in data/saddle_dyn/ or the run logs in agent-runs/saddle-dyn/); (2) the registered cells, windows and decision rules were applied as registered; list deviations 1-7 and judge each: methodological (integrator, stop rules, re-referencing) vs outcome-driven; was any window or rule changed after seeing outcomes; (3) the three cells added from the lead's message before the commit are disclosed; (4) the decisions SUPPORTED / INCONCLUSIVE / REFUTED in the REPORT table follow the registered rules; (5) "every seed within 0.05" and "route law worse by an RSS factor 1e3-2e5" are computed as stated; (6) the claim that the earlier numerics verifier's PS06 "12/12 q-first" was a misclassification (7 of 12 learn the residual pair first) is correct. Write notes/reviews/saddle_dyn_verification_integrity.md and return the structured verdicts.`, { label: 'verify:integrity', phase: 'Verify', schema: VERDICT }),
  () => agent(`${COMMON}

LENS: INDEPENDENT NUMERICS. Write your own analysis code in a NEW folder experiments/theory_checks/saddle_dyn_verify/ (you may create files there; nothing else) and recompute from the raw JSON or committed per-run CSVs: the per-seed escape exponents and their means for PS08, W6, HT4W, HT4L, B8W05 (claimed 2.001, 2.001, 3.996, 3.997, 3.998 with every seed within 0.05), B8W (4.48 +- 0.15) and HT4S (4.72 +- 0.14), and the small-init pair-star exponent (2.002 +- 0.001, 16 of 40 seeds visit). Then re-run from scratch with your own integration (ttn.dynamics.integrate_gf with method="lsoda", or the verifier code in experiments/theory_checks/saddle_order_verify/) at least 2 directions x 4 eps values for PS08 and HT4W and check escape times and exponents. Check the accuracy claims (tolerance reruns <= 1.6e-7) on a sample. Write notes/reviews/saddle_dyn_verification_numerics.md and return the structured verdicts. Budget: at most ~45 min on at most 6 processes.`, { label: 'verify:numerics', phase: 'Verify', schema: VERDICT }),
])

phase('Synthesize')
const synth = await agent(`${COMMON}

Two independent verifications of saddle-dyn are done:
INTEGRITY (JSON): ${JSON.stringify(integrity)}
NUMERICS (JSON): ${JSON.stringify(numerics)}
Their files: notes/reviews/saddle_dyn_verification_integrity.md and notes/reviews/saddle_dyn_verification_numerics.md.
Write notes/reviews/saddle_dyn_verification.md: a combined verdict, the errata, and a SAFE-TO-CITE list for the report with status tags (proven / derived / empirical / conjecture) and the exact wording for three places that currently hold placeholders: (a) Sec. 1 (one or two sentences in the finding on the saddle-order theorem), (b) the "When is k* the saddle order?" subsection of Sec. 4 (a short paragraph with the key numbers and the scope: which cells support, which are inconclusive or refuted and why), (c) App. A.4 (one or two sentences on the dynamical consequence). Be conservative: claim only what both verifications support. Return a short summary (at most 200 words).`, { label: 'synthesize', phase: 'Synthesize' })
return { integrity, numerics, synth }