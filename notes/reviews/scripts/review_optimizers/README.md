# review_optimizers scratch index (reproduction reviewer)
- notes_1_setup.md        README check, workaround, run-id matching
- notes_2_independent.md  independent checks (own Adam/RMSProp/GF-event code), eps law, stall tolerance
- notes_3_code_issues.md  bugs / analysis issues
- notes_4_claims.md       claim-level observations from committed CSVs
- notes_5_reproduction.md per-run comparison of reproduced runs (agent code) vs committed rows
Scripts: driver.py (spawn pool around run.work), rep_*.yaml (subset configs), compare*.py, indep_*.py,
reanalyze_eps.py, fit_b8_balanced.py, indep_fit.py
Data: c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_optimizers/{raw,results,indep}
