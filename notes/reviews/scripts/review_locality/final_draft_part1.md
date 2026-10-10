# Reproduction review: locality round 1 (draft part 1)

Verdict in one line: everything I reran reproduces the committed numbers to all printed digits (deterministic),
the independent reimplementation agrees with the runner, and the headline claims (1)-(3), (6) survive; the
problems are in bookkeeping and in how some numbers are summarized: (7)'s "2/1008" is 16/1008 in the
committed table, the chi = 4 stall count omits a fully stalled case, claim (4)'s 98.7% is close to tautological,
claim (5) fails for small-init k = 2 on N >= 6, and the saddle-start k = 2 lambda_min is -0.44..-0.48 (a gauge
effect), not -0.49..-0.50.
