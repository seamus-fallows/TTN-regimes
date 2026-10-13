### Later events (all events after the escape from the origin), per target and tree

Medians [quartiles] over fitted series (seed x event). d_eff: growth-law depth; p: duration ~ sigma^-p; prediction d_eff = k_region, p = k_region - 2.

| target | tree | events | series | k_region | d_eff | p | d_eff within 0.3 of k_region | p within 0.3 of k_region - 2 | p within 0.3 of d_eff - 2 | log law preferred | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pattern (control) | binary8 | all | 90 | 3 [3, 4] | 3.10 [3.04, 4.18] | 1.01 [1.00, 2.00] | 96% | 100% | 96% | 0% | HOLDS |
| pattern (control) | binary8 | k=3 | 60 | 3 [3, 3] | 3.04 [3.04, 3.10] | 1.00 [1.00, 1.01] | 93% | 100% | 93% | 0% | HOLDS |
| pattern (control) | binary8 | k=4 | 30 | 4 [4, 4] | 4.18 [4.18, 4.18] | 2.00 [2.00, 2.00] | 100% | 100% | 100% | 0% | HOLDS |
| pattern (control) | mps6 | all | 81 | 3 [2, 4] | 3.30 [1.73, 4.09] | 1.01 [0.18, 2.00] | 62% | 100% | 60% | 26% | PARTIAL |
| pattern (control) | mps6 | k=2 | 21 | 2 [2, 2] | 1.51 [1.50, 1.54] | 0.14 [0.13, 0.15] | 5% | 100% | 0% | 100% | PARTIAL |
| pattern (control) | mps6 | k=3 | 30 | 3 [3, 3] | 3.24 [3.07, 3.36] | 1.00 [1.00, 1.01] | 63% | 100% | 63% | 0% | PARTIAL |
| pattern (control) | mps6 | k=4 | 30 | 4 [4, 4] | 4.09 [4.09, 4.10] | 2.00 [1.99, 2.00] | 100% | 100% | 100% | 0% | HOLDS |
| CP, overlap 0.3 | binary8 | all | 55 | 7 [7, 7] | 3.09 [2.34, 4.08] | 1.00 [0.93, 1.08] | 12% | 9% | 29% | 2% | FAILS |
| CP, overlap 0.3 | binary8 | k=3 | 6 | 3 [3, 3] | 3.09 [3.09, 3.09] | 1.09 [1.01, 1.12] | 100% | 83% | 83% | 0% | INCONCLUSIVE |
| CP, overlap 0.3 | binary8 | k=7 | 49 | 7 [7, 7] | 3.02 [2.29, 4.16] | 0.99 [0.91, 1.08] | 0% | 0% | 21% | 2% | FAILS |
| CP, overlap 0.3 | mps6 | all | 60 | 6 [6, 6] | 1.53 [1.28, 1.67] | 0.12 [0.10, 0.14] | 0% | 0% | 2% | 87% | FAILS |
| CP, overlap 0.3 | mps6 | k=6 | 60 | 6 [6, 6] | 1.53 [1.28, 1.67] | 0.12 [0.10, 0.14] | 0% | 0% | 2% | 87% | FAILS |
| CP, overlap 0.6 | binary8 | all | 77 | 7 [3, 7] | 3.06 [3.04, 3.09] | 1.00 [1.00, 1.01] | 43% | 42% | 97% | 5% | FAILS |
| CP, overlap 0.6 | binary8 | k=3 | 34 | 3 [3, 3] | 3.06 [3.05, 3.07] | 1.00 [0.99, 1.00] | 100% | 94% | 97% | 3% | HOLDS |
| CP, overlap 0.6 | binary8 | k=7 | 43 | 7 [7, 7] | 3.05 [3.01, 3.13] | 1.00 [1.00, 1.03] | 0% | 0% | 98% | 7% | FAILS |
| CP, overlap 0.6 | mps6 | all | 45 | 6 [6, 6] | 1.97 [1.90, 2.00] | 0.13 [0.12, 0.15] | 0% | 0% | 89% | 100% | FAILS |
| CP, overlap 0.6 | mps6 | k=6 | 45 | 6 [6, 6] | 1.97 [1.90, 2.00] | 0.13 [0.12, 0.15] | 0% | 0% | 89% | 100% | FAILS |
| random TTN | binary8 | all | 71 | 3 [3, 3] | 3.04 [3.02, 3.07] | 1.00 [1.00, 1.00] | 83% | 83% | 100% | 0% | HOLDS |
| random TTN | binary8 | k=3 | 59 | 3 [3, 3] | 3.02 [3.01, 3.04] | 1.00 [1.00, 1.00] | 100% | 100% | 100% | 0% | HOLDS |
| random TTN | binary8 | k=5 | 5 | 5 [5, 5] | 3.10 [3.10, 3.10] | 1.00 [1.00, 1.00] | 0% | 0% | 100% | 0% | INCONCLUSIVE |
| random TTN | binary8 | k=7 | 7 | 7 [7, 7] | 3.19 [3.12, 3.25] | 1.02 [1.02, 1.03] | 0% | 0% | 100% | 0% | INCONCLUSIVE |
| random TTN | mps6 | all | 64 | 2 [2, 3] | 2.03 [1.83, 2.09] | 0.15 [0.11, 0.18] | 38% | 55% | 72% | 80% | FAILS |
| random TTN | mps6 | k=2 | 35 | 2 [2, 2] | 2.00 [1.66, 2.09] | 0.12 [0.10, 0.15] | 69% | 100% | 66% | 74% | PARTIAL |
| random TTN | mps6 | k=3 | 14 | 3 [3, 3] | 2.04 [2.02, 2.24] | 0.18 [0.16, 0.22] | 0% | 0% | 71% | 93% | FAILS |
| random TTN | mps6 | k=4 | 2 | 4 [4, 4] | 2.12 [2.11, 2.12] | 0.14 [0.13, 0.14] | 0% | 0% | 100% | 100% | INCONCLUSIVE |
| random TTN | mps6 | k=5 | 6 | 5 [5, 5] | 2.03 [2.02, 2.06] | 0.16 [0.14, 0.17] | 0% | 0% | 83% | 67% | INCONCLUSIVE |
| random TTN | mps6 | k=6 | 7 | 6 [6, 6] | 2.05 [2.01, 2.06] | 0.16 [0.11, 0.18] | 0% | 0% | 86% | 86% | INCONCLUSIVE |

### Event 0 (escape from the origin; p = K - 2 is an exact identity: a check of the estimators)

| target | tree | series | k_region | d_eff | p |
|---|---|---|---|---|---|
| pattern (control) | binary8 | 30 | 7 [7, 7] | 7.005 [6.995, 7.013] | 5.000 [5.000, 5.000] |
| pattern (control) | mps6 | 21 | 6 [6, 6] | 5.993 [5.986, 5.999] | 4.000 [4.000, 4.000] |
| CP, overlap 0.3 | binary8 | 30 | 7 [7, 7] | 7.001 [7.000, 7.002] | 5.000 [5.000, 5.000] |
| CP, overlap 0.3 | mps6 | 30 | 6 [6, 6] | 5.999 [5.993, 6.001] | 4.000 [4.000, 4.000] |
| CP, overlap 0.6 | binary8 | 30 | 7 [7, 7] | 6.997 [6.993, 7.001] | 5.000 [5.000, 5.000] |
| CP, overlap 0.6 | mps6 | 30 | 6 [6, 6] | 6.003 [5.999, 6.031] | 4.000 [4.000, 4.000] |
| random TTN | binary8 | 30 | 7 [7, 7] | 7.001 [7.000, 7.003] | 5.000 [5.000, 5.000] |
| random TTN | mps6 | 25 | 6 [6, 6] | 5.990 [5.972, 5.995] | 4.000 [4.000, 4.000] |

### Runs and counts

| target | tree | runs | converged / completed | trapped | runs with a relearned rise | floor-flagged events (runs) | series fitted / all |
|---|---|---|---|---|---|---|---|
| pattern (control) | binary8 | 330 | 330 | 0 | 0 | 45 (45) | 120 / 121 |
| pattern (control) | mps6 | 330 | 330 | 0 | 0 | 0 (0) | 102 / 146 |
| CP, overlap 0.3 | binary8 | 330 | 330 | 0 | 90 | 110 (110) | 85 / 110 |
| CP, overlap 0.3 | mps6 | 330 | 330 | 0 | 0 | 0 (0) | 90 / 98 |
| CP, overlap 0.6 | binary8 | 330 | 330 | 0 | 86 | 68 (68) | 107 / 115 |
| CP, overlap 0.6 | mps6 | 330 | 330 | 0 | 0 | 0 (0) | 75 / 117 |
| random TTN | binary8 | 330 | 330 | 0 | 0 | 37 (37) | 101 / 111 |
| random TTN | mps6 | 330 | 330 | 0 | 0 | 0 (0) | 89 / 210 |
| MNIST | binary16_quadtree | 90 | 90 | 0 | 26 | 0 (0) | 0 / 126 |
| MNIST | mps16_snake | 22 | 22 | 0 | 1 | 0 (0) | 0 / 107 |
