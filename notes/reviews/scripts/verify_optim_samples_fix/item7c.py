import pandas as pd
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
s = pd.read_csv(R+"sgd_limit_test.csv")
print(pd.crosstab([s.B, s.ratio], s.outcome))
