import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
p = pd.read_csv(R+"p9_sgd.csv")
print(len(p), p.groupby(['b0','B']).size().to_dict())
print("seeds:", sorted(p.seed.unique()))
c = pd.read_csv(R+"p9_conditions.csv"); print(c.round(4).to_string())
print(pd.read_csv(R+"p9_fit.csv").to_string())
