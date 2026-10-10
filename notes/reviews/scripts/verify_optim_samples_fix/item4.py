import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
po = pd.read_csv(R+"sgd_pooled.csv")
print(po.round(4).to_string())
print(pd.read_csv(R+"sgd_fits.csv").to_string())
