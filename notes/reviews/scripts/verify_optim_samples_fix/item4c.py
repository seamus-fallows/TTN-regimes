import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
su = pd.read_csv(R+"sgd_summary.csv")
f = su[su.init=='fanin']
print("fanin std1_over_pred: n", f.std1_over_pred.notna().sum(), "min", f.std1_over_pred.min().round(3), "max", f.std1_over_pred.max().round(3), "quantiles", f.std1_over_pred.quantile([0.1,0.25,0.5,0.75,0.9]).round(3).to_dict())
a = su[su.init=='aligned']
print("aligned std1_over_pred:", a.std1_over_pred.round(3).tolist())
sp = pd.read_csv(R+"sgd_second_plateau.csv")
print(sp.to_string())
