import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 200)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
gf = pd.read_csv(R+"centred_chi9_gf.csv")
print(gf.to_string())
print(pd.read_csv(R+"centred_chi9_lambda_fit.csv").to_string())
