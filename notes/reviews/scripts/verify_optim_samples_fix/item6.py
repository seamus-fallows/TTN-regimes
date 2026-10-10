import pandas as pd, numpy as np
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
s = pd.read_csv(R+"samples_summary.csv")
print(s.setup.unique())
print(s[['setup','label','n','test_mean','test_se','test_median','p_rec','eff_rank','train_rel','rel_change','n_seeds','frac_converged']].to_string(float_format=lambda v: f"{v:.3g}"))
