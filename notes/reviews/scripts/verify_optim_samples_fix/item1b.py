import pandas as pd, numpy as np
pd.set_option('display.width', 250)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
of = pd.read_csv(R+"order_flips.csv")
print(pd.crosstab(of.sigma, of.ratio))
# duplicates?
print("dups (sigma,seed,ratio):", of.duplicated(['sigma','seed','ratio']).sum())
gs = pd.read_csv(R+"gd_sweep.csv")
print(gs.groupby(['tree','centred','chi']).size())
print("ratio_ref values", gs.ratio_ref.unique())
# The step-1 sweep: plain model, chi=3
s1 = gs[(gs.centred==False)|(gs.centred==0)]
print(s1.groupby(['tree','chi','ratio_ref']).size())
