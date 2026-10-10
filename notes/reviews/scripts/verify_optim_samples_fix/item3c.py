import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
rich = pd.read_csv(R+"shift_richardson.csv")
print("Richardson n:", len(rich), "median:", rich.rel_dev_richardson.median(), "max:", rich.rel_dev_richardson.max())
print(rich.sort_values('rel_dev_richardson', ascending=False).head(5).to_string())
# which (tree, sigma) feed Richardson
print(rich.groupby(['tree','sigma']).size().to_dict())
sa = pd.read_csv(R+"shift_accuracy_runs.csv")
low = sa[sa.ratio<=0.1+1e-12]
print(low.sort_values('rel_dev', ascending=False).head(16).to_string())
ex = sa[(sa.tree=='binary4')&(sa.sigma==0.03)&(sa.seed==1)&(sa.event==2)]
print(ex.to_string())
# ratio of measured/predicted extremes
q = low.measured/low.predicted
print("measured/predicted range at <=0.1:", q.min(), q.max())
