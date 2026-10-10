import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
sa = pd.read_csv(R+"shift_accuracy_runs.csv")
print(sa.head().to_string())
print(sa.groupby('tree').size(), sorted(sa.ratio.unique()), sa.event.unique())
print("rel_dev def check:", np.allclose(sa.rel_dev.abs(), (sa.measured/sa.predicted-1).abs(), equal_nan=True), np.nanmax(np.abs(sa.rel_dev.abs()-np.abs(sa.measured/sa.predicted-1))))
low = sa[sa.ratio<=0.1+1e-12]
a = low.rel_dev.abs()
print("n<=0.1:", len(low), "nan:", a.isna().sum())
print("within 0.3/1/5%:", (a<=0.003).mean().round(3), (a<=0.01).mean().round(3), (a<=0.05).mean().round(3), " >5%:", (a>0.05).sum())
for r in sorted(low.ratio.unique()):
    aa = low[low.ratio==r].rel_dev.abs()
    print("ratio", r, "n", len(aa), "within 0.3%:", (aa<=0.003).mean().round(3))
# first events: event index meaning? 
print(sa.groupby('event').size())
