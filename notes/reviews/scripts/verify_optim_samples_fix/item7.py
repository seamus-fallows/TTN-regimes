import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
print(pd.read_csv(R+"p5_excess_loss.csv").round(4).to_string())
gsu = pd.read_csv(R+"gd_summary.csv")
rs = gsu[gsu.root_scale.notna()][['root_scale','ratio','n_runs','frac_gf_like','frac_catapult','frac_osc','frac_div','frac_other','frac_reentry']]
print(rs.to_string())
k2 = pd.read_csv(R+"k2_saddle_sweep.csv")
print(k2.groupby('sigma').agg(n=('lambda_min_raw','count'), raw=('lambda_min_raw','mean'), raw_sd=('lambda_min_raw','std'), proj=('lambda_min','mean')).round(4).to_string())
st = pd.read_csv(R+"sgd_stability.csv"); print(st[['tree','N','K','B','eta_ms']].assign(per_B=lambda d: d.eta_ms/d.B).round(6).to_string())
