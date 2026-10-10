import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
of = pd.read_csv(R+"order_flips.csv")
d = of[of.duplicated(['sigma','seed','ratio'], keep=False)].sort_values(['sigma','seed'])
print(d.to_string())
gs = pd.read_csv(R+"gd_sweep.csv")
b = gs[(gs.tree=='binary')&(gs.chi==3)&(gs.centred==False)]
print(b.groupby(['n','root_scale']).size())
# which binary-4 runs at sigma 0.1/0.3 ratio .5 with seeds in the dups
for _, r in d.drop_duplicates(['sigma','seed']).iterrows():
    m = b[(b.n==4)&(b.sigma==r.sigma)&(b.seed==r.seed)&(np.isclose(b.ratio,0.5))]
    print(r.sigma, r.seed, len(m), m[['root_scale','cls','same_order','t1_gd','t2_gd']].to_dict('records'))
