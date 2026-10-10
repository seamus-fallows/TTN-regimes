import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
gs = pd.read_csv(R+"gd_sweep.csv")
c9 = gs[gs.chi==9].copy()
c9['lrel'] = c9.loss_end/c9.loss0
c9['lgf_rel'] = c9.loss_gf_end/c9.loss0
for cls in ['gf_like','catapult']:
    g = c9[c9.cls==cls]
    print(cls, len(g))
    print(g.groupby(['centred','sigma','ratio']).agg(n=('seed','size'), lrel_max=('lrel','max'), lrel_med=('lrel','median'), lgf=('lgf_rel','max'), stab_min=('stab_end','min'), stab_max=('stab_end','max')).to_string())
print("loss0 values:", c9.groupby(['centred','sigma']).loss0.mean().round(3).to_dict())
g = c9[c9.cls=='catapult']
print("catapult stab_end range:", g.stab_end.min(), g.stab_end.max(), " lrel>1e-6:", (g.lrel>1e-6).sum(), "of", len(g), " lrel>1e-4:", (g.lrel>1e-4).sum())
# gf_like above 2/lam
gg = c9[(c9.cls=='gf_like')&(c9.ratio>1)]
print(gg[['sigma','seed','ratio','loss_end','loss_gf_end','stab_end','sharp_end_rel','sharp_gf_rel','max_rise']].to_string())
