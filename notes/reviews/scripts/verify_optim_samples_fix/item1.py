import pandas as pd, numpy as np
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
of = pd.read_csv(R+"order_flips.csv")
print(of.head(10).to_string())
print(of.describe(include='all').T[['count','unique','mean','min','max']] if 'unique' in of.describe(include='all').T else of.describe().T)
print("ratio values:", sorted(of.ratio.unique()))
print("sigma values:", sorted(of.sigma.unique()))
print("flip value counts:", of.flip.value_counts(dropna=False).to_dict())
print("transient value counts:", of.transient.value_counts(dropna=False).to_dict())
# ratio==0 might be GF
for r, g in of.groupby('ratio'):
    print(r, len(g), "flips", g.flip.sum(), "transient", g.transient.sum())
