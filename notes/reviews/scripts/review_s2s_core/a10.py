import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
de=pd.read_csv(R+'dormant_decay.csv')
m=de[(de.tree=='mps4')&(de.chi==3)]
print(m.groupby(['teacher','seed','core']).rate_over_sigma2.agg(['min','max','size']).assign(ratio=lambda x:x['max']/x['min']).to_string())
# per (teacher, seed): end cores 0 and 3 (MPS ends)
for (t,s),g in m.groupby(['teacher','seed']):
    piv=g.pivot_table(index='sigma',columns='core',values='rate_over_sigma2')
    print(t,s); print(piv.to_string())
