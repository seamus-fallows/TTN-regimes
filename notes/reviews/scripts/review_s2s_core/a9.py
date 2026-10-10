import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
d=pd.read_csv(R+'main_runs.csv')
x=d[d['class'].isin(['trapped','stalled','unresolved'])]
x=x.assign(lms=np.log10(x.min_dormant_slice_over_sigma))
print(x.groupby(['sweep','tree','chi','init','class']).lms.describe(percentiles=[.1,.5,.9]).round(1).to_string())
# drift
print(d.groupby(['sweep']).drift.max())
print(d.groupby(['sweep','init']).q0_max.median())
