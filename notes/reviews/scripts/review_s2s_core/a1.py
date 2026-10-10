import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
d=pd.read_csv(R+'main_runs.csv')
print(d.groupby(['sweep','tree','chi','teacher','init'])['class'].value_counts().unstack(fill_value=0))
print(d.groupby(['sweep','tree','chi','teacher','init']).size())
