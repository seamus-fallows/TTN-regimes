import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
d=pd.read_csv(R+'main_runs.csv')
for sw,tree,init,chi in [('k78','binary8','fanin',3),('k78','ht4','fanin',3),('k78','mps8','fanin',3),('k78','ht4','balanced',3),('main','binary8','fanin',3),('main','ht4','fanin',3),('main','mps8','fanin',3)]:
    x=d[(d.sweep==sw)&(d.tree==tree)&(d.init==init)&(d.chi==chi)&(d.teacher=='separated')]
    print(sw,tree,init,chi); print(pd.crosstab(x.seed,x.sigma,values=x['class'].str[0]+':'+x.order.astype(str),aggfunc='first'))
    print(x.groupby('sigma')['class'].apply(lambda c:(c=='trapped').mean()).round(2).to_dict())
x=d[(d.sweep=='main')&(d.tree=='mps4')&(d.chi==3)]
for t in ['separated','close']:
    y=x[x.teacher==t]; print('mps4 chi3',t); print(pd.crosstab(y.seed,y.sigma,values=y['class'].str[0],aggfunc='first'))
