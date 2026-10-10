import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',300); pd.set_option('display.max_columns',40)
f=pd.read_csv('experiments/order/results/h1b_fits.csv')
nb=f[f.set=='natural_b4'].dropna(axis=1,how='all')
print(nb.to_string())
print(f[f.set=='natural_mps4'].dropna(axis=1,how='all').to_string())
for s in ['saddle_b4_exponents','h1b_saddle_mps4_exponents','h1b_saddle_b4_exponents']:
    print(f[f.set==s].dropna(axis=1,how='all').to_string())
print(f[f.set=='h1b_saddle_mps4_by_sigma'].dropna(axis=1,how='all').to_string())
n=pd.read_csv('experiments/order/results/h1b_natural_b4.csv')
print(n.groupby(['rho','sigma']).agg(n=('seed','size'),wrong=('wrong','sum'),stalled=('stalled','sum'),status=('status','min'),tend=('t_end','min'),final_loss=('final_loss','max')).to_string())
lp=lambda r:(1-r)*r**(1/3)
w=n[n.wrong & (n.a2_min_after_t1<0.1)].copy()
w['scaled']=(w.t1-w.t2)*lp(w.rho)
print(w.groupby(['rho','sigma']).scaled.describe().round(3).to_string())
# pooled over rho by sigma
print(w.groupby('sigma').scaled.agg(['mean','median','count','std']).round(3))
# regress scaled on ln(1/sigma) for sigma<=0.01
ww=w[w.sigma<=0.01]
b,a=np.polyfit(np.log(1/ww.sigma),ww.scaled,1); print('slope of scaled vs ln(1/sigma), sigma<=0.01:',b)
import statsmodels.api as sm
