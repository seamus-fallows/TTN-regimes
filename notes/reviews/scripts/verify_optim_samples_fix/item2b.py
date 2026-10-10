import pandas as pd, numpy as np
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
gf = pd.read_csv(R+"centred_chi9_gf.csv")
lam_bal = 3*3**(2-2/3)
c = gf[gf.centred]
print("n centred:", len(c), "max loss_end:", c.loss_end.max(), "all <=1e-8:", (c.loss_end<=1e-8).all())
p = gf[~gf.centred]
print("plain:", p.groupby('sigma').agg(n=('seed','size'), loss_max=('loss_end','max'), lam_rel=('lam_gf', lambda x: (x/lam_bal).mean()), t=('t_end','mean')).to_string())
g = c.groupby('sigma').agg(lam=('lam_gf','mean'), lam_min=('lam_gf','min'), lam_max=('lam_gf','max'), t_mean=('t_end','mean'), t_med=('t_end','median'))
g['lam_rel'] = g.lam/lam_bal; g['2/lam over eta_max'] = lam_bal/g.lam
print(g.to_string())
# per-seed fits
for lo, hi in [(1.41,16),(1.0,16),(2,16)]:
    ex=[]
    for s, gg in c.groupby('seed'):
        q = gg[(gg.sigma>=lo-1e-9)&(gg.sigma<=hi+1e-9)].sort_values('sigma')
        for sig_exact in [False, True]:
            pass
        sl = np.polyfit(np.log(q.sigma), np.log(q.lam_gf), 1)[0]
        ex.append(sl)
    ex=np.array(ex)
    print(f"range {lo}-{hi}: per-seed exponents {np.round(ex,3)} mean {ex.mean():.4f} se {ex.std(ddof=1)/np.sqrt(len(ex)):.4f}")
# exact sqrt2 sigmas
smap = {1.41: np.sqrt(2), 2.83: 2*np.sqrt(2), 5.66: 4*np.sqrt(2), 11.3: 8*np.sqrt(2)}
ex=[]
for s, gg in c.groupby('seed'):
    q = gg[gg.sigma>=1.4]
    sg = q.sigma.map(lambda x: smap.get(x, x))
    ex.append(np.polyfit(np.log(sg), np.log(q.lam_gf), 1)[0])
ex=np.array(ex); print("with exact sqrt2 sigmas:", ex.mean().round(4), (ex.std(ddof=1)/np.sqrt(5)).round(4))
# local slopes using means of log lam
m = c.groupby('sigma').lam_gf.apply(lambda x: np.mean(np.log(x)))
sig = m.index.values
for i in range(len(sig)-1):
    print(f"local slope {sig[i]}-{sig[i+1]}: {(m.values[i+1]-m.values[i])/(np.log(sig[i+1])-np.log(sig[i])):.3f}")
