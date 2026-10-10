import pandas as pd, numpy as np, re
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 500)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/"
s = pd.read_csv(R + "route_amps_summary.csv")
num = lambda x: [float(v) for v in re.findall(r"np\.float64\(([-0-9.eE+naN]+)\)", str(x))]
rows = []
for _, r in s.iterrows():
    a, ase = num(r.end_vs_sigma); b, bse = num(r.interior_vs_sigma); g, gse = num(r.gap_exponent)
    k = r.k
    est = (k - 2) * (2 * a + k - 2) / k
    rows.append(dict(tree=r.tree, S=r.S, start=r.start, k=k, seeds=r.seeds, alpha=round(a, 2), alpha_se=round(ase, 2), beta=round(b, 2), gap=round(g, 2), gap_se=round(gse, 2), pred=k - 2, estimate=round(est, 2), est_minus_gap=round(est - g, 2)))
d = pd.DataFrame(rows)
print(d.to_string())
print([c for c in s.columns])
