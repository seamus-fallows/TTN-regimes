"""Compare reproduced runs (summarize.py / route_amplitudes.py / plateau_hessians.py outputs) with the
committed per-run reference values."""
import json
import sys
from pathlib import Path

import numpy as np

RES = Path("c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/real_data/results")
A = Path(sys.argv[1])  # analysis dir with repro_summary.json, routes.json, hessians.json
summ = {r["name"]: r for r in json.load(open(A / "repro_summary.json")) if "error" not in r}
reg = {r["name"]: r for r in json.load(open(RES / "regime_analysis.json"))}
routes_ref = {r["name"]: r for r in json.load(open(RES / "routes_raw.json")) if "pairs" in r}
hess_ref = {r["name"]: r for r in json.load(open(RES / "hessians_raw.json")) if "raw_min" in r}
ev_ref = json.load(open(RES / "events_table.json"))
routes = {r["name"]: r for r in json.load(open(A / "routes.json"))} if (A / "routes.json").exists() else {}
hess = {r["name"]: r for r in json.load(open(A / "hessians.json"))} if (A / "hessians.json").exists() else {}


def share(pairs, kmax=5):
    lam = np.array([p["lam"] for p in pairs])
    kk = np.array([p["k"] for p in pairs])
    return float((lam[kk <= kmax] ** 2).sum() / (lam ** 2).sum())


for name, r in sorted(summ.items(), key=lambda kv: (kv[1]["tree"], kv[1]["ordering"], kv[1]["sigma"])):
    tag = name.replace("mnist4_3v5_n4000_raw_", "")
    drops = r.get("drops", [])
    print(f"== {tag}: t_a {r['t_a']:.10g} t1 {r.get('t1', float('nan')):.6g} n_drops {r.get('n_drops')} "
          f"drop t {[round(d['t'], 2) for d in drops]} gaps {[round(g, 2) for g in r.get('gaps', [])]} "
          f"rho {[round(x, 2) for x in r.get('rho', [])]} first_rank2 {r.get('first_rank2_edges')} t_first_rank2 {r.get('t_first_rank2', float('nan')):.4g} "
          f"tL0.25 {r.get('t_loss_0.25')} tL0.2 {r.get('t_loss_0.2')} L_end {r.get('loss_end'):.4f} (tau {r.get('t2')}) drift {r.get('drift_max'):.2g}")
    if name in reg:
        g = reg[name]
        print(f"   ref regime_analysis: t_a {g['t_a']:.10g} (rel diff {(r['t_a'] - g['t_a']) / g['t_a']:.2e}) n_drops(1e4) {g['n_drops']} rho {[round(x, 2) for x in g['rho']]}")
    # events_table reference for this seed / tree
    for e in ev_ref:
        tr = r["tree"] + ":" + r["ordering"]
        if e["tree"] == tr and e["features"] == "raw" and e["seed"] == r["seed"] and r["sigma"] in e["sigmas"]:
            i = e["sigmas"].index(r["sigma"])
            j = e["drop_index"][i]
            mine = drops[j]["t"] - drops[j - 1]["t"] if len(drops) > j else float("nan")
            print(f"   ref events_table: signature {e['signature']} k {e['k']} drop {j} dt_ref {e['dt'][i]:.2f} | mine {mine:.2f} (drop_edges mine {r['drop_edges'][j] if len(drops) > j else None})")
    if name in routes_ref:
        rr = routes_ref[name]
        print(f"   ref routes: tau_mid {rr['tau_mid']} L_mid {rr['train_loss']:.4f} share(k<=5) {share(rr['pairs']):.3f} best-k3 lam {max(p['lam'] for p in rr['pairs'] if p['k'] == 3):.4f} max lam {max(p['lam'] for p in rr['pairs']):.4f} rank1 err {rr['rank1_rel_err']:.3g} first {rr['first_rank2_edges']}")
    if name in routes and "pairs" in routes[name]:
        rr = routes[name]
        print(f"   mine routes: tau_mid {rr['tau_mid']} L_mid {rr['train_loss']:.4f} share(k<=5) {share(rr['pairs']):.3f} best-k3 lam {max(p['lam'] for p in rr['pairs'] if p['k'] == 3):.4f} max lam {max(p['lam'] for p in rr['pairs']):.4f} rank1 err {rr['rank1_rel_err']:.3g}")
    if name in hess_ref:
        h = hess_ref[name]
        print(f"   ref hessian: tau_mid {h['tau_mid']} loss_mid {h['loss_mid']:.4f} raw_min {h['raw_min']:.5f} proj_min {h['proj_min']:.5f} n_neg {h['proj_n_neg']}")
    if name in hess and "raw_min" in hess[name]:
        h = hess[name]
        print(f"   mine hessian: tau_mid {h['tau_mid']} loss_mid {h['loss_mid']:.4f} raw_min {h['raw_min']:.5f} proj_min {h['proj_min']:.5f} n_neg {h['proj_n_neg']}")
