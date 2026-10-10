"""Summarize gf_origin results: q-first seeds, plateau length dt = t_esc - t_q vs sigma, escape pattern."""
import json, pathlib, sys
from collections import defaultdict
import numpy as np

R = pathlib.Path(__file__).parent / "results"
out = {}
for s in (0.8, 0.6):
    runs = defaultdict(dict)
    for f in R.glob(f"gf_s{s}_seed*_sigma*.json"):
        r = json.loads(f.read_text())
        runs[r["seed"]][r["sigma"]] = r
    out[s] = {}
    for seed in sorted(runs):
        rs = runs[seed]
        sig = sorted(rs, reverse=True)
        rows = []
        for x in sig:
            r = rs[x]
            qfirst = r["t_q"] is not None and (r["t_esc3"] is None or r["t_esc3"] > r["t_q"])
            e = r["snaps"].get("esc3", {}).get("amps", {})
            rows.append(dict(sigma=x, qfirst=qfirst, t_q=r["t_q"],
                             dt3=(r["t_esc3"] - r["t_q"]) if qfirst and r["t_esc3"] else None,
                             dt2=(r["t_esc2"] - r["t_q"]) if qfirst and r["t_esc2"] else None,
                             trapped=qfirst and r["t_esc3"] is None, L_end=r["L_end"], t_end=r["t_end"],
                             esc_amps={k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}))
        dts = [(z["sigma"], z["dt3"]) for z in rows if z["dt3"]]
        loc = [float(-np.log(b[1] / a[1]) / np.log(b[0] / a[0])) for a, b in zip(dts, dts[1:])]
        out[s][seed] = dict(rows=rows, local_exponents_dt3=loc)
        tag = "q-first" if all(z["qfirst"] for z in rows) else ("mixed" if any(z["qfirst"] for z in rows) else "r-first")
        print(f"s={s} seed={seed} {tag} trapped={[z['sigma'] for z in rows if z['trapped']]}")
        if dts:
            print("   sigma:", [z[0] for z in dts])
            print("   dt3  :", [round(z[1], 2) for z in dts])
            print("   local exponents:", np.round(loc, 3).tolist())
            last = [z for z in rows if z["dt3"]][-1]
            print("   amps at escape (smallest sigma):", last["esc_amps"])
(R / "gf_summary.json").write_text(json.dumps(out, indent=1))
