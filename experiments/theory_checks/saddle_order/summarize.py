"""One line per case: k*, local slopes at the three smallest t, m/t^k* at the smallest t."""

import json
from pathlib import Path

for f in sorted((Path(__file__).parent / "results").glob("*.json")):
    o = json.loads(f.read_text())
    rows = o["rows"]
    sl = [r.get("slope") for r in rows[-3:]]
    last = rows[-1]
    k = o["k_star"]
    print(f"{o['name']:22s} k*={k}  slopes={[None if s is None else round(s, 3) for s in sl]}  "
          f"m/t^k*={last['m_over_tk']:+.4g}  route={last['route_pred'] / last['t'] ** k:+.4g}  "
          f"old_frac={last['old_frac']:.1e}")
