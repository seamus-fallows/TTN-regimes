import json
d = json.load(open("c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/route_sequences.json"))
for k, v in d.items():
    print(k, " -> ".join(f"L={s['loss']:.4f} k*={s['k_star']} C={[r[0] for r in s['routes']][:3]}" for s in v))
