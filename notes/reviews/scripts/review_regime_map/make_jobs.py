import json, sys, yaml
sys.path.insert(0, "experiments/regime_map")
from sweep import expand
C = "experiments/regime_map/configs/"
def load(name): return expand(yaml.safe_load(open(C + name)))
sp = sys.argv[1]
imb = [c for c in load("imbalance_b4.yaml") if c["chi"] == 16]
chib4 = [c for c in load("chi_b4.yaml") if c["model"] == "centred" and c["chi"] in (16, 64, 128) and c["seed"] in (0, 1)]
large = [c for c in load("large_chi_b4.yaml") if c["model"] == "centred" and c["sigma"] == 1.5 and c["chi"] >= 16]
sc = [c for c in load("sigma_chi_b4.yaml") if c["model"] == "centred" and c["chi"] in (3, 9, 36) and c["seed"] in (0, 1)]
K = [c for c in load("sigma_K_centred_d2.yaml") if (c["kind"], c["n"], c["chi"]) in (("binary", 4, 4), ("mps", 5, 4)) and c["seed"] in (0, 1, 2)]
A = imb + sorted(chib4, key=lambda c: (-c["sigma"], c["chi"]))
B = large + sorted(sc, key=lambda c: (c["chi"] == 3, -c["sigma"]))
json.dump(A, open(sp + "/jobs_A.json", "w")); json.dump(B, open(sp + "/jobs_B.json", "w")); json.dump(K, open(sp + "/jobs_C.json", "w"))
print(len(imb), len(chib4), len(large), len(sc), len(K))
print(imb[:2]); print(chib4[0]); print(large[0]); print(K[0])
