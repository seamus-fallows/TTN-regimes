import json, sys, yaml
sys.path.insert(0, "experiments/regime_map")
from sweep import expand
C = "experiments/regime_map/configs/"
def load(name): return expand(yaml.safe_load(open(C + name)))
sp = sys.argv[1]
imb = [c for c in load("imbalance_b4.yaml") if c["chi"] == 16]
chib4 = [c for c in load("chi_b4.yaml") if c["model"] == "centred" and (c["chi"], c["seed"]) in ((64, 0), (128, 1))]
large = [c for c in load("large_chi_b4.yaml") if c["model"] == "centred" and c["sigma"] == 1.5 and c["chi"] >= 16 and c["seed"] in (0, 1)]
sc = load("sigma_chi_b4.yaml")
sc36 = [c for c in sc if c["model"] == "centred" and c["chi"] == 36 and c["seed"] == 0]
sc12 = [c for c in sc if c["model"] == "centred" and c["chi"] == 12 and c["seed"] == 4 and c["sigma"] in (0.01, 0.01585)]
sc3 = [c for c in sc if c["model"] == "centred" and c["chi"] == 3 and c["seed"] == 0 and c["sigma"] in (0.6325, 2.52)]
K = [c for c in load("sigma_K_centred_d2.yaml") if (c["kind"], c["n"], c["chi"]) == ("mps", 5, 4) and c["seed"] == 0]
D1 = imb + sorted(chib4, key=lambda c: (c["chi"], -c["sigma"]))
D2 = large + sc12 + sc36 + sc3 + K
json.dump(D1, open(sp + "/jobs_D1.json", "w")); json.dump(D2, open(sp + "/jobs_D2.json", "w"))
print(len(D1), len(D2))
