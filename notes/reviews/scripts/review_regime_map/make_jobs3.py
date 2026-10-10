import json, sys, yaml
sys.path.insert(0, "experiments/regime_map")
from sweep import expand
C = "experiments/regime_map/configs/"
K = [c for c in expand(yaml.safe_load(open(C + "sigma_K_centred_d2.yaml")))
     if (c["kind"], c["n"], c["chi"], c["seed"]) == ("mps", 8, 4, 0) and c["sigma"] in (0.5125, 0.6729, 0.8835, 1.16)]
json.dump(K, open(sys.argv[1] + "/jobs_D3.json", "w")); print(len(K), [c["sigma"] for c in K])
