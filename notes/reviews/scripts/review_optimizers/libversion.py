"""Which reproduced runs were made with the uncommitted (modified 21:54-21:55) library: the new Recorder writes
fine rows ('fine_spectra') and the new Problem.describe() writes 'component_targets'."""
import collections, glob, json, os
import numpy as np
c = collections.Counter()
for f in glob.glob("c:/Users/seamu/Iliad-ML/TTN_regimes/data/review_optimizers/raw/*/*.npz"):
    if f.endswith(".tmp.npz"):
        continue
    z = np.load(f, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    sweep = os.path.basename(os.path.dirname(f))
    new = "component_targets" in meta.get("problem", {})
    c[(sweep, meta["cfg"]["optimizer"], "new_lib" if new else "old_lib")] += 1
for k, v in sorted(c.items()):
    print(k, v)
