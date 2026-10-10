"""Re-run the correlation-flow (LO) blow-up for binary4 close-teacher seeds where the LO-predicted
first component disagrees with GF, with larger stopping amplitudes, to see whether r1 flips."""
import sys, json, time
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])
import s2s_copy as s2s
out = sys.argv[2]
cases = [(3, 'fanin', 4), (3, 'balanced', 6), (6, 'fanin', 2), (6, 'balanced', 2), (6, 'balanced', 4), (3, 'fanin', 0)]
rows = []
for chi, init, seed in cases:
    spec, teacher = s2s.setup('binary', 4, 4, chi, [1.0, 0.85, 0.7], 1000)
    cores1 = s2s.make_init(spec, teacher, init, 1.0, seed)
    th = spec.flatten(cores1)
    for amp in (1e6, 1e10, 1e14):
        t0 = time.time()
        p = s2s.correlation_flow_blowup(spec, teacher, th, amp_stop=amp)
        rows.append({'chi': chi, 'init': init, 'seed': seed, 'amp_stop': amp, 'T': p['T'], 'r1': p['r1'], 'wall': time.time() - t0})
        print(rows[-1], flush=True)
json.dump(rows, open(out, 'w'), indent=1)
