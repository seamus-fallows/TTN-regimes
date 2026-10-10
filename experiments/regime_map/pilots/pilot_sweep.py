"""Throwaway: run pilot_gf over a small sigma x chi grid in parallel, print summaries."""
import itertools
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "experiments/regime_map/pilots")
from pilot_gf import events, run  # noqa: E402


def job(args):
    kind, n, chi, sigma, seed, construction = args
    try:
        r = run(kind=kind, n=n, chi=chi, sigma=sigma, seed=seed, cap=(n > 4), construction=construction)
    except Exception as e:  # noqa: BLE001
        return args, None, repr(e)
    ev = sorted(events(r["t"], r["A"]))
    rho = [(ev[i + 1][0] - ev[i][0]) / max(ev[i][1], ev[i + 1][1]) for i in range(len(ev) - 1)]
    s = dict(Lfin=r["L"][-1] / r["L"][0], dK=r["dK"], dth=r["dth"], ev=ev, rho=rho, wall=r["wall"],
             nfev=r["nfev"], K=r["K"], t=r["t"], L=r["L"], A=r["A"])
    return args, s, None


if __name__ == "__main__":
    kind, n = sys.argv[1], int(sys.argv[2])
    chis = [int(x) for x in sys.argv[3].split(",")]
    sigmas = [float(x) for x in sys.argv[4].split(",")]
    seeds = range(int(sys.argv[5]))
    out = sys.argv[6]
    construction = sys.argv[7] if len(sys.argv) > 7 else "centred"
    grid = list(itertools.product([kind], [n], chis, sigmas, seeds, [construction]))
    with Pool(min(len(grid), 300)) as p:
        res = p.map(job, grid, chunksize=1)
    pickle.dump(res, open(out, "wb"))
    for args, s, err in res:
        if s is None:
            print(args, "ERR", err)
            continue
        print(f"{args} K={s['K']} wall={s['wall']:.0f}s Lfin={s['Lfin']:.1e} dK={s['dK']:.3g} "
              f"dth={s['dth']:.3g} t_ev={[f'{e[0]:.3g}' for e in s['ev']]} rho={[f'{x:.2f}' for x in s['rho']]}")
