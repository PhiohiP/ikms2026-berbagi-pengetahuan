"""v3.2: data untuk gambar penjelas: lintasan waktu melewati rotasi, peta kebijakan h x beta, potret jaringan."""
import numpy as np, json, time
from multiprocessing import Pool
import model_v1 as m
TRAJ = {"rotasi saja": [(1500, 0.0, False), (3000, 0.0, True)],
        "serentak": [(1500, 0.0, False), (3000, 1.0, True)],
        "bertahap": [(1500, 1.0, False), (3000, 1.0, True)],
        "bonus dicabut saat rotasi": [(1500, 1.0, False), (3000, 0.0, True)]}
def traj_one(a):
    seed, ph = a
    tr = m.traj_phases(1.0, [{"mcs": x, "beta": b, "rotated": r} for x, b, r in ph], seed=seed)
    return tr[::10].tolist()
def grid_one(a):
    seed, h, bt = a
    st, *_ = m.run_state(h=h, beta=bt, seed=seed)
    return float(st[1])
if __name__ == "__main__":
    t0 = time.time(); R = {}
    with Pool(2) as pool:
        R["traj"] = {k: pool.map(traj_one, [(s, ph) for s in range(64)]) for k, ph in TRAJ.items()}
        print("traj", round(time.time() - t0), flush=True)
        hs = np.round(np.arange(0, 1.51, 0.1), 2); bs = np.round(np.arange(0, 1.51, 0.125), 3)
        R["grid_h"] = hs.tolist(); R["grid_b"] = bs.tolist()
        R["grid"] = [[float(np.mean(pool.map(grid_one, [(s, float(h), float(b)) for s in range(32)]))) for h in hs] for b in bs]
        print("grid", round(time.time() - t0), flush=True)
    json.dump(R, open("figures/v32_results.json", "w"))
