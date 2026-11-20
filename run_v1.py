"""Eksperimen v1. Jalankan: python run_v1.py  (sekitar 20 menit dengan 2 inti)."""
import numpy as np, json, time, sys
from multiprocessing import Pool
import model_v1 as m

SEEDS = 16
SCEN = {"rendah": 0.3, "sedang": 0.6, "tinggi": 1.0}   # skenario h (placeholder; kalibrasi ke literatur di paper)
LEVERS = {"beta": np.round(np.arange(0, 1.51, 0.25), 3), "phi": np.round(np.arange(0, 0.61, 0.1), 3),
          "rot": np.round(np.arange(0, 0.51, 0.1), 3), "rho": np.round(np.arange(0, 0.151, 0.025), 3)}

def one(args):
    seed, kw = args
    stats, strat, net, p = m.run_state(seed=seed, **kw)
    sur, spend = m.welfare(stats, p["b"], p["c"], kw.get("beta", 0.0), kw.get("phi", 0.0))
    return [stats[0], stats[1], sur, spend]

def batch(pool, kw):
    r = np.array(pool.map(one, [(s, kw) for s in range(SEEDS)]))
    return {"mean": r.mean(0).tolist(), "sd": r.std(0).tolist(), "high_frac": float((r[:, 1] > 0.5).mean())}

def nk_one(args):
    seed, kw, force = args
    stats, strat, net, p = m.run_state(seed=seed, **kw)
    if force is not None: strat[:] = force
    rng = np.random.default_rng(10_000 + seed)
    deps, table = m.make_nk(15, 5, rng); mx = m.nk_max(deps, table)
    tr = m.nk_search(*net, strat, deps, table, 100, seed) / mx
    return tr.tolist(), float(stats[1])

if __name__ == "__main__":
    t0 = time.time(); R = {"seeds": SEEDS, "scen": SCEN, "params": m.DEFAULT}
    with Pool(2) as pool:
        # E0: cek konvergensi
        R["conv"] = {f"{n}|{T}": batch(pool, dict(h=h, mcs=T, avg_last=1000)) for n, h in SCEN.items() for T in (3000, 6000)}
        print("conv", round(time.time() - t0), flush=True)
        # Kurva silo
        hs = np.round(np.arange(0, 1.51, 0.1), 2)
        R["curve"] = {str(h): batch(pool, dict(h=h)) for h in hs}
        print("curve", round(time.time() - t0), flush=True)
        # E1: skenario x tuas
        R["levers"] = {}
        for n, h in SCEN.items():
            for lv, vals in LEVERS.items():
                for v in vals:
                    R["levers"][f"{n}|{lv}|{v}"] = batch(pool, {"h": h, lv: float(v)})
            print("levers", n, round(time.time() - t0), flush=True)
        # E2: sensitivitas pada skenario sedang
        sens = {"dasar": {}, "ukuran 150/100/50": {"sizes": (150, 100, 50)}, "derajat 6/1,5": {"k_in": 6.0, "k_out": 1.5},
                "b/c = 10": {"b": 10.0}, "b/c = 15": {"b": 15.0}, "K = 0,25": {"K": 0.25}, "K = 1": {"K": 1.0},
                "mu = 0,01": {"mu": 0.01}}
        R["sens"] = {}
        for name, extra in sens.items():
            for lvname, lv in (("tanpa", {}), ("bonus 1c", {"beta": 1.0}), ("rotasi 50%", {"rot": 0.5})):
                R["sens"][f"{name}|{lvname}"] = batch(pool, {"h": SCEN["sedang"], **extra, **lv})
        print("sens", round(time.time() - t0), flush=True)
        # E3: lapisan NK
        conds = {"tanpa bias": ({"h": 0.0}, None), "bias tinggi": ({"h": 1.0}, None),
                 "bias tinggi + bonus 1c": ({"h": 1.0, "beta": 1.0}, None),
                 "bias tinggi + rotasi 50%": ({"h": 1.0, "rot": 0.5}, None),
                 "kontrol: semua berbagi": ({"h": 0.0}, 3), "kontrol: silo penuh": ({"h": 0.0}, 1)}
        R["nk"] = {}
        for name, (kw, force) in conds.items():
            res = pool.map(nk_one, [(s, kw, force) for s in range(20)])
            tr = np.array([r[0] for r in res])
            R["nk"][name] = {"mean": tr.mean(0).tolist(), "sd": tr.std(0).tolist(), "out": float(np.mean([r[1] for r in res]))}
        print("nk", round(time.time() - t0), flush=True)
    json.dump(R, open("figures/v1_results.json", "w"), indent=1)
    print("done", round(time.time() - t0))
