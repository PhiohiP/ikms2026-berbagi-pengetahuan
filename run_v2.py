"""Eksperimen v2 (perbaikan dari judge analysis). Jalankan: python run_v2.py
Perubahan dari v1: 64 ulangan, data mentah per ulangan disimpan (untuk selang kepercayaan bootstrap),
rotasi realistis (r <= 0,2) plus r = 0,3 dan 0,5 sebagai pembanding, reputasi masuk uji sensitivitas,
sapuan b/c, eksperimen urutan kebijakan, dan desain 2 x 2 untuk RQ3 pada NK landscape K = 5 dan 10."""
import numpy as np, json, time
from multiprocessing import Pool
import model_v1 as m

SEEDS = 64
SCEN = {"rendah": 0.3, "sedang": 0.6, "tinggi": 1.0}
LEVERS = {"beta": [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], "phi": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
          "rot": [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5], "rho": [0, 0.025, 0.05, 0.075, 0.1, 0.125]}

def one(args):
    seed, kw = args
    stats, strat, net, p = m.run_state(seed=seed, **kw)
    sur, spend = m.welfare(stats, p["b"], p["c"], kw.get("beta", 0.0), kw.get("phi", 0.0))
    return [float(stats[0]), float(stats[1]), float(sur), float(spend)]

def seq_one(args):
    seed, h, p1, p2 = args
    stats, strat, net, p = m.run_sequence(h, p1, p2, seed=seed)
    return [float(stats[0]), float(stats[1])]

def batch(pool, kw, fn=one, extra=None):
    args = [(s, kw) for s in range(SEEDS)] if extra is None else [(s, *extra) for s in range(SEEDS)]
    return pool.map(fn, args)

def nk_one(args):
    seed, kw, force, Knk = args
    stats, strat, net, p = m.run_state(seed=seed, **kw)
    if force is not None: strat[:] = force
    rng = np.random.default_rng(10_000 + seed)
    deps, table = m.make_nk(15, Knk, rng); mx = m.nk_max(deps, table)
    tr = m.nk_search(*net, strat, deps, table, 100, seed) / mx
    return [float(tr[10]), float(tr[100])]

if __name__ == "__main__":
    t0 = time.time(); R = {"seeds": SEEDS, "scen": SCEN, "params": m.DEFAULT}
    with Pool(2) as pool:
        R["conv"] = {f"{n}|{T}": batch(pool, dict(h=h, mcs=T, avg_last=1000)) for n, h in SCEN.items() for T in (3000, 6000)}
        print("conv", round(time.time() - t0), flush=True)
        R["curve"] = {str(h): batch(pool, dict(h=float(h))) for h in np.round(np.arange(0, 1.51, 0.1), 2)}
        print("curve", round(time.time() - t0), flush=True)
        R["levers"] = {f"{n}|{lv}|{v}": batch(pool, {"h": h, lv: float(v)}) for n, h in SCEN.items() for lv, vals in LEVERS.items() for v in vals}
        print("levers", round(time.time() - t0), flush=True)
        sens = {"dasar": {}, "ukuran 150/100/50": {"sizes": (150, 100, 50)}, "derajat 6/1,5": {"k_in": 6.0, "k_out": 1.5},
                "b/c = 10": {"b": 10.0}, "b/c = 15": {"b": 15.0}, "K = 0,25": {"K": 0.25}, "K = 1": {"K": 1.0}, "mu = 0,01": {"mu": 0.01}}
        LV = {"tanpa": {}, "bonus": {"beta": 0.6}, "reputasi": {"rho": 0.05}, "forum": {"phi": 0.3}, "rotasi": {"rot": 0.2}}
        R["sens"] = {f"{n}|{l}": batch(pool, {"h": SCEN["sedang"], **e, **lv}) for n, e in sens.items() for l, lv in LV.items()}
        print("sens", round(time.time() - t0), flush=True)
        R["bc"] = {f"{bc}|{l}": batch(pool, {"h": SCEN["sedang"], "b": float(bc), **lv}) for bc in (8, 10, 12, 15, 20) for l, lv in LV.items()}
        print("bc", round(time.time() - t0), flush=True)
        SEQ = {"tanpa kebijakan": ({}, {}), "rotasi saja": ({}, {"rot": 0.2}), "bonus saja": ({"beta": 1.0}, {"beta": 1.0}),
               "bonus dan rotasi serentak": ({}, {"beta": 1.0, "rot": 0.2}), "bonus dulu, lalu bonus + rotasi": ({"beta": 1.0}, {"beta": 1.0, "rot": 0.2}),
               "bonus dulu, lalu rotasi tanpa bonus": ({"beta": 1.0}, {"rot": 0.2})}
        R["seq"] = {f"{n}|{k}": batch(pool, None, seq_one, (h, p1, p2)) for n, h in (("sedang", 0.6), ("tinggi", 1.0)) for k, (p1, p2) in SEQ.items()}
        print("seq", round(time.time() - t0), flush=True)
        COND = {"hasil permainan, tanpa rotasi": ({"h": 1.0}, None), "hasil permainan, rotasi 0,2": ({"h": 1.0, "rot": 0.2}, None),
                "semua berbagi, tanpa rotasi": ({"h": 1.0}, 3), "semua berbagi, rotasi 0,2": ({"h": 1.0, "rot": 0.2}, 3),
                "silo penuh, tanpa rotasi": ({"h": 1.0}, 1)}
        R["nk"] = {f"{Knk}|{n}": pool.map(nk_one, [(s, kw, f, Knk) for s in range(32)]) for Knk in (5, 10) for n, (kw, f) in COND.items()}
        print("nk", round(time.time() - t0), flush=True)
    json.dump(R, open("figures/v2_results.json", "w"))
    print("done", round(time.time() - t0))
