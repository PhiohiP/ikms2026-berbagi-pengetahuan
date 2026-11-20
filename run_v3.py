"""Eksperimen v3: (1) urutan kebijakan dengan paparan insentif setara, (2) NK dengan konteks berbeda per
entitas (lam) dan daya serap (alpha). Jalankan: python run_v3.py"""
import numpy as np, json, time
from multiprocessing import Pool
import model_v1 as m

SEEDS = 128
COND = {  # nama: daftar fase (mcs, beta, rotated); rotasi selalu terjadi pada akhir 1.500 MCS pertama
    "rotasi saja":                         [(1500, 0.0, False), (3000, 0.0, True)],
    "bertahap (paparan 4.500)":            [(1500, 1.0, False), (3000, 1.0, True)],
    "serentak (paparan 3.000)":            [(1500, 0.0, False), (3000, 1.0, True)],
    "serentak (paparan 4.500)":            [(1500, 0.0, False), (4500, 1.0, True)],
    "bertahap (paparan 3.000)":            [(1500, 1.0, False), (1500, 1.0, True)],
    "bertahap, bonus dicabut (paparan 1.500)": [(1500, 1.0, False), (3000, 0.0, True)],
}
def seq_one(a):
    seed, h, ph = a
    stats, *_ = m.run_phases(h, [{"mcs": x, "beta": b, "rotated": r} for x, b, r in ph], seed=seed)
    return [float(stats[0]), float(stats[1])]

def nk_one(a):
    seed, kw, force, lam, alpha = a
    stats, strat, net, p = m.run_state(seed=seed, **kw)
    if force is not None: strat[:] = force
    rng = np.random.default_rng(20_000 + seed)
    D, T = m.make_nk_set(15, 5, 3, rng); mx = m.nk_group_max(D, T, lam, 3)
    tr = m.nk_search_het(*net, strat, D, T, mx, lam, alpha, 100, seed)
    return [float(tr[100])]

if __name__ == "__main__":
    t0 = time.time(); R = {"seeds": SEEDS}
    with Pool(2) as pool:
        R["seq"] = {f"{n}|{k}": pool.map(seq_one, [(s, h, ph) for s in range(SEEDS)])
                    for n, h in (("sedang", 0.6), ("tinggi", 1.0)) for k, ph in COND.items()}
        print("seq", round(time.time() - t0), flush=True)
        NKC = {"hasil permainan, tanpa rotasi": ({"h": 1.0}, None), "hasil permainan, rotasi 0,2": ({"h": 1.0, "rot": 0.2}, None),
               "semua berbagi": ({"h": 1.0}, 3), "silo penuh": ({"h": 1.0}, 1)}
        R["nk"] = {f"{lam}|{al}|{n}": pool.map(nk_one, [(s, kw, f, lam, al) for s in range(32)])
                   for lam in (0.0, 0.5) for al in (1.0, 0.5) for n, (kw, f) in NKC.items()}
        print("nk", round(time.time() - t0), flush=True)
    json.dump(R, open("figures/v3_results.json", "w"))
