"""v3.1: eksperimen urutan kebijakan dengan rotasi sesungguhnya (pemindahan 20% tautan dalam kelompok)."""
import numpy as np, json, time
from multiprocessing import Pool
import model_v1 as m
from run_v3 import COND
SEEDS = 128
def one(a):
    seed, h, ph = a
    st, _ = m.run_phases_true(h, [{"mcs": x, "beta": b, "rotated": r} for x, b, r in ph], seed=seed)
    return [float(st[0]), float(st[1])]
if __name__ == "__main__":
    t0 = time.time(); R = {"seeds": SEEDS}
    with Pool(2) as pool:
        R["seq"] = {f"{n}|{k}": pool.map(one, [(s, h, ph) for s in range(SEEDS)]) for n, h in (("sedang", 0.6), ("tinggi", 1.0)) for k, ph in COND.items()}
    json.dump(R, open("figures/v31_results.json", "w")); print("done", round(time.time() - t0))
