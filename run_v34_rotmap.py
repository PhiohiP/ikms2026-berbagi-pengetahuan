"""v3.4 (baru, untuk poster): peta efek rotasi terhadap in-group bias.
Rotasi dimodelkan seperti eksperimen instrumen v2 (jaringan statis berderajat hasil rotasi),
64 ulangan per sel dengan bibit 0..63 (sama dengan v2), sehingga sel h = 0,3 / 0,6 / 1,0 dan
r = 0,2 dapat dicocokkan dengan Gambar instrumen di paper."""
import json, time, numpy as np
from multiprocessing import Pool
import model_v1 as m

HS = [round(x, 2) for x in np.arange(0, 1.51, 0.1)]
RS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
SEEDS = list(range(64))

def one(a):
    seed, h, r = a
    st, *_ = m.run_state(h=h, rot=r, seed=seed)
    return [float(st[0]), float(st[1])]   # dalam, lintas

if __name__ == "__main__":
    t0 = time.time(); R = {"h": HS, "r": RS, "seeds": SEEDS, "cell": {}}
    with Pool(2) as pool:
        for h in HS:
            for r in RS:
                R["cell"][f"{h}|{r}"] = pool.map(one, [(s, h, r) for s in SEEDS])
            print("h", h, round(time.time() - t0), flush=True)
    json.dump(R, open("figures/v34_rotmap.json", "w"))
