"""v3.3: lintasan waktu dengan 128 ulangan (sama dengan Tabel 5)."""
import json, time
from multiprocessing import Pool
import model_v1 as m
from run_v32 import TRAJ, traj_one
if __name__ == "__main__":
    t0=time.time(); R=json.load(open("figures/v32_results.json"))
    with Pool(2) as pool:
        R["traj"]={k: pool.map(traj_one,[(s,ph) for s in range(128)]) for k,ph in TRAJ.items()}
    json.dump(R,open("figures/v33_results.json","w")); print("done",round(time.time()-t0))
