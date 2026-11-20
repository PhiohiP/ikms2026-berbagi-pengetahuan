"""Eksperimen v0: (1) kurva silo vs h, (2) peta fase h x bonus, (3) peta fase h x rotasi."""
import numpy as np, json, time, model
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = dict(k_in=4.0, k_out=1.0, b=12.0, K=0.5, mu=0.001)  # rezim v0 (placeholder, belum dikalibrasi)
SEEDS = 16
P.update(mcs=3000, avg_last=1000)
from multiprocessing import Pool
POOL = None
def _one(a):
    s, kw = a
    return model.run(seed=s, **P, **kw)
def avg(**kw):
    r = np.array(POOL.map(_one, [(s, kw) for s in range(SEEDS)]))
    return r.mean(0).tolist(), r.std(0).tolist()
POOL = Pool(2)
out = {"params": P, "seeds": SEEDS}; t0 = time.time()
hs = np.round(np.arange(0, 1.51, 0.1), 2)
out["curve"] = {str(h): avg(h=h) for h in hs}; print("curve", time.time()-t0, flush=True)
betas = np.round(np.arange(0, 1.51, 0.25), 2)
out["bonus"] = {f"{h}|{bt}": avg(h=h, beta=bt)[0] for h in hs for bt in betas}; print("bonus", time.time()-t0, flush=True)
rots = np.round(np.arange(0, 0.51, 0.1), 2)
out["rot"] = {f"{h}|{r}": avg(h=h, rot=r)[0] for h in hs for r in rots}; print("rot", time.time()-t0, flush=True)
json.dump(out, open("figures/v0_results.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(6, 4))
m = np.array([out["curve"][str(h)][0] for h in hs]); s = np.array([out["curve"][str(h)][1] for h in hs])
ax.errorbar(hs, m[:, 0], s[:, 0], fmt="o-", capsize=3, label="Berbagi ke kelompok sendiri")
ax.errorbar(hs, m[:, 1], s[:, 1], fmt="s-", capsize=3, label="Berbagi ke entitas lain")
ax.set_xlabel("Bias in-group, h"); ax.set_ylabel("Tingkat berbagi (fraksi tautan)")
ax.set_title("Silo muncul saat bias in-group naik (v0, %d ulangan, ±1 SD)" % SEEDS); ax.set_ylim(-0.02, 1.02)
ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig("figures/v0_silo_curve.png", dpi=150)

def phase(key, ys, ylabel, fname, title):
    Z = np.array([[out[key][f"{h}|{y}"][1] for h in hs] for y in ys])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(Z, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis",
                   extent=[hs[0]-0.05, hs[-1]+0.05, ys[0]-(ys[1]-ys[0])/2, ys[-1]+(ys[1]-ys[0])/2])
    fig.colorbar(im, label="Tingkat berbagi lintas entitas")
    ax.set_xlabel("Bias in-group, h"); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(fname, dpi=150)
phase("bonus", betas, "Bonus per berbagi lintas entitas, beta (satuan c)", "figures/v0_phase_bonus.png", "Peta fase: bias in-group x bonus (v0)")
phase("rot", rots, "Rotasi: porsi tautan dipindah lintas entitas", "figures/v0_phase_rotation.png", "Peta fase: bias in-group x rotasi (v0)")
print("done", time.time()-t0)
