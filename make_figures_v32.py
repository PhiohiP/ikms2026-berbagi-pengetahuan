"""Gambar penjelas v3.2: potret jaringan, lintasan waktu melewati rotasi, peta kebijakan h x beta."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedFormatter
from matplotlib.lines import Line2D
import model_v1 as m
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
_fmt = FuncFormatter(lambda v, p: ("%g" % round(v, 6)).replace(".", ","))
_orig_save = plt.Figure.savefig
def _save_koma(self, *a, **k):
    for ax in self.axes:
        if getattr(ax, "_nofmt", False): continue
        for axis in (ax.xaxis, ax.yaxis):
            if not isinstance(axis.get_major_formatter(), FixedFormatter):
                axis.set_major_formatter(_fmt)
    return _orig_save(self, *a, **k)
plt.Figure.savefig = _save_koma
rng = np.random.default_rng(0)
IT = lambda s: "$\\mathit{" + s + "}$"

# ---------- (1) potret jaringan ----------
SCOL = {0: "#BDBDBD", 1: "#4C78A8", 2: "#F58518", 3: "#2E8B57"}   # H, I, O, S
SNAME = {0: "H (menahan)", 1: "I (ke kelompok sendiri)", 2: "O (ke kelompok lain)", 3: "S (ke semua)"}
def snapshot(ax, h, beta, seed, title):
    st, strat, (ip, ix, g), _ = m.run_state(h=h, beta=beta, seed=seed)
    r = np.random.default_rng(42); n = len(g)
    cen = {0: (0, 1.0), 1: (-0.95, -0.55), 2: (0.95, -0.55)}
    ang = r.uniform(0, 2 * np.pi, n); rad = 0.52 * np.sqrt(r.uniform(0, 1, n))
    pos = np.array([[cen[g[i]][0] + rad[i] * np.cos(ang[i]), cen[g[i]][1] + rad[i] * np.sin(ang[i])] for i in range(n)])
    for i in range(n):
        for j in ix[ip[i]:ip[i + 1]]:
            if j <= i: continue
            if g[i] == g[j]:
                act = (strat[i] & 1) or (strat[j] & 1)
                ax.plot(*pos[[i, j]].T, color="#4C78A8" if act else "#E0E0E0", lw=0.35 if act else 0.25, alpha=0.5, zorder=1)
            else:
                act = (strat[i] & 2) or (strat[j] & 2)
                ax.plot(*pos[[i, j]].T, color="#B23A48" if act else "#D0D0D0", lw=0.7 if act else 0.4,
                        alpha=0.85 if act else 0.5, ls="-" if act else (0, (2, 2)), zorder=2)
    ax.scatter(pos[:, 0], pos[:, 1], s=9, c=[SCOL[s] for s in strat], edgecolors="white", linewidths=0.2, zorder=3)
    for k, nm in {0: "eks-KPI", 1: "eks-PIS", 2: "Patra Niaga"}.items():
        x, y = cen[k]; ax.text(x + (0.62 if k == 0 else 0), y + (0.35 if k == 0 else -0.66), nm, ha="left" if k == 0 else "center", fontsize=10)
    ax.set_title(title + "\n" + ("dalam kelompok %.2f; lintas kelompok %.2f" % (st[0], st[1])).replace(".", ","), fontsize=10.5); ax.set_aspect("equal"); ax.axis("off"); ax._nofmt = True
    return st

fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.9))
s0 = snapshot(axs[0], 1.0, 0.0, 3, "(a) " + IT("Bias") + " tinggi, tanpa kebijakan")
s1 = snapshot(axs[1], 1.0, 1.0, 3, "(b) " + IT("Bias") + " tinggi, bonus $\\beta = c$")
hd = [Line2D([], [], marker="o", ls="", color=SCOL[k], label=SNAME[k], markersize=5) for k in range(4)]
hd += [Line2D([], [], color="#B23A48", lw=1.2, label="tautan lintas kelompok yang dipakai berbagi"),
       Line2D([], [], color="#D0D0D0", lw=1, ls=(0, (2, 2)), label="tautan lintas kelompok yang menganggur")]
fig.legend(handles=hd, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
fig.tight_layout(rect=(0, 0.13, 1, 0.94)); fig.savefig("figures/v32_f1_jaringan.png", dpi=220)
print("jaringan", s0, s1)
if len(sys.argv) > 1 and sys.argv[1] == "net": sys.exit()

R = json.load(open("figures/v33_results.json"))  # lintasan 128 ulangan; peta dari v3.2
# ---------- (2) lintasan waktu ----------
LBL = {"rotasi saja": "rotasi saja", "serentak": "serentak (bonus + rotasi bersamaan)",
       "bertahap": "bertahap (bonus dulu, lalu rotasi)", "bonus dicabut saat rotasi": "bonus dicabut saat rotasi dimulai"}
CC = {"rotasi saja": "#7F7F7F", "serentak": "#F58518", "bertahap": "#2E8B57", "bonus dicabut saat rotasi": "#B23A48"}
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
ts = None
summ = {}
for k, runs in R["traj"].items():
    A = np.array(runs)                     # (seed, t, 2)
    ts = np.arange(A.shape[1]) * 10
    for col, ax in ((1, axs[0]), (0, axs[1])):
        Y = A[:, :, col]; mu = Y.mean(0)
        bs = np.array([Y[rng.integers(0, len(Y), len(Y))].mean(0) for _ in range(300)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        ax.fill_between(ts, lo, hi, color=CC[k], alpha=0.18, lw=0)
        ax.plot(ts, mu, color=CC[k], lw=1.4, label=LBL[k])
    summ[k] = [float(A[:, -100:, 1].mean()), float(A[:, -100:, 0].mean())]
for ax, t in zip(axs, ("(a) Berbagi lintas kelompok", "(b) Berbagi dalam kelompok")):
    ax.axvline(1500, color="k", lw=0.8, ls="--")
    ax.text(1450, 0.03, "rotasi dimulai", fontsize=8.5, rotation=90, ha="right", va="bottom")
    ax.text(750, 0.99, "fase 1", fontsize=8.5, ha="center", va="top", color="#555")
    ax.text(3000, 0.99, "fase 2: 20% tautan dirotasi", fontsize=8.5, ha="center", va="top", color="#555")
    ax.set_title(t, fontsize=10.5); ax.set_xlabel("Waktu (MCS)"); ax.set_ylim(0, 1); ax.set_xlim(0, 4500); ax.set_xticks([0, 1500, 3000, 4500]); ax.xaxis.set_major_formatter(FixedFormatter(["0", "1.500", "3.000", "4.500"]))
axs[0].set_ylabel("Fraksi agen yang berbagi")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig("figures/v32_f2_lintasan.png", dpi=220)
print("lintasan akhir [out,in]", summ)

# ---------- (3) peta kebijakan ----------
H = np.array(R["grid_h"]); B = np.array(R["grid_b"]); G = np.array(R["grid"])
fig, ax = plt.subplots(figsize=(5.6, 4.0))
dh, db = H[1] - H[0], B[1] - B[0]
im = ax.imshow(G, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
               extent=(H[0] - dh / 2, H[-1] + dh / 2, B[0] - db / 2, B[-1] + db / 2))
from scipy.ndimage import gaussian_filter
cs = ax.contour(H, B, gaussian_filter(G, 1.0, mode="nearest"), levels=[0.3, 0.5, 0.7], colors="white", linewidths=0.8)
ax.clabel(cs, fmt=lambda v: ("%.1f" % v).replace(".", ","), fontsize=7.5)
xx = np.linspace(H[0], H[-1], 50)
ax.plot(xx, xx * 1.0, color="#FF6F61", lw=1.6, ls="--", label="$\\beta = hc$ (bias dinetralkan)")
for hv, nm in ((0.3, "rendah"), (0.6, "sedang"), (1.0, "tinggi")):
    ax.axvline(hv, color="white", lw=0.5, ls=":"); ax.text(hv, B[-1] + db * 0.7, nm, ha="center", fontsize=7.5)
ax.set_xlim(H[0] - dh / 2, H[-1] + dh / 2); ax.set_ylim(B[0] - db / 2, B[-1] + db / 2)
ax.set_xlabel(IT("In") + "-" + IT("group") + " " + IT("bias") + ", $h$"); ax.set_ylabel("Bonus lintas kelompok, $\\beta$ (dalam satuan $c$)")
ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor="white")
cb = fig.colorbar(im, ax=ax); cb.set_label("Berbagi lintas kelompok"); cb.ax.yaxis.set_major_formatter(_fmt)
fig.tight_layout(); fig.savefig("figures/v32_f3_peta.png", dpi=220)
# angka turunan: berbagi pada garis beta = h
diag = {float(h): float(G[np.argmin(abs(B - h)), i]) for i, h in enumerate(H) if h <= B[-1]}
json.dump({"traj_end": summ, "diag": diag, "corner": {"h0b0": G[0, 0], "h1b0": G[0, list(H).index(1.0)]}},
          open("figures/v32_derived.json", "w"), indent=1)
print("diag", diag)
