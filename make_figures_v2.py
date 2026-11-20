"""Gambar, tabel, dan angka turunan v2 dari figures/v2_results.json (selang kepercayaan bootstrap 95%)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("figures/v2_results.json")); rng = np.random.default_rng(0)
from matplotlib.ticker import FuncFormatter, FixedFormatter
_fmt = FuncFormatter(lambda v, p: ("%g" % round(v, 6)).replace(".", ","))
_orig_save = plt.Figure.savefig
def _save_koma(self, *a, **k):
    """Desimal koma pada semua sumbu numerik (sumbu kategori dibiarkan)."""
    for ax in self.axes:
        for axis in ((ax.yaxis,) if getattr(ax, "_kat", False) else (ax.xaxis, ax.yaxis)):
            if not isinstance(axis.get_major_formatter(), FixedFormatter):
                axis.set_major_formatter(_fmt)
    return _orig_save(self, *a, **k)
plt.Figure.savefig = _save_koma
COL = {"rendah": "#4C78A8", "sedang": "#F58518", "tinggi": "#B23A48"}
IG = "$\\mathit{In}$-$\\mathit{group}$ $\\mathit{bias}$, $h$"

def boot(x, f=np.mean, B=2000):
    x = np.asarray(x); idx = rng.integers(0, len(x), (B, len(x))); s = f(x[idx], axis=1)
    return float(f(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
def stat(runs, col=1):
    return boot([r[col] for r in runs])
def high(runs, thr=0.5):
    return boot([1.0 if r[1] > thr else 0.0 for r in runs])
D = {}

# Kurva silo + porsi kondisi berbagi tinggi
hs = sorted(R["curve"], key=float); x = np.array([float(h) for h in hs])
cin = np.array([stat(R["curve"][h], 0) for h in hs]); cout = np.array([stat(R["curve"][h], 1) for h in hs])
hf = np.array([high(R["curve"][h]) for h in hs])
fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
for arr, c, lab, mk in ((cin, "#4C78A8", "Ke rekan satu kelompok", "o"), (cout, "#B23A48", "Ke rekan kelompok lain", "s")):
    axs[0].plot(x, arr[:, 0], marker=mk, color=c, label=lab); axs[0].fill_between(x, arr[:, 1], arr[:, 2], color=c, alpha=0.18)
axs[0].set_xlabel(IG); axs[0].set_ylabel("Tingkat berbagi"); axs[0].set_ylim(0, 1); axs[0].legend(frameon=False, fontsize=8, loc="lower left")
axs[0].set_title("(a) Rata-rata tingkat berbagi", loc="left", fontsize=10)
axs[1].plot(x, hf[:, 0], "o-", color="#B23A48"); axs[1].fill_between(x, hf[:, 1], hf[:, 2], color="#B23A48", alpha=0.18)
axs[1].axhline(0.5, color="0.6", lw=0.8, ls=":"); axs[1].set_ylim(0, 1); axs[1].set_xlabel(IG)
axs[1].set_ylabel("Porsi ulangan berbagi\nlintas kelompok > 0,5"); axs[1].set_title("(b) Peluang kondisi tidak silo", loc="left", fontsize=10)
for ax in axs:
    for n, h in R["scen"].items(): ax.axvline(h, color="0.8", lw=0.8, ls=":")
fig.tight_layout(); fig.savefig("figures/v2_f2_silo.png", dpi=200)
# ambang h*: interpolasi linear porsi tinggi = 0,5
hstar = None
for i in range(len(x) - 1):
    if hf[i, 0] >= 0.5 > hf[i + 1, 0]:
        hstar = x[i] + (hf[i, 0] - 0.5) / (hf[i, 0] - hf[i + 1, 0]) * (x[i + 1] - x[i]); break
D["hstar"] = hstar; D["curve_in_range"] = [float(cin[:, 0].min()), float(cin[:, 0].max())]
D["curve"] = {h: {"in": cin[i].tolist(), "out": cout[i].tolist(), "high": hf[i].tolist()} for i, h in enumerate(hs)}

# Konvergensi
D["conv"] = {k: {"out": stat(v), "high": high(v)} for k, v in R["conv"].items()}

# Empat tuas
LAB = {"beta": "Bonus, $\\beta$ (satuan $c$)", "phi": "Penurunan biaya lewat forum, $\\phi$",
       "rot": "Rotasi, porsi tautan dipindah $r$", "rho": "Imbalan reputasi, $\\rho$"}
TIT = {"beta": "(a) Bonus", "phi": "(b) Forum", "rot": "(c) Rotasi", "rho": "(d) Reputasi"}
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=True); D["levers"] = {}
for ax, lv in zip(axs.ravel(), LAB):
    for n in R["scen"]:
        ks = sorted([k for k in R["levers"] if k.startswith(f"{n}|{lv}|")], key=lambda k: float(k.split("|")[2]))
        xv = [float(k.split("|")[2]) for k in ks]; st = np.array([stat(R["levers"][k]) for k in ks])
        ax.plot(xv, st[:, 0], "o-", ms=3, color=COL[n], label=f"bias {n}"); ax.fill_between(xv, st[:, 1], st[:, 2], color=COL[n], alpha=0.15)
        for k, s in zip(ks, st): D["levers"][k] = {"out": s.tolist(), "in": stat(R["levers"][k], 0), "high": high(R["levers"][k]),
                                                 "sur": float(np.mean([r[2] for r in R["levers"][k]])), "spend": float(np.mean([r[3] for r in R["levers"][k]]))}
    if lv == "rot": ax.axvspan(0, 0.2, color="0.92", zorder=0); ax.text(0.1, 0.95, "rentang realistis", ha="center", fontsize=7.5, color="0.4")
    ax.set_title(TIT[lv], loc="left", fontsize=10); ax.set_xlabel(LAB[lv], fontsize=8.5); ax.set_ylim(0, 1)
axs[0, 0].set_ylabel("Berbagi lintas kelompok"); axs[1, 0].set_ylabel("Berbagi lintas kelompok"); axs[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/v2_f3_levers.png", dpi=200)

# Kurva biaya: berbagi lintas entitas vs belanja (bonus vs forum)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
for ax, n in zip(axs, ("sedang", "tinggi")):
    for lv, mk, ls, lab in (("beta", "o", "-", "Bonus"), ("phi", "s", "--", "Forum")):
        ks = sorted([k for k in R["levers"] if k.startswith(f"{n}|{lv}|")], key=lambda k: float(k.split("|")[2]))
        ax.plot([D["levers"][k]["spend"] for k in ks], [D["levers"][k]["out"][0] for k in ks], marker=mk, ls=ls, color=COL[n] if lv == "beta" else "0.45", label=lab)
    ax.set_title(f"Bias {n}", loc="left", fontsize=10); ax.set_xlabel("Belanja program per periode (satuan $c$)"); ax.set_ylim(0, 1)
axs[0].set_ylabel("Berbagi lintas kelompok"); axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/v2_f4_cost.png", dpi=200)
def at_spend(n, lv, S):
    ks = sorted([k for k in R["levers"] if k.startswith(f"{n}|{lv}|")], key=lambda k: float(k.split("|")[2]))
    sp = np.array([D["levers"][k]["spend"] for k in ks]); y = np.array([D["levers"][k]["out"][0] for k in ks])
    return float(np.interp(S, sp, y)) if S <= sp.max() else None
D["equal_spend"] = {f"{n}|{S}": {"bonus": at_spend(n, "beta", S), "forum": at_spend(n, "phi", S)} for n in ("sedang", "tinggi") for S in (100, 200, 400)}

# Sensitivitas dan sapuan b/c
D["sens"] = {k: {"out": stat(v), "high": high(v)} for k, v in R["sens"].items()}
D["bc"] = {k: {"out": stat(v), "in": stat(v, 0)} for k, v in R["bc"].items()}
names = list(dict.fromkeys(k.split("|")[0] for k in R["sens"])); LV = ["tanpa", "bonus", "reputasi", "forum", "rotasi"]
LL = {"tanpa": "tanpa kebijakan", "bonus": "bonus $\\beta = 0{,}6c$", "reputasi": "reputasi $\\rho b = 0{,}6c$", "forum": "forum $\\phi = 0{,}3$", "rotasi": "rotasi $r = 0{,}2$"}
CC = {"tanpa": "0.65", "bonus": "#F58518", "reputasi": "#E9C46A", "forum": "#4C78A8", "rotasi": "#6F4E7C"}
NM = {"mu = 0,01": "$\\mu$ = 0,01", "K = 0,25": "$K$ = 0,25", "K = 1": "$K$ = 1", "b/c = 10": "$b/c$ = 10", "b/c = 15": "$b/c$ = 15"}
fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.16; xs = np.arange(len(names))
for i, l in enumerate(LV):
    st = np.array([D["sens"][f"{n}|{l}"]["out"] for n in names])
    ax.bar(xs + (i - 2) * w, st[:, 0], w, yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], capsize=1.5, color=CC[l], label=LL[l])
ax._kat = True; ax.set_xticks(xs); ax.set_xticklabels([NM.get(n, n) for n in names], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Berbagi lintas kelompok"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout(); fig.savefig("figures/v2_f6_sens.png", dpi=200)

# Urutan kebijakan
D["seq"] = {k: {"out": stat(v), "in": stat(v, 0), "high": high(v)} for k, v in R["seq"].items()}
# NK 2x2
D["nk"] = {k: {"t10": boot([r[0] for r in v]), "t100": boot([r[1] for r in v])} for k, v in R["nk"].items()}

# Valuasi ilustratif (batas atas nilai per gugus)
V = 9e12 / 180
SC = {"konservatif": (0.05, 0.10), "moderat": (0.10, 0.20), "optimis": (0.20, 0.30)}
# Delta s terhadap rata-rata kondisi tanpa kebijakan pada h = 0,9; 1,0; 1,1 (mengurangi derau satu titik)
base = np.mean([D["curve"][h]["out"][0] for h in ("0.9", "1.0", "1.1")])
d = D["levers"]["tinggi|beta|1.0"]["out"][0] - base
d_range = [D["levers"]["tinggi|beta|1.0"]["out"][0] - D["curve"][h]["out"][0] for h in ("0.9", "1.0", "1.1")]
D["valuation"] = {"base": base, "d": d, "d_range": [min(d_range), max(d_range)], **{k: 180 * p * d * f * V for k, (p, f) in SC.items()}}
json.dump(D, open("figures/v2_derived.json", "w"), indent=1)
print(json.dumps({k: D[k] for k in ("hstar", "curve_in_range", "valuation", "equal_spend")}, indent=1))

# Gambar urutan kebijakan
order = ["tanpa kebijakan", "rotasi saja", "bonus saja", "bonus dan rotasi serentak", "bonus dulu, lalu bonus + rotasi", "bonus dulu, lalu rotasi tanpa bonus"]
short = ["tanpa\nkebijakan", "rotasi\nsaja", "bonus\nsaja", "bonus +\nrotasi\nserentak", "bonus dulu,\nlalu bonus\n+ rotasi", "bonus dulu,\nlalu rotasi,\nbonus dicabut"]
fig, ax = plt.subplots(figsize=(8, 3.4)); xs = np.arange(len(order)); w = 0.38
for i, n in enumerate(("sedang", "tinggi")):
    st = np.array([D["seq"][f"{n}|{k}"]["out"] for k in order])
    ax.bar(xs + (i - 0.5) * w, st[:, 0], w, yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], capsize=2, color=COL[n], label=f"bias {n}")
ax._kat = True; ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=8); ax.set_ylabel("Berbagi lintas kelompok"); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig("figures/v2_f7_seq.png", dpi=200)

# Gambar acuan (replikasi), gaya seragam dan desimal koma
B = json.load(open("figures/baseline_results.json")); ks = sorted(B["lattice"], key=float); T = [float(k) for k in ks]
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(T, [B["lattice"][k] for k in ks], "o-", color="#4C78A8", label="Kisi persegi 100 $\\times$ 100")
ax.plot(T, [B["wellmixed"][k] for k in ks], "s--", color="0.5", label="Populasi tercampur sempurna")
ax.set_xlabel("Godaan membelot, $T$"); ax.set_ylabel("Fraksi kooperator"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/baseline_lattice.png", dpi=200)
