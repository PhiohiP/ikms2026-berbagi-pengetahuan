"""Membuat gambar dan tabel v1 dari figures/v1_results.json."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("figures/v1_results.json")); S = R["seeds"]
COL = {"rendah": "#4C78A8", "sedang": "#F58518", "tinggi": "#B23A48"}
se = lambda sd: np.array(sd) / np.sqrt(S)

# F2 kurva silo
hs = sorted(R["curve"], key=float); x = np.array([float(h) for h in hs])
mi = np.array([R["curve"][h]["mean"][0] for h in hs]); mo = np.array([R["curve"][h]["mean"][1] for h in hs])
si = se([R["curve"][h]["sd"][0] for h in hs]); so = se([R["curve"][h]["sd"][1] for h in hs])
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.errorbar(x, mi, si, fmt="o-", color="#4C78A8", capsize=2, label="Ke rekan satu entitas")
ax.errorbar(x, mo, so, fmt="s-", color="#B23A48", capsize=2, label="Ke rekan entitas lain")
for n, h in R["scen"].items(): ax.axvline(h, color="0.75", lw=0.8, ls=":"); ax.text(h, 1.01, n, ha="center", fontsize=8, color="0.4")
ax.set_xlabel("$\\mathit{In}$-$\\mathit{group}$ $\\mathit{bias}$, $h$"); ax.set_ylabel("Tingkat berbagi"); ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", frameon=False); fig.tight_layout(); fig.savefig("figures/f2_silo.png", dpi=200)

# F3 empat tuas
LAB = {"beta": "Bonus per berbagi lintas entitas, $\\beta$ (satuan $c$)", "phi": "Penurunan biaya lewat forum, $\\phi$",
       "rot": "Rotasi: porsi tautan dipindah, $r$", "rho": "Imbalan reputasi, $\\rho$"}
TIT = {"beta": "(a) Bonus", "phi": "(b) Forum", "rot": "(c) Rotasi", "rho": "(d) Reputasi"}
fig, axs = plt.subplots(2, 2, figsize=(7, 5.4), sharey=True)
for ax, lv in zip(axs.ravel(), LAB):
    for n in R["scen"]:
        ks = sorted([k for k in R["levers"] if k.startswith(f"{n}|{lv}|")], key=lambda k: float(k.split("|")[2]))
        xv = [float(k.split("|")[2]) for k in ks]
        ax.errorbar(xv, [R["levers"][k]["mean"][1] for k in ks], se([R["levers"][k]["sd"][1] for k in ks]),
                    fmt="o-", ms=3, capsize=2, color=COL[n], label=f"bias {n}")
    ax.set_title(TIT[lv], loc="left", fontsize=10); ax.set_xlabel(LAB[lv], fontsize=8.5); ax.set_ylim(-0.02, 1.02)
axs[0, 0].set_ylabel("Berbagi lintas entitas"); axs[1, 0].set_ylabel("Berbagi lintas entitas")
axs[0, 0].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig("figures/f3_levers.png", dpi=200)

# F4 manfaat bersih per belanja (bonus vs forum)
fig, ax = plt.subplots(figsize=(6, 3.6)); table_ce = []
for n in ("sedang", "tinggi"):
    base = R["levers"][f"{n}|beta|0.0"]["mean"][2]
    for lv, mk, ls in (("beta", "o", "-"), ("phi", "s", "--")):
        ks = sorted([k for k in R["levers"] if k.startswith(f"{n}|{lv}|")], key=lambda k: float(k.split("|")[2]))[1:]
        spend = np.array([R["levers"][k]["mean"][3] for k in ks]); net = np.array([R["levers"][k]["mean"][2] for k in ks]) - base - spend
        ax.plot(spend, net, marker=mk, ls=ls, color=COL[n], label=f"{'Bonus' if lv=='beta' else 'Forum'}, bias {n}")
        for k, sp, nt in zip(ks, spend, net): table_ce.append((n, lv, float(k.split("|")[2]), sp, nt, nt / sp))
ax.axhline(0, color="0.6", lw=0.8); ax.set_xlabel("Belanja program per periode (satuan $c$)")
ax.set_ylabel("Manfaat bersih organisasi\n(tambahan surplus $-$ belanja)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/f4_cost.png", dpi=200)

# F5 NK
fig, ax = plt.subplots(figsize=(6, 3.6))
sty = {"kontrol: semua berbagi": ("0.3", ":"), "kontrol: silo penuh": ("0.6", ":"), "tanpa bias": ("#4C78A8", "-"),
       "bias tinggi": ("#B23A48", "-"), "bias tinggi + bonus 1c": ("#F58518", (0, (4, 3))), "bias tinggi + rotasi 50%": ("#6F4E7C", "-.")}
for n, v in R["nk"].items():
    c, l = sty[n]; lab = n.replace("bonus 1c", "bonus $\\beta = c$").replace("rotasi 50%", "rotasi $r = 0{,}5$")
    ax.plot(np.arange(len(v["mean"])), v["mean"], color=c, ls=l, lw=2.2 if "bonus" in n else 1.5, zorder=3 if "bonus" in n else 2, label=lab)
ax.set_xlabel("Putaran pemecahan masalah"); ax.set_ylabel("Kualitas solusi rata-rata\n(fraksi dari optimum global)")
ax.set_ylim(0.6, 1.0); ax.legend(frameon=False, fontsize=7.5, loc="lower right"); fig.tight_layout(); fig.savefig("figures/f5_nk.png", dpi=200)

# F6 sensitivitas
names = list(dict.fromkeys(k.split("|")[0] for k in R["sens"])); lv = ["tanpa", "bonus 1c", "rotasi 50%"]
fig, ax = plt.subplots(figsize=(7, 3.4)); w = 0.26; xs = np.arange(len(names))
for i, (l, c) in enumerate(zip(lv, ("0.6", "#F58518", "#6F4E7C"))):
    ax.bar(xs + (i - 1) * w, [R["sens"][f"{n}|{l}"]["mean"][1] for n in names], w,
           yerr=se([R["sens"][f"{n}|{l}"]["sd"][1] for n in names]), capsize=2, color=c,
           label={"tanpa": "tanpa kebijakan", "bonus 1c": "bonus $\\beta = c$", "rotasi 50%": "rotasi $r = 0{,}5$"}[l])
NM = {"mu = 0,01": "$\\mu$ = 0,01", "K = 0,25": "$K$ = 0,25", "K = 1": "$K$ = 1", "b/c = 10": "$b/c$ = 10", "b/c = 15": "$b/c$ = 15"}
ax.set_xticks(xs); ax.set_xticklabels([NM.get(n, n) for n in names], rotation=20, ha="right", fontsize=8); ax.set_ylabel("Berbagi lintas entitas")
ax.set_title("Uji sensitivitas pada skenario bias sedang", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig("figures/f6_sens.png", dpi=200)

# Valuasi Rupiah ilustratif
V = 9e12 / 180  # nilai rata-rata per tim inovasi APQ 2025
SC = {"konservatif": (0.05, 0.10), "moderat": (0.10, 0.20), "optimis": (0.20, 0.30)}
d_bonus = R["levers"]["tinggi|beta|1.0"]["mean"][1] - R["levers"]["tinggi|beta|0.0"]["mean"][1]
val = {k: 180 * p * d_bonus * f * V for k, (p, f) in SC.items()}
json.dump({"cost_effect": table_ce, "valuation": val, "d_bonus_tinggi": d_bonus}, open("figures/v1_derived.json", "w"), indent=1)
print("d_bonus", round(d_bonus, 3), {k: round(v / 1e9, 1) for k, v in val.items()})
for r in table_ce: print(r[0], r[1], r[2], round(r[3], 1), round(r[4], 1), round(r[5], 2))

# F1 baseline, gaya seragam
B = json.load(open("figures/baseline_results.json")); ks = sorted(B["lattice"], key=float); T = [float(k) for k in ks]
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(T, [B["lattice"][k] for k in ks], "o-", color="#4C78A8", label="Kisi persegi 100 $\\times$ 100")
ax.plot(T, [B["wellmixed"][k] for k in ks], "s--", color="0.5", label="Populasi tercampur sempurna")
ax.set_xlabel("Godaan membelot, $T$"); ax.set_ylabel("Fraksi kooperator"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/baseline_lattice.png", dpi=200)
