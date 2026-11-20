"""Gambar poster IKMS 2026 (versi akhir, palet Navy Pertamina).

Membaca hasil simulasi di ../figures/*.json, menulis PDF dan PNG ke poster/figs_poster/.
Gaya dasar: publication.mplstyle dari K-Dense scientific-agent-skills (MIT), diskalakan ke ukuran cetak A0.
Selang kepercayaan: bootstrap persentil 95% dengan bibit tetap; tidak ada nilai data yang diubah.

Pemakaian (dari folder poster/):  python make_figs_poster.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..")
sys.path.insert(0, SRC)
FIG = os.path.join(SRC, "figures")
OUT = os.path.join(HERE, "figs_poster"); os.makedirs(OUT, exist_ok=True)
plt.style.use(os.path.join(HERE, "publication.mplstyle"))

W = 14.23                     # lebar teks di dalam kartu kolom poster (inci)
INK, RED, GREY, LIGHT = "#0E2A47", "#C8102E", "#56677D", "#B4C1D0"   # palet Navy Pertamina, diaudit dengan palette_audit.py
plt.rcParams.update({
    "font.family": "Fira Sans", "font.size": 21, "axes.labelsize": 22, "xtick.labelsize": 20,
    "ytick.labelsize": 20, "legend.fontsize": 20, "axes.linewidth": 1.2, "xtick.major.width": 1.2,
    "ytick.major.width": 1.2, "xtick.major.size": 6, "ytick.major.size": 6, "lines.linewidth": 3.0,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "pdf.fonttype": 42})
koma = FuncFormatter(lambda v, p: ("%g" % round(v, 6)).replace(".", ",").replace("-", "−"))
rng = np.random.default_rng(20261120)

def boot(x, B=4000):
    x = np.asarray(x, float); s = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)

def panel(ax, letter):
    ax.text(-0.11, 1.04, letter, transform=ax.transAxes, fontsize=24, fontweight="bold", va="bottom")

def center(fig):
    """Geser semua sumbu mendatar agar isi figur (termasuk label) seimbang kiri-kanan."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(r); wfig = fig.get_figwidth()
    dx = ((wfig - bb.x1) - bb.x0) / 2 / wfig
    for ax in fig.axes:
        p = ax.get_position(); ax.set_position([p.x0 + dx, p.y0, p.width, p.height])

def save(fig, name):
    center(fig)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=100)
    plt.close(fig)

R2 = json.load(open(os.path.join(FIG, "v2_results.json")))
NUM = {}

# ---------- Gambar 2: (a) tingkat berbagi, (b) peluang tidak silo, terhadap h ----------
hs = sorted(R2["curve"], key=float); x = np.array([float(h) for h in hs])
cin = np.array([boot([r[0] for r in R2["curve"][h]]) for h in hs])
cout = np.array([boot([r[1] for r in R2["curve"][h]]) for h in hs])
pns = np.array([boot([1.0 if r[1] > 0.5 else 0.0 for r in R2["curve"][h]]) for h in hs])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.05), gridspec_kw={"wspace": 0.42})
for ax in (a1, a2):
    ax.axvspan(0.32, 0.43, color=LIGHT, alpha=0.35, lw=0, zorder=0)   # rentang metaanalisis (lihat keterangan)
    ax.set_xlim(0, 1.5); ax.set_xticks([0, 0.3, 0.6, 1.0, 1.5]); ax.xaxis.set_major_formatter(koma)
    ax.yaxis.set_major_formatter(koma); ax.set_xlabel("In-group bias, h")
for arr, c, ls in ((cin, INK, "-"), (cout, RED, "-")):
    a1.fill_between(x, arr[:, 1], arr[:, 2], color=c, alpha=0.15, lw=0)
    a1.plot(x, arr[:, 0], color=c, ls=ls)
a1.text(1.5, cin[-1, 0] + 0.07, "dalam kelompok", color=INK, ha="right", va="bottom", fontsize=20)
a1.text(0.6, 0.36, "lintas kelompok", color=RED, ha="left", va="bottom", fontsize=20)
a1.set_ylim(0, 1); a1.set_yticks([0, 0.5, 1]); a1.set_ylabel("Tingkat\nberbagi")
a2.fill_between(x, pns[:, 1] * 100, pns[:, 2] * 100, color=RED, alpha=0.15, lw=0)
a2.plot(x, pns[:, 0] * 100, color=RED, marker="o", ms=7)
a2.axhline(50, color=GREY, lw=1.2, ls=(0, (4, 4)), zorder=0)
a2.set_ylim(0, 100); a2.set_yticks([0, 25, 50, 75, 100]); a2.set_ylabel("Tidak silo (%)")
panel(a1, "(a)"); panel(a2, "(b)")
fig.subplots_adjust(left=0.0743, right=0.9743, bottom=0.30 * 3.2 / 3.05, top=1 - 0.18 * 3.2 / 3.05)
save(fig, "g2_silo")
NUM["silo"] = {"out": {h: cout[i].tolist() for i, h in enumerate(hs)}, "in_range": [cin[:, 0].min(), cin[:, 0].max()],
               "nonsilo_pct": {h: (pns[i] * 100).tolist() for i, h in enumerate(hs)}}

# ---------- Gambar 3: instrumen pada bias rendah dan tinggi ----------
ROWS = [("Tanpa kebijakan", "beta|0"), ("Rotasi, r = 0,2", "rot|0.2"), ("Forum, φ = 0,6", "phi|0.6"),
        ("Bonus, β = c", "beta|1.0"), ("Reputasi, ρ = 0,1", "rho|0.1")]
vals = {(l, sc): boot([r[1] for r in R2["levers"][f"{sc}|{k}"]]) for l, k in ROWS for sc in ("rendah", "tinggi")}
fig, ax = plt.subplots(figsize=(W, 3.1))
ys = np.arange(len(ROWS))[::-1]
for y, (lab, _) in zip(ys, ROWS):
    lo, hi = vals[(lab, "rendah")], vals[(lab, "tinggi")]
    ax.errorbar(lo[0], y + 0.14, xerr=[[lo[0] - lo[1]], [lo[2] - lo[0]]], fmt="o", ms=12, mfc="white",
                mec=INK, mew=2.2, ecolor=INK, elinewidth=2, capsize=0)
    ax.errorbar(hi[0], y - 0.14, xerr=[[hi[0] - hi[1]], [hi[2] - hi[0]]], fmt="o", ms=12, mfc=RED,
                mec=RED, ecolor=RED, elinewidth=2, capsize=0)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in ROWS]); ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False); ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.xaxis.set_major_formatter(koma)
ax.set_xlabel("Berbagi lintas kelompok")
ax.plot([], [], "o", ms=11, mfc=RED, mec=RED, label="h = 1,0 (uji tekanan)")
ax.plot([], [], "o", ms=11, mfc="white", mec=INK, mew=2.2, label="h = 0,3 (rentang kalibrasi)")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3, borderaxespad=0.1, columnspacing=1.5)
fig.subplots_adjust(left=0.194, right=0.944, bottom=0.896 / 3.1, top=1 - 0.576 / 3.1)
save(fig, "g3_instrumen")
NUM["instrumen"] = {f"{l}|{s}": v for (l, s), v in vals.items()}

# ---------- Gambar 4: perubahan berbagi akibat rotasi, terhadap h (analisis tambahan) ----------
R4 = json.load(open(os.path.join(FIG, "v34_rotmap.json"))); H = R4["h"]; hx = np.array(H)
def delta(h, r, col):
    a = np.array(R4["cell"][f"{h}|{r}"])[:, col]; b = np.array(R4["cell"][f"{h}|0.0"])[:, col]
    return boot(a - b)
dout = np.array([delta(h, 0.2, 1) for h in H]); din = np.array([delta(h, 0.2, 0) for h in H])
dout1 = np.array([delta(h, 0.1, 1) for h in H])
fig, ax = plt.subplots(figsize=(W, 3.6))
ax.axhline(0, color=GREY, lw=1.2, zorder=0)
for arr, c in ((dout, RED), (din, INK)):
    ax.fill_between(hx, arr[:, 1], arr[:, 2], color=c, alpha=0.15, lw=0)
    ax.plot(hx, arr[:, 0], color=c)
ax.plot(hx, dout1[:, 0], color=RED, lw=1.8, ls=(0, (5, 4)))
ax.text(1.52, dout[-1, 0], "lintas kelompok", color=RED, va="center", fontsize=20)
ax.text(1.52, dout1[-1, 0] + 0.07, "lintas, r = 0,1", color=RED, va="center", fontsize=18)
ax.text(1.52, din[-1, 0], "dalam kelompok", color=INK, va="center", fontsize=20)
ax.set_xlim(0, 1.5); ax.set_xticks([0, 0.3, 0.6, 1.0, 1.5]); ax.xaxis.set_major_formatter(koma)
ax.set_ylim(-0.5, 0.4); ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4]); ax.yaxis.set_major_formatter(koma)
ax.set_xlabel("In-group bias, h"); ax.set_ylabel("Perubahan berbagi")
fig.subplots_adjust(left=0.1087, right=0.8087, bottom=0.245, top=0.97)
save(fig, "g4_rotasi")
edge = 0.0
for hv, lo_ in zip(hx, dout[:, 1]):
    if lo_ > 0: edge = float(hv)
    else: break
NUM["rotasi"] = {"edge_ci_pos_from0": edge, "dout": {str(h): dout[i].tolist() for i, h in enumerate(H)},
                 "din": {str(h): din[i].tolist() for i, h in enumerate(H)}}

# ---------- Gambar 5: lintasan waktu berbagi lintas kelompok ----------
R3 = json.load(open(os.path.join(FIG, "v33_results.json")))
ST = [("rotasi saja", GREY, (0, (1.5, 2.5)), 2.4, "rotasi saja"),
      ("serentak", INK, (0, (6, 4)), 2.4, "serentak"),
      ("bertahap", INK, "-", 3.0, "bertahap"),
      ("bonus dicabut saat rotasi", RED, "-", 3.0, "bonus dicabut")]
fig, ax = plt.subplots(figsize=(W, 3.1))
end = {}
for k, c, ls, lw, lab in ST:
    A = np.array(R3["traj"][k])[:, :, 1]; ts = np.arange(A.shape[1]) * 10; mu = A.mean(0)
    bs = np.array([A[rng.integers(0, len(A), len(A))].mean(0) for _ in range(400)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    ax.fill_between(ts, lo, hi, color=c, alpha=0.15 if lw >= 3 else 0.10, lw=0)
    ax.plot(ts, mu, color=c, ls=ls, lw=lw)
    end[k] = float(A[:, -100:].mean())
ly = {"bertahap": 0.04, "serentak": -0.045, "bonus dicabut saat rotasi": 0.0, "rotasi saja": -0.03}
for k, c, ls, lw, lab in ST:
    ax.text(4560, end[k] + ly[k], lab, color=c, va="center", fontsize=20)
ax.axvline(1500, color=GREY, lw=1.2, ls=(0, (4, 4)), zorder=0)
ax.text(1540, 0.96, "rotasi dimulai", fontsize=18, color=GREY, va="top")
yb, yd = end["bertahap"], end["bonus dicabut saat rotasi"]
ax.plot([4380, 4420, 4420, 4380], [yb, yb, yd, yd], color=INK, lw=1.4)
ax.text(4360, (yb + yd) / 2, "0,50 [0,43; 0,56]", ha="right", va="center", fontsize=19)
ax.set_xlim(0, 4500); ax.set_xticks([0, 1500, 3000, 4500]); ax.set_xticklabels(["0", "1.500", "3.000", "4.500"])
ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.yaxis.set_major_formatter(koma)
ax.set_xlabel("Langkah Monte Carlo (MCS)"); ax.set_ylabel("Berbagi lintas\nkelompok")
fig.subplots_adjust(left=0.1065, right=0.8215, bottom=1 - (1 - 0.265) * 3.3 / 3.1 if False else 0.265 * 3.3 / 3.1, top=1 - 0.03 * 3.3 / 3.1)
save(fig, "g5_lintasan")
NUM["lintasan_akhir"] = end


# ---------- Gambar 1(b): potret jaringan, satu ulangan (h = 1) ----------
import model_v1 as m
SCOL = {0: LIGHT, 1: INK, 2: GREY, 3: RED}      # H, I, O, S
def snapshot(ax, beta, seed=3):
    st, strat, (ip, ix, g), _ = m.run_state(h=1.0, beta=beta, seed=seed)
    r = np.random.default_rng(42); n = len(g)
    cen = {0: (0, 1.0), 1: (-0.95, -0.55), 2: (0.95, -0.55)}
    ang = r.uniform(0, 2 * np.pi, n); rad = 0.5 * np.sqrt(r.uniform(0, 1, n))
    pos = np.array([[cen[g[i]][0] + rad[i] * np.cos(ang[i]), cen[g[i]][1] + rad[i] * np.sin(ang[i])] for i in range(n)])
    for i in range(n):
        for j in ix[ip[i]:ip[i + 1]]:
            if j <= i: continue
            if g[i] == g[j]:
                ax.plot(*pos[[i, j]].T, color="#D6DDE6", lw=0.5, zorder=1)
            else:
                act = (strat[i] & 2) or (strat[j] & 2)
                ax.plot(*pos[[i, j]].T, color=RED if act else "#D6DDE6", lw=1.1 if act else 0.6,
                        ls="-" if act else (0, (2, 2)), zorder=2 if act else 1)
    ax.scatter(pos[:, 0], pos[:, 1], s=40, c=[SCOL[s] for s in strat], edgecolors=INK, linewidths=0.4, zorder=3)
    for k, nm in {0: "kelompok A", 1: "kelompok B", 2: "kelompok C"}.items():
        x0, y0 = cen[k]; ax.text(x0, y0 + (0.62 if k == 0 else -0.66), nm, ha="center", va="center", fontsize=19, color=INK)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.35, 1.75)
    return st
fig, axs = plt.subplots(1, 2, figsize=(W, 3.7))
s0 = snapshot(axs[0], 0.0); s1 = snapshot(axs[1], 1.0)
for ax, title in zip(axs, ("(b) tanpa kebijakan", "(c) bonus β = c")):
    ax.set_title(title, fontsize=21, fontweight="bold", pad=14, loc="center")
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.87, wspace=-0.25)
save(fig, "g1_jaringan")
NUM["jaringan"] = {"tanpa": [float(s0[0]), float(s0[1])], "bonus": [float(s1[0]), float(s1[1])]}

json.dump(NUM, open(os.path.join(OUT, "angka_figur.json"), "w"), indent=1, default=float)
print("silo h0,h1:", np.round(cout[0], 3), np.round(cout[list(x).index(1.0)], 3))
print("non-silo %: h0", np.round(pns[0] * 100, 1), "h0.2", np.round(pns[2] * 100, 1), "h1", np.round(pns[list(x).index(1.0)] * 100, 1))
print("jaringan", NUM["jaringan"]); print("rotasi edge", edge, "| akhir lintasan", {k: round(v, 3) for k, v in end.items()})
for k in ("Bonus, β = c|tinggi", "Forum, φ = 0,6|tinggi", "Rotasi, r = 0,2|tinggi", "Reputasi, ρ = 0,1|tinggi"):
    print(k, np.round(NUM["instrumen"][k], 3))
