"""Gambar urutan kebijakan v3 (paparan insentif dikendalikan)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("figures/v31_results.json")); rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x); b = [rng.choice(x, len(x)).mean() for _ in range(3000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
order = ["rotasi saja", "serentak (paparan 3.000)", "bertahap (paparan 3.000)", "serentak (paparan 4.500)",
         "bertahap (paparan 4.500)", "bertahap, bonus dicabut (paparan 1.500)"]
lab = ["rotasi\nsaja", "serentak\n(3.000)", "bertahap\n(3.000)", "serentak\n(4.500)", "bertahap\n(4.500)", "bonus\ndicabut\n(1.500)"]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
for ax, col, ttl in ((axs[0], 1, "(a) Berbagi lintas kelompok"), (axs[1], 0, "(b) Berbagi dalam kelompok")):
    for i, (n, c) in enumerate((("sedang", "#F58518"), ("tinggi", "#B23A48"))):
        st = np.array([ci([r[col] for r in R["seq"][f"{n}|{k}"]]) for k in order]); xs = np.arange(len(order))
        ax.bar(xs + (i - 0.5) * 0.38, st[:, 0], 0.38, yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], capsize=2, color=c, label=f"bias {n}")
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(lab, fontsize=7.5); ax.set_ylim(0, 1); ax.set_title(ttl, loc="left", fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: ("%g" % round(v, 6)).replace(".", ",")))
axs[0].set_ylabel("Tingkat berbagi"); axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/v3_f7_seq.png", dpi=200)
