"""Angka tambahan di poster yang tidak tertulis di makalah, dihitung dari hasil simulasi yang sama.

1. Kepekaan ambang silo: porsi ulangan tidak silo (berbagi lintas kelompok > ambang) pada ambang 0,4; 0,5; 0,6.
2. Penurunan berbagi lintas kelompok dari tingkat sebelum rotasi (rata-rata MCS 1.400-1.500) ke akhir
   lintasan (100 titik terakhir) pada kondisi "bonus dicabut saat rotasi", dengan SK bootstrap 95%.
3. SK bootstrap 95% untuk porsi tidak silo.

Pemakaian (dari folder poster/):  python analisis_tambahan.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
rng = np.random.default_rng(20261120)


def boot_ci(x, B=4000):
    x = np.asarray(x, float)
    s = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


curve = json.load(open(os.path.join(FIG, "v2_results.json")))["curve"]
print("1) Porsi tidak silo (%) menurut ambang; 64 ulangan per h")
for thr in (0.4, 0.5, 0.6):
    row = []
    for h in ("0.0", "0.3", "0.4", "1.0"):
        m, lo, hi = boot_ci([1.0 if r[1] > thr else 0.0 for r in curve[h]])
        row.append(f"h={h}: {100 * m:.0f} [{100 * lo:.0f}; {100 * hi:.0f}]")
    print(f"   ambang {thr}: " + ", ".join(row))

traj = json.load(open(os.path.join(FIG, "v33_results.json")))["traj"]
A = np.array(traj["bonus dicabut saat rotasi"])[:, :, 1]
ts = np.arange(A.shape[1]) * 10
pre = A[:, (ts >= 1400) & (ts < 1500)].mean(1)
end = A[:, -100:].mean(1)
m, lo, hi = boot_ci(pre - end)
print(f"2) Bonus dicabut: sebelum rotasi {pre.mean():.3f}, akhir {end.mean():.3f}, "
      f"penurunan {m:.2f} [{lo:.2f}; {hi:.2f}] (128 ulangan)")
