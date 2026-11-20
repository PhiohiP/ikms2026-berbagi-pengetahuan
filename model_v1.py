"""
Model v1: perluasan model v0.

Tambahan dibanding v0
1. simulate_state(): mengembalikan strategi akhir dan statistik berbagi, untuk dipakai lapisan NK.
2. Ukuran komunitas boleh tidak sama (uji sensitivitas).
3. welfare(): surplus organisasi dan belanja program per periode, untuk perbandingan efektivitas biaya.
4. Lapisan collective intelligence: pencarian solusi pada NK landscape (Kauffman dan Levin, 1987)
   dengan aturan ala Lazer dan Friedman (2007). Agen hanya bisa menyalin solusi dari tetangga
   yang berbagi kepadanya, sesuai strategi akhir permainan.

Catatan: reputasi (rho) tetap bentuk tereduksi indirect reciprocity: imbalan rho*b per pembagian
lintas entitas yang diakui publik. Bentuk ini diakui sebagai penyederhanaan di bagian keterbatasan.
"""
import numpy as np, numba
from model import make_sbm, utility

@numba.njit(cache=True)
def simulate_state(indptr, indices, group, b, c, h, beta, phi, rho, K, mu, mcs, avg_last, seed, init):
    np.random.seed(seed)
    N = len(group)
    c_in = c * (1.0 - phi)
    c_out = c * (1.0 + h) * (1.0 - phi) - beta
    if len(init) == N:
        strat = init.copy()
    else:
        strat = np.random.randint(0, 4, N).astype(np.int64)
    acc = np.zeros(4)  # in-rate, out-rate, n_in_shares, n_out_shares (rata-rata jendela akhir)
    cnt = 0
    for step in range(mcs):
        for _ in range(N):
            i = np.random.randint(N)
            if np.random.random() < mu:
                strat[i] = np.random.randint(4); continue
            d = indptr[i + 1] - indptr[i]
            if d == 0: continue
            j = indices[indptr[i] + np.random.randint(d)]
            if strat[i] == strat[j]: continue
            ui = utility(i, strat, indptr, indices, group, b, c_in, c_out, rho)
            uj = utility(j, strat, indptr, indices, group, b, c_in, c_out, rho)
            if np.random.random() < 1.0 / (1.0 + np.exp((ui - uj) / K)):
                strat[i] = strat[j]
        if step >= mcs - avg_last:
            n_in = 0; s_in = 0; n_out = 0; s_out = 0
            for x in range(N):
                for e in range(indptr[x], indptr[x + 1]):
                    y = indices[e]
                    if group[y] == group[x]:
                        n_in += 1
                        if strat[x] & 1: s_in += 1
                    else:
                        n_out += 1
                        if strat[x] & 2: s_out += 1
            acc[0] += s_in / max(n_in, 1); acc[1] += s_out / max(n_out, 1)
            acc[2] += s_in; acc[3] += s_out; cnt += 1
    return acc / cnt, strat

def welfare(stats, b, c, beta, phi):
    """Surplus organisasi per periode (satuan c) dan belanja program.
    Setiap pembagian bernilai b bagi penerima dan memakan waktu c dari pemberi (biaya riil, tanpa bias).
    Belanja program: bonus beta per pembagian lintas entitas + subsidi forum phi*c per pembagian."""
    s_in, s_out = stats[2], stats[3]
    surplus = (s_in + s_out) * (b - c)
    spend = beta * s_out + phi * c * (s_in + s_out)
    return surplus, spend

# ---------------- NK landscape ----------------
def make_nk(Nbits, Kdep, rng):
    deps = np.zeros((Nbits, Kdep + 1), dtype=np.int64)
    for i in range(Nbits):
        others = rng.choice([x for x in range(Nbits) if x != i], Kdep, replace=False)
        deps[i] = np.concatenate(([i], others))
    table = rng.random((Nbits, 2 ** (Kdep + 1)))
    return deps, table

@numba.njit(cache=True)
def nk_fit(sol, deps, table):
    Nbits = deps.shape[0]; f = 0.0
    for i in range(Nbits):
        idx = 0
        for k in range(deps.shape[1]):
            idx = (idx << 1) | ((sol >> deps[i, k]) & 1)
        f += table[i, idx]
    return f / Nbits

@numba.njit(cache=True)
def nk_max(deps, table):
    Nbits = deps.shape[0]; best = 0.0
    for s in range(1 << Nbits):
        v = nk_fit(s, deps, table)
        if v > best: best = v
    return best

@numba.njit(cache=True)
def nk_search(indptr, indices, group, strat, deps, table, rounds, seed):
    """Setiap putaran (sinkron): agen melihat tetangga yang BERBAGI kepadanya
    (bit strategi tetangga sesuai relasi), menyalin solusi terbaik jika lebih baik;
    jika tidak ada yang lebih baik, agen bereksplorasi dengan membalik satu bit acak
    dan menyimpannya bila lebih baik. Mengembalikan rata-rata kebugaran per putaran."""
    np.random.seed(seed)
    N = len(group); Nbits = deps.shape[0]
    sol = np.random.randint(0, 1 << Nbits, N).astype(np.int64)
    fit = np.empty(N)
    for i in range(N): fit[i] = nk_fit(sol[i], deps, table)
    traj = np.empty(rounds + 1); traj[0] = fit.mean()
    for t in range(rounds):
        new_sol = sol.copy(); new_fit = fit.copy()
        for i in range(N):
            best_j = -1; best_f = fit[i]
            for e in range(indptr[i], indptr[i + 1]):
                j = indices[e]
                same = group[j] == group[i]
                shares = (strat[j] & 1) if same else (strat[j] & 2)
                if shares and fit[j] > best_f:
                    best_f = fit[j]; best_j = j
            if best_j >= 0:
                new_sol[i] = sol[best_j]; new_fit[i] = best_f
            else:
                cand = sol[i] ^ (1 << np.random.randint(Nbits))
                cf = nk_fit(cand, deps, table)
                if cf > fit[i]:
                    new_sol[i] = cand; new_fit[i] = cf
        sol = new_sol; fit = new_fit
        traj[t + 1] = fit.mean()
    return traj

DEFAULT = dict(sizes=(100, 100, 100), k_in=4.0, k_out=1.0, b=12.0, c=1.0, K=0.5, mu=0.001,
               mcs=3000, avg_last=1000)

NOINIT = np.zeros(0, dtype=np.int64)

def run_state(h=0.0, beta=0.0, phi=0.0, rho=0.0, rot=0.0, seed=0, init=None, net_seed=None, **kw):
    p = {**DEFAULT, **kw}
    ip, ix, g = make_sbm(p["sizes"], p["k_in"], p["k_out"], rot=rot, seed=seed if net_seed is None else net_seed)
    stats, strat = simulate_state(ip, ix, g, p["b"], p["c"], h, beta, phi, rho, p["K"], p["mu"],
                                  p["mcs"], p["avg_last"], seed + 1000, NOINIT if init is None else init)
    return stats, strat, (ip, ix, g), p

def run_sequence(h, phase1, phase2, seed=0, mcs1=1500, **kw):
    """Kebijakan bertahap: fase 1 di jaringan awal, lalu fase 2 (misalnya rotasi) dengan strategi terbawa.
    phase1/phase2: dict berisi beta, phi, rho, rot."""
    p1 = {"beta": 0.0, "phi": 0.0, "rho": 0.0, "rot": 0.0, **phase1}
    p2 = {"beta": 0.0, "phi": 0.0, "rho": 0.0, "rot": 0.0, **phase2}
    _, strat, _, _ = run_state(h=h, seed=seed, mcs=mcs1, avg_last=1, **p1, **kw)
    return run_state(h=h, seed=seed + 7, init=strat, net_seed=seed + 50_000, **p2, **kw)

# ---------------- v3: kebijakan bertahap dengan fase eksplisit ----------------
def run_phases(h, phases, seed=0, **kw):
    """phases: daftar dict {mcs, beta, rotated}. Jaringan awal (rot = 0) dipakai sampai fase pertama
    yang rotated=True; sejak itu dipakai jaringan hasil rotasi r = 0,2 yang sama untuk semua kondisi.
    Strategi terbawa antarfase. Rata-rata diambil dari 1.000 MCS terakhir fase terakhir."""
    strat = None
    for k, ph in enumerate(phases):
        last = k == len(phases) - 1
        net_seed, rot = (seed + 50_000, 0.2) if ph.get("rotated") else (seed, 0.0)
        stats, strat, net, p = run_state(h=h, beta=ph.get("beta", 0.0), rot=rot, seed=seed + 13 * k,
                                         init=strat, net_seed=net_seed, mcs=ph["mcs"],
                                         avg_last=min(1000, ph["mcs"]) if last else 1, **kw)
    return stats, strat, net, p

# ---------------- v3: NK dengan konteks berbeda per entitas dan daya serap ----------------
@numba.njit(cache=True)
def nk_search_het(indptr, indices, group, strat, deps, tables, maxes, lam, alpha, rounds, seed):
    """tables[0] = lanskap bersama, tables[1..G] = lanskap khas entitas.
    Kebugaran agen kelompok g: (1 - lam) * F0 + lam * F_{g+1}, dinormalisasi oleh optimum kelompoknya.
    Agen menilai solusi tetangga dengan lanskapnya sendiri. Penyalinan berhasil dengan peluang alpha
    (daya serap); bila gagal, agen bereksplorasi satu bit."""
    np.random.seed(seed)
    N = len(group); Nbits = deps.shape[1]
    sol = np.random.randint(0, 1 << Nbits, N).astype(np.int64)
    def_fit = np.empty(N)
    for i in range(N):
        g = group[i]
        def_fit[i] = ((1 - lam) * nk_fit(sol[i], deps[0], tables[0]) + lam * nk_fit(sol[i], deps[g + 1], tables[g + 1])) / maxes[g]
    fit = def_fit
    traj = np.empty(rounds + 1); traj[0] = fit.mean()
    for t in range(rounds):
        new_sol = sol.copy(); new_fit = fit.copy()
        for i in range(N):
            g = group[i]; best_j = -1; best_f = fit[i]
            for e in range(indptr[i], indptr[i + 1]):
                j = indices[e]
                same = group[j] == g
                shares = (strat[j] & 1) if same else (strat[j] & 2)
                if shares:
                    fj = ((1 - lam) * nk_fit(sol[j], deps[0], tables[0]) + lam * nk_fit(sol[j], deps[g + 1], tables[g + 1])) / maxes[g]
                    if fj > best_f:
                        best_f = fj; best_j = j
            if best_j >= 0 and np.random.random() < alpha:
                new_sol[i] = sol[best_j]; new_fit[i] = best_f
            else:
                cand = sol[i] ^ (1 << np.random.randint(Nbits))
                cf = ((1 - lam) * nk_fit(cand, deps[0], tables[0]) + lam * nk_fit(cand, deps[g + 1], tables[g + 1])) / maxes[g]
                if cf > fit[i]:
                    new_sol[i] = cand; new_fit[i] = cf
        sol = new_sol; fit = new_fit
        traj[t + 1] = fit.mean()
    return traj

@numba.njit(cache=True)
def nk_group_max(deps, tables, lam, G):
    Nbits = deps.shape[1]; out = np.zeros(G)
    for s in range(1 << Nbits):
        f0 = nk_fit(s, deps[0], tables[0])
        for g in range(G):
            v = (1 - lam) * f0 + lam * nk_fit(s, deps[g + 1], tables[g + 1])
            if v > out[g]: out[g] = v
    return out

def make_nk_set(Nbits, Kdep, G, rng):
    D = []; T = []
    for _ in range(G + 1):
        d, tb = make_nk(Nbits, Kdep, rng); D.append(d); T.append(tb)
    return np.array(D), np.array(T)

# ---------------- v3.1: rotasi sebagai pemindahan tautan pada jaringan yang sama ----------------
def rewire_rotation(indptr, indices, group, r, seed):
    """Rotasi sesungguhnya: setiap tautan dalam kelompok dipindah dengan peluang r. Satu ujungnya
    (dipilih acak) dihubungkan ke agen acak di kelompok lain; tautan lain tidak berubah."""
    rng = np.random.default_rng(seed); N = len(group)
    E = set()
    for u in range(N):
        for e in range(indptr[u], indptr[u + 1]):
            v = int(indices[e])
            if u < v: E.add((u, v))
    out = set(); moved = []
    for (u, v) in E:
        if group[u] == group[v] and rng.random() < r: moved.append((u, v))
        else: out.add((u, v))
    for (u, v) in moved:
        a = u if rng.random() < 0.5 else v
        for _ in range(100):
            w = int(rng.integers(N))
            if group[w] != group[a] and (min(a, w), max(a, w)) not in out:
                out.add((min(a, w), max(a, w))); break
    edges = np.array(sorted(out), dtype=np.int64)
    deg = np.bincount(edges.ravel(), minlength=N)
    ip = np.zeros(N + 1, dtype=np.int64); ip[1:] = np.cumsum(deg)
    ix = np.empty(ip[-1], dtype=np.int64); fill = ip[:-1].copy()
    for u, v in edges:
        ix[fill[u]] = v; fill[u] += 1; ix[fill[v]] = u; fill[v] += 1
    return ip, ix

def run_phases_true(h, phases, seed=0, r=0.2, **kw):
    """Seperti run_phases, tetapi rotasi memindah porsi r tautan dalam kelompok pada jaringan awal yang sama."""
    p = {**DEFAULT, **kw}
    ip0, ix0, g = make_sbm(p["sizes"], p["k_in"], p["k_out"], rot=0.0, seed=seed)
    ipr, ixr = rewire_rotation(ip0, ix0, g, r, seed + 77)
    strat = NOINIT
    for k, ph in enumerate(phases):
        last = k == len(phases) - 1
        ip, ix = (ipr, ixr) if ph.get("rotated") else (ip0, ix0)
        stats, strat = simulate_state(ip, ix, g, p["b"], p["c"], h, ph.get("beta", 0.0), 0.0, 0.0, p["K"], p["mu"],
                                      ph["mcs"], min(1000, ph["mcs"]) if last else 1, seed + 1000 + 13 * k, strat)
    return stats, strat

# ---------------- v3.2: lintasan waktu untuk gambar ----------------
@numba.njit(cache=True)
def simulate_traj(indptr, indices, group, b, c, h, beta, K, mu, mcs, seed, init):
    """Seperti simulate_state (phi = rho = 0), tetapi mencatat tingkat berbagi dalam dan lintas kelompok setiap MCS."""
    np.random.seed(seed)
    N = len(group)
    c_in = c; c_out = c * (1.0 + h) - beta
    if len(init) == N: strat = init.copy()
    else: strat = np.random.randint(0, 4, N).astype(np.int64)
    tr = np.zeros((mcs, 2))
    for step in range(mcs):
        for _ in range(N):
            i = np.random.randint(N)
            if np.random.random() < mu:
                strat[i] = np.random.randint(4); continue
            d = indptr[i + 1] - indptr[i]
            if d == 0: continue
            j = indices[indptr[i] + np.random.randint(d)]
            if strat[i] == strat[j]: continue
            ui = utility(i, strat, indptr, indices, group, b, c_in, c_out, 0.0)
            uj = utility(j, strat, indptr, indices, group, b, c_in, c_out, 0.0)
            if np.random.random() < 1.0 / (1.0 + np.exp((ui - uj) / K)):
                strat[i] = strat[j]
        n_in = 0; s_in = 0; n_out = 0; s_out = 0
        for x in range(N):
            for e in range(indptr[x], indptr[x + 1]):
                y = indices[e]
                if group[y] == group[x]:
                    n_in += 1
                    if strat[x] & 1: s_in += 1
                else:
                    n_out += 1
                    if strat[x] & 2: s_out += 1
        tr[step, 0] = s_in / max(n_in, 1); tr[step, 1] = s_out / max(n_out, 1)
    return tr, strat

def traj_phases(h, phases, seed=0, r=0.2, **kw):
    p = {**DEFAULT, **kw}
    ip0, ix0, g = make_sbm(p["sizes"], p["k_in"], p["k_out"], rot=0.0, seed=seed)
    ipr, ixr = rewire_rotation(ip0, ix0, g, r, seed + 77)
    strat = NOINIT; out = []
    for k, ph in enumerate(phases):
        ip, ix = (ipr, ixr) if ph.get("rotated") else (ip0, ix0)
        tr, strat = simulate_traj(ip, ix, g, p["b"], p["c"], h, ph.get("beta", 0.0), p["K"], p["mu"], ph["mcs"], seed + 1000 + 13 * k, strat)
        out.append(tr)
    return np.concatenate(out)
