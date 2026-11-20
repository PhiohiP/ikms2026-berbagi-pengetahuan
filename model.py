"""
Model v0: berbagi pengetahuan lintas entitas asal pascapenggabungan.

Struktur
- Jaringan stochastic block model (SBM) dengan G komunitas (default 3: eks-KPI, eks-PIS, Patra Niaga).
  Rata-rata derajat dalam komunitas k_in dan lintas komunitas k_out.
- Strategi agen (2 bit): bit0 = berbagi ke rekan satu entitas, bit1 = berbagi ke rekan entitas lain.
  0 = H (menahan), 1 = I (hanya ke kelompok sendiri), 2 = O (hanya ke kelompok lain), 3 = S (ke semua).
- Setiap tautan adalah permainan donasi: pemberi menanggung biaya, penerima mendapat manfaat b.

Utilitas yang dirasakan agen i (dasar imitasi):
  U_i = sum manfaat b yang diterima dari tetangga yang berbagi ke i
        - sum biaya berbagi ke tetangga satu entitas:  c_in  = c * (1 - phi)
        - sum biaya berbagi ke tetangga entitas lain:  c_out = c * (1 + h) * (1 - phi) - beta
        + rho * b untuk setiap pembagian lintas entitas (reputasi, versi disederhanakan)
  h    : bias in-group (biaya psikologis tambahan saat berbagi ke kelompok lain)
  beta : bonus per pembagian lintas entitas (analog penghargaan inovasi APQ)
  phi  : penurunan biaya berbagi lewat forum (analog KOMET/JITU), 0..1
  rho  : imbalan reputasi per pembagian lintas entitas (stub indirect reciprocity, v1 akan diganti)
  rot  : rotasi, porsi tautan dalam-komunitas yang dipindah menjadi tautan lintas komunitas

Dinamika: pembaruan asinkron, i meniru tetangga acak j dengan peluang Fermi
  1 / (1 + exp((U_i - U_j) / K)), ditambah mutasi strategi acak dengan peluang mu.

Keluaran: tingkat berbagi dalam dan lintas entitas (fraksi tautan berarah yang benar-benar dipakai berbagi).
Catatan: nilai parameter default adalah placeholder v0, belum dikalibrasi ke literatur.
"""
import numpy as np, numba

def make_sbm(sizes, k_in, k_out, rot=0.0, seed=0):
    rng = np.random.default_rng(seed)
    sizes = np.asarray(sizes); N = sizes.sum(); G = len(sizes)
    group = np.repeat(np.arange(G), sizes)
    k_in_eff = k_in * (1 - rot); k_out_eff = k_out + k_in * rot
    edges = []
    for a in range(G):
        ia = np.where(group == a)[0]
        p = k_in_eff / (len(ia) - 1)
        for x in range(len(ia)):
            mask = rng.random(len(ia) - x - 1) < p
            for y in np.nonzero(mask)[0]:
                edges.append((ia[x], ia[x + 1 + y]))
    others = N - sizes  # node count outside each group
    for a in range(G):
        for bgrp in range(a + 1, G):
            ia = np.where(group == a)[0]; ib = np.where(group == bgrp)[0]
            # p chosen so that expected cross-degree of a node equals k_out_eff
            p = k_out_eff / others[a]
            m = rng.random((len(ia), len(ib))) < p
            for x, y in zip(*np.nonzero(m)):
                edges.append((ia[x], ib[y]))
    edges = np.array(edges, dtype=np.int64)
    deg = np.bincount(edges.ravel(), minlength=N)
    indptr = np.zeros(N + 1, dtype=np.int64); indptr[1:] = np.cumsum(deg)
    indices = np.empty(indptr[-1], dtype=np.int64); fill = indptr[:-1].copy()
    for u, v in edges:
        indices[fill[u]] = v; fill[u] += 1
        indices[fill[v]] = u; fill[v] += 1
    return indptr, indices, group.astype(np.int64)

@numba.njit(cache=True)
def utility(i, strat, indptr, indices, group, b, c_in, c_out, rho):
    u = 0.0; si = strat[i]
    for e in range(indptr[i], indptr[i + 1]):
        j = indices[e]; sj = strat[j]
        if group[j] == group[i]:
            if sj & 1: u += b
            if si & 1: u -= c_in
        else:
            if sj & 2: u += b
            if si & 2: u += rho * b - c_out
    return u

@numba.njit(cache=True)
def simulate(indptr, indices, group, b, c, h, beta, phi, rho, K, mu, mcs, avg_last, seed):
    np.random.seed(seed)
    N = len(group)
    c_in = c * (1.0 - phi)
    c_out = c * (1.0 + h) * (1.0 - phi) - beta
    strat = np.random.randint(0, 4, N).astype(np.int64)
    acc_in = 0.0; acc_out = 0.0; cnt = 0
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
            acc_in += s_in / max(n_in, 1); acc_out += s_out / max(n_out, 1); cnt += 1
    return acc_in / cnt, acc_out / cnt

DEFAULT = dict(sizes=(100, 100, 100), k_in=6.0, k_out=2.0, b=6.0, c=1.0, K=0.5, mu=0.01,
               mcs=1500, avg_last=500)

def run(h=0.0, beta=0.0, phi=0.0, rho=0.0, rot=0.0, seed=0, **kw):
    p = {**DEFAULT, **kw}
    ip, ix, g = make_sbm(p["sizes"], p["k_in"], p["k_out"], rot=rot, seed=seed)
    return simulate(ip, ix, g, p["b"], p["c"], h, beta, phi, rho, p["K"], p["mu"],
                    p["mcs"], p["avg_last"], seed + 1000)
