"""
Replikasi baseline: Prisoner's Dilemma lemah pada kisi persegi dengan aturan imitasi Fermi
(Szabo & Toke 1998; ulasan Szabo & Fath 2007).

Payoff: R=1 (C-C), T=b (D-C), S=P=0. Tetangga von Neumann (4), batas periodik.
Pembaruan asinkron: pilih x acak, pilih tetangga y acak, x meniru y dengan peluang
1 / (1 + exp((P_x - P_y)/K)), K = 0.1.

Hasil yang dicek: kooperator bertahan pada kisi untuk b sedikit di atas 1 dan punah di atas
ambang b_c. Literatur melaporkan b_c sekitar 1.04 untuk K = 0.1 (cek angka pastinya di sumber).
Pada populasi tercampur sempurna (well-mixed), kooperator punah untuk semua b > 1.
"""
import numpy as np, numba, time, json, sys

@numba.njit(cache=True)
def payoff(S, L, x, y, b):
    s = S[x, y]; p = 0.0
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        t = S[(x+dx) % L, (y+dy) % L]
        if s == 1 and t == 1: p += 1.0
        elif s == 0 and t == 1: p += b
    return p

@numba.njit(cache=True)
def run(L, b, K, mcs, avg_last, seed):
    np.random.seed(seed)
    S = (np.random.random((L, L)) < 0.5).astype(np.int8)
    N = L * L; acc = 0.0; cnt = 0
    for step in range(mcs):
        for _ in range(N):
            x = np.random.randint(L); y = np.random.randint(L)
            r = np.random.randint(4)
            if r == 0: x2, y2 = (x+1) % L, y
            elif r == 1: x2, y2 = (x-1) % L, y
            elif r == 2: x2, y2 = x, (y+1) % L
            else: x2, y2 = x, (y-1) % L
            if S[x, y] == S[x2, y2]: continue
            px = payoff(S, L, x, y, b); py = payoff(S, L, x2, y2, b)
            if np.random.random() < 1.0 / (1.0 + np.exp((px - py) / K)):
                S[x, y] = S[x2, y2]
        rho = S.sum() / N
        if rho == 0.0 or rho == 1.0:
            return rho
        if step >= mcs - avg_last:
            acc += rho; cnt += 1
    return acc / cnt

@numba.njit(cache=True)
def run_wellmixed(N, b, K, mcs, seed):
    # Populasi tercampur: payoff = rata-rata terhadap seluruh populasi
    np.random.seed(seed)
    S = (np.random.random(N) < 0.5).astype(np.int8)
    for step in range(mcs):
        nc = S.sum()
        for _ in range(N):
            x = np.random.randint(N); y = np.random.randint(N)
            if S[x] == S[y]: continue
            fc = nc / N
            px = fc * (1.0 if S[x] == 1 else b)
            py = fc * (1.0 if S[y] == 1 else b)
            if np.random.random() < 1.0 / (1.0 + np.exp((px - py) / K)):
                nc += 1 if S[y] == 1 else -1
                S[x] = S[y]
        if nc == 0 or nc == N: return nc / N
    return S.sum() / N

if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mcs = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    bs = np.round(np.arange(1.00, 1.065, 0.005), 3)
    res = {"L": L, "mcs": mcs, "K": 0.1, "lattice": {}, "wellmixed": {}}
    t0 = time.time()
    for b in bs:
        vals = [run(L, b, 0.1, mcs, 1000, s) for s in range(3)]
        res["lattice"][str(b)] = float(np.mean(vals))
        res["wellmixed"][str(b)] = float(run_wellmixed(L*L, b, 0.1, 2000, 0))
        print(b, res["lattice"][str(b)], res["wellmixed"][str(b)], round(time.time()-t0), flush=True)
    json.dump(res, open("figures/baseline_results.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bs, [res["lattice"][str(b)] for b in bs], "o-", label="Kisi persegi (L=%d)" % L)
    ax.plot(bs, [res["wellmixed"][str(b)] for b in bs], "s--", label="Tercampur sempurna")
    ax.set_xlabel("Godaan untuk membelot, b"); ax.set_ylabel("Fraksi kooperator")
    ax.set_title("Replikasi baseline: PD lemah, aturan Fermi K = 0,1")
    ax.legend(); fig.tight_layout(); fig.savefig("figures/baseline_lattice.png", dpi=150)
