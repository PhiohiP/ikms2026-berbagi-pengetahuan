import json, numpy as np
R=json.load(open("figures/v31_results.json")); rng=np.random.default_rng(7); S=R["seq"]
def ci(x):
    x=np.asarray(x); b=[rng.choice(x,len(x)).mean() for _ in range(3000)]; return round(x.mean(),3), np.round(np.percentile(b,[2.5,97.5]),3).tolist()
def diff(a,b,B=5000):
    a=np.array(a);b=np.array(b);d=a.mean()-b.mean()
    bs=[rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean() for _ in range(B)]
    pool=np.concatenate([a,b]);n=len(a);c=0
    for _ in range(B):
        rng.shuffle(pool); c+=abs(pool[:n].mean()-pool[n:].mean())>=abs(d)
    return round(d,3), np.round(np.percentile(bs,[2.5,97.5]),3).tolist(), round((c+1)/(B+1),4)
for k,v in S.items(): print(k,"out",ci([r[1] for r in v]),"in",ci([r[0] for r in v]))
for n in ("sedang","tinggi"):
    for col,nm in ((1,"out"),(0,"in")):
        g=lambda k:[r[col] for r in S[f"{n}|{k}"]]
        print(n,nm,"B4500-S4500",diff(g("bertahap (paparan 4.500)"),g("serentak (paparan 4.500)")))
        print(n,nm,"B3000-S3000",diff(g("bertahap (paparan 3.000)"),g("serentak (paparan 3.000)")))
        print(n,nm,"S4500-S3000",diff(g("serentak (paparan 4.500)"),g("serentak (paparan 3.000)")))
        print(n,nm,"B4500-dicabut",diff(g("bertahap (paparan 4.500)"),g("bertahap, bonus dicabut (paparan 1.500)")))
        print(n,nm,"rotasi saja vs serentak3000",diff(g("serentak (paparan 3.000)"),g("rotasi saja")))
