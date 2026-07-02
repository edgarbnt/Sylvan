import re
pat = re.compile(r'Episode (\d+) \| Step (\d+).*Energy: ([\-0-9.]+) \| Thirst: ([\-0-9.]+).*food_d: ([\-0-9.]+) \| water_d: ([\-0-9.]+)')
SCEN = [  # name, expected target
    ("hungry_FR", "FOOD"), ("hungry_FL", "FOOD"),
    ("thirsty_FR", "WATER"), ("thirsty_FL", "WATER"),
    ("thirst_vs_prox", "WATER"), ("hunger_vs_prox", "FOOD"),
]
print(f"{'scenario':16s} {'ep':>2} {'E':>4} {'T':>4} {'minFood':>7} {'minWat':>7} {'reached':>8} {'expect':>6} {'OK':>3}")
print("-"*72)
ok_n = tot = 0
for name, exp in SCEN:
    try:
        lines = open(f"/tmp/arb_{name}.log").readlines()
    except FileNotFoundError:
        print(f"{name:16s} MISSING"); continue
    eps = {}
    for ln in lines:
        m = pat.search(ln)
        if not m: continue
        ep=int(m.group(1)); fd=float(m.group(5)); wd=float(m.group(6)); E=float(m.group(3)); T=float(m.group(4))
        eps.setdefault(ep, {"fd":[], "wd":[], "E":E, "T":T})
        eps[ep]["fd"].append(fd); eps[ep]["wd"].append(wd)
    for ep in sorted(eps):
        d=eps[ep]; mf=min(d["fd"]); mw=min(d["wd"])
        # which did it actually reach/approach: reached if <1.2, else whichever min is lower
        reached = "FOOD" if (mf<1.2 and mf<=mw) else ("WATER" if (mw<1.2 and mw<mf) else ("FOOD" if mf<mw else "WATER"))
        ok = reached==exp; ok_n+=ok; tot+=1
        print(f"{name:16s} {ep:>2} {d['E']:>4.0f} {d['T']:>4.0f} {mf:>7.2f} {mw:>7.2f} {reached:>8} {exp:>6} {'Y' if ok else '.':>3}")
print("-"*72)
print(f"ARBITRAGE CORRECT: {ok_n}/{tot}")
