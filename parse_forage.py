import re, sys, statistics
pat = re.compile(r'Episode (\d+) \| Step (\d+) \| Energy: ([\-0-9.]+)')

def analyze(path, label):
    eps = {}
    try:
        lines = open(path).readlines()
    except FileNotFoundError:
        print(f"{label}: MISSING {path}"); return
    for line in lines:
        m = pat.search(line)
        if not m:
            continue
        ep = int(m.group(1)); step = int(m.group(2)); en = float(m.group(3))
        eps.setdefault(ep, []).append((step, en))
    survivals = []
    never_ate = 0
    for ep in sorted(eps):
        rows = eps[ep]
        last_step = max(r[0] for r in rows)
        survivals.append(last_step)
        # "ate" heuristic: energy ever rose above its starting value (a pellet restored energy)
        e0 = rows[0][1]
        peak = max(r[1] for r in rows)
        if peak <= e0 + 1.0:
            never_ate += 1
    survivals.sort()
    n = len(survivals)
    med = statistics.median(survivals) if survivals else 0
    mean = statistics.mean(survivals) if survivals else 0
    print(f"{label}: n={n} | median={med:.0f} | mean={mean:.0f} | min={min(survivals) if survivals else 0} "
          f"max={max(survivals) if survivals else 0} | never_ate={never_ate}/{n}")
    print(f"   survivals sorted: {survivals}")

analyze("/tmp/forage_ab_0.0.log", "heading_w=0 (control)")
analyze("/tmp/forage_ab_2.0.log", "heading_w=2 (on)    ")
