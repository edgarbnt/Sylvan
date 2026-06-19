import re, statistics
pat = re.compile(r'Episode (\d+) \| Step (\d+) \| Energy: ([\-0-9.]+).*food_d: ([\-0-9.]+)')
EAT_R = 1.0

def analyze(fc):
    f = f"/tmp/forage_c{fc}.log"
    try:
        lines = open(f).readlines()
    except FileNotFoundError:
        print(f"FOOD_COUNT={fc}: MISSING"); return
    eps = {}
    for ln in lines:
        m = pat.search(ln)
        if not m: continue
        ep=int(m.group(1)); step=int(m.group(2)); E=float(m.group(3)); fd=float(m.group(4))
        eps.setdefault(ep, []).append((step, E, fd))
    survivals, meals_all, min_fd_per_approach = [], [], []
    stall_count = closed_count = 0
    for ep in sorted(eps):
        rows = eps[ep]
        survivals.append(max(r[0] for r in rows))
        # meals = energy jumps > 5
        meals = 0
        prevE = rows[0][1]
        for _,E,_fd in rows[1:]:
            if E > prevE + 5: meals += 1
            prevE = E
        meals_all.append(meals)
        # approach analysis: track local minima of food_d (closest approaches)
        fds = [r[2] for r in rows]
        for i in range(1, len(fds)-1):
            if fds[i] <= fds[i-1] and fds[i] < fds[i+1]:  # local min
                min_fd_per_approach.append(fds[i])
                if fds[i] < EAT_R: closed_count += 1
                elif fds[i] < EAT_R + 1.0: stall_count += 1  # got close (1.0-2.0) but didn't close
        # also the global min
    med_surv = statistics.median(survivals)
    mean_meals = statistics.mean(meals_all)
    starved0 = sum(1 for m in meals_all if m == 0)
    closest = sorted(min_fd_per_approach)[:5] if min_fd_per_approach else []
    print(f"FC={fc:>2} | n={len(survivals)} | surv med={med_surv:>5.0f} | repas moy={mean_meals:.2f} | "
          f"0-repas={starved0}/{len(survivals)} | approches: closées(<{EAT_R})={closed_count} stallées(1-2m)={stall_count}")
    if closest:
        print(f"        5 meilleures approches (min food_d): {[round(x,2) for x in closest]}")

print("=== H1 (saturation): le foraging EMPIRE-t-il quand FOOD_COUNT monte ? ===")
print("=== H2 (terminale): bcp d'approches 'stallées 1-2m' vs 'closées <1m' ? ===\n")
for fc in (1, 3, 6, 12):
    analyze(fc)
