import re
pat = re.compile(r'Episode (\d+) \| Step (\d+).*food_d: ([\-0-9.]+).*brg: ([\-0-9.]+)')
print(f"{'azim':>5} {'ep':>3} {'brg0':>6} {'d0':>5} {'dmin':>5} {'reach<0.8':>9}")
print("-"*40)
summary = {}
for A in [0, 45, 90, 135, 180, 225, 270, 315]:
    f = f"/tmp/nav_ab_{A}.log"
    eps = {}
    try:
        lines = open(f).readlines()
    except FileNotFoundError:
        print(f"{A:>5}  MISSING LOG")
        continue
    for line in lines:
        m = pat.search(line)
        if not m:
            continue
        ep = int(m.group(1)); step = int(m.group(2)); d = float(m.group(3)); brg = float(m.group(4))
        eps.setdefault(ep, []).append((step, d, brg))
    for ep in sorted(eps):
        rows = eps[ep]
        brg0 = rows[0][2]; d0 = rows[0][1]
        dmin = min(r[1] for r in rows)
        # orbiting flag: reached close (<1.2) then drifted back out (final > dmin + 0.8)
        argmin = min(range(len(rows)), key=lambda i: rows[i][1])
        dfin = rows[-1][1]
        orbit = "ORBIT" if (dmin < 1.2 and dfin > dmin + 0.8) else ""
        ok = dmin < 0.8
        summary[(A, ep)] = (brg0, dmin, ok)
        print(f"{A:>5} {ep:>3} {brg0:>6.0f} {d0:>5.2f} {dmin:>5.2f} {'YES' if ok else 'no':>9}  step@min={rows[argmin][0]:<5} dfin={dfin:.2f} {orbit}")

print("\n=== success vs |initial bearing| ===")
buckets = {'front |brg|<45': [], 'side 45-135': [], 'rear |brg|>135': []}
for (A, ep), (brg0, dmin, ok) in summary.items():
    ab = abs(brg0)
    k = 'front |brg|<45' if ab < 45 else ('side 45-135' if ab <= 135 else 'rear |brg|>135')
    buckets[k].append(ok)
for k, v in buckets.items():
    print(f"{k:>18}: {sum(v)}/{len(v)} reached <0.8m")
tot = sum(1 for v in summary.values() if v[2])
print(f"\nOVERALL: {tot}/{len(summary)} episodes reached <0.8m")
