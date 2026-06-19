#!/usr/bin/env python3
"""Parse [Godot] step lines and summarise CPG locomotion quality.
Line: ...| Step N | ... | Yaw: <deg> | fwd_v: <m/s> | disp: <m> | ..."""
import re, sys, math

label, log, vx, om = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
pat = re.compile(r"Step (\d+).*?Yaw: (-?\d+).*?fwd_v: (-?[\d.]+).*?disp: (-?[\d.]+)")
rows = []
try:
    with open(log) as f:
        for ln in f:
            m = pat.search(ln)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
except FileNotFoundError:
    print(f"  {label}: NO LOG"); sys.exit()

if len(rows) < 3:
    # surface why it produced nothing
    tail = ""
    try:
        with open(log) as f:
            L = f.readlines()
        tail = "".join(L[-4:]).strip().replace("\n", " | ")
    except Exception:
        pass
    print(f"  {label}: EMPTY ({len(rows)} rows) tail=[{tail[:160]}]"); sys.exit()

# dt per decision-step print = 10 decision steps; need wall time. action_repeat & physics dt unknown here,
# so report RATES per print-interval-normalised using step index (relative comparison is what matters).
steps = [r[0] for r in rows]
yaws = [r[1] for r in rows]
fwds = [r[2] for r in rows]
disps = [r[3] for r in rows]

# unwrap yaw
uy = [yaws[0]]
for y in yaws[1:]:
    d = y - (uy[-1] % 360 if False else 0)
    # simple unwrap
    prev = uy[-1]
    cand = y
    while cand - prev > 180: cand -= 360
    while cand - prev < -180: cand += 360
    uy.append(cand)
total_yaw = uy[-1] - uy[0]
nstep = steps[-1] - steps[0]
# per 100 decision-steps (comparable across runs of same length)
yaw_per100 = total_yaw / max(1, nstep) * 100
mean_fwd = sum(fwds[2:]) / len(fwds[2:])   # skip startup
final_disp = disps[-1]
disp_per100 = final_disp / max(1, nstep) * 100
# fwd_v should be ~vx for good straight; for curving run stays high while yaw changes
print(f"  {label:26s} vx={vx:.2f} om={om:+.2f} | fwd_v={mean_fwd:+.2f} | yaw/100={yaw_per100:+6.1f}deg | disp={final_disp:.2f} (disp/100={disp_per100:.2f})")
