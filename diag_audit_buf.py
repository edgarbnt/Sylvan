"""Audit b2f acquisition density d'un buffer (clé de voûte 3b). Usage: python diag_audit_buf.py <buf>"""
import json, glob, math, sys, statistics as st
buf = sys.argv[1]
files = sorted(glob.glob(f"godot/data/replay_buffer/{buf}/episode_*.jsonl") or
               glob.glob(f"data/replay_buffer/{buf}/episode_*.jsonl"))
nep = nfr = rot = behind = vis = b2f = f2b = 0; oms = []
for f in files:
    prev = None
    for line in open(f):
        r = json.loads(line); w = r.get("wm", {})
        fr = w.get("food_rel0"); cmd = w.get("cmd")
        if not fr or not cmd:
            continue
        nfr += 1; om = abs(cmd[1]); oms.append(om)
        if om > 0.3: rot += 1
        if fr[2] > 0.5:
            vis += 1; brg = math.atan2(fr[0], fr[1]); bh = abs(brg) > math.pi / 2
            if bh: behind += 1
            if prev is not None and prev[0]:
                if prev[1] and not bh: b2f += 1
                elif (not prev[1]) and bh: f2b += 1
            prev = (True, bh)
        else:
            prev = (False, False)
    nep += 1
if nfr == 0:
    print(f"[{buf}] vide"); sys.exit()
print(f"[{buf}] {nep}ép {nfr}fr | |ω|={st.mean(oms):.2f} vir={100*rot/nfr:.0f}% "
      f"vis={100*vis/nfr:.0f}% derr={100*behind/max(1,vis):.0f}% | b2f={b2f} f2b={f2b} | "
      f"b2f/ép={b2f/max(1,nep):.1f}")
