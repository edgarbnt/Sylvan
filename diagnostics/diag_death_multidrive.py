"""diag_death_multidrive.py — DIAGNOSTIC GRATUIT : POURQUOI l'entité meurt en multi-pulsions (avant de fixer l'horizon).
Parse le log riche godot (/tmp/bmds_free.log) : par pas Episode/Step/Energy/Thirst/food_d/water_d/om. Pour chaque
épisode mort de faim, caractérise les K derniers pas avant la mort :
  - la bouffe était-elle PROCHE (food_d petit) pendant que l'énergie chutait → il l'IGNORE (arbitrage/oscillation),
    OU LOIN (food_d grand) → trajet/économie (resources trop espacées) ?
  - oscille-t-il (la cible « la plus proche » faim↔soif change souvent) ?
Ça dit ce que le fix d'horizon doit corriger (mieux arbitrer/s'engager vs mieux router les visites).
Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_death_multidrive.py [/tmp/bmds_free.log]
"""
import sys, re, statistics as st

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bmds_free.log"
K = 400  # fenêtre avant la mort
EAT = 1.0
NEAR = 1.5

pat = re.compile(
    r"Episode (\d+) \| Step (\d+) \| Energy: ([\d.]+) \| Thirst: ([\d.]+).*?food_d: (-?[\d.]+) \| water_d: (-?[\d.]+) \| om: (-?[\d.]+)")
eps = {}
for line in open(LOG):
    m = pat.search(line)
    if not m:
        continue
    ep = int(m.group(1))
    eps.setdefault(ep, []).append({
        "step": int(m.group(2)), "E": float(m.group(3)), "T": float(m.group(4)),
        "fd": float(m.group(5)), "wd": float(m.group(6)), "om": float(m.group(7)),
    })


def median(x):
    return st.median(x) if x else float("nan")


print(f"épisodes parsés : {len(eps)}\n")
hunger_eps = 0
agg_near = []   # frac du temps bouffe PROCHE pendant le déclin (mort de faim)
agg_fd = []     # food_d médian dans la fenêtre de mort
agg_osc = []    # oscillations cible la plus proche
for ep in sorted(eps):
    rows = sorted(eps[ep], key=lambda r: r["step"])
    last = rows[-1]
    full = last["step"] >= 2999
    cause = "PLEIN" if full else ("faim" if last["E"] <= 1.5 else ("soif" if last["T"] <= 1.5 else "autre"))
    win = rows[-K:] if len(rows) >= K else rows
    fd_med = median([r["fd"] for r in win if r["fd"] > 0])
    wd_med = median([r["wd"] for r in win if r["wd"] > 0])
    frac_food_near = sum(1 for r in win if 0 < r["fd"] < NEAR) / max(1, len(win))
    frac_food_eatable = sum(1 for r in win if 0 < r["fd"] < EAT) / max(1, len(win))
    # oscillation : la ressource la plus proche bascule faim↔soif
    closer = [("f" if (0 < r["fd"] < r["wd"] or r["wd"] <= 0) else "w") for r in win if r["fd"] > 0 or r["wd"] > 0]
    flips = sum(1 for i in range(1, len(closer)) if closer[i] != closer[i - 1])
    print(f"Ep{ep:>2} [{cause:5}] survie={last['step']:>5} | fenêtre-mort: food_d méd={fd_med:.2f} water_d méd={wd_med:.2f} "
          f"| bouffe<{NEAR}m {100*frac_food_near:.0f}% , <{EAT}m {100*frac_food_eatable:.0f}% | flips cible={flips}")
    if cause == "faim":
        hunger_eps += 1
        agg_near.append(frac_food_near); agg_fd.append(fd_med); agg_osc.append(flips)

print()
if hunger_eps:
    print(f"=== MORTS DE FAIM : {hunger_eps} épisodes ===")
    print(f"food_d médian (fenêtre de mort) = {median(agg_fd):.2f} m")
    print(f"% du temps bouffe PROCHE (<{NEAR}m) pendant le déclin = {100*median(agg_near):.0f}%")
    print(f"flips de cible (oscillation) médian = {median(agg_osc):.0f}")
    fd = median(agg_fd); near = median(agg_near)
    if near > 0.25:
        verdict = "IGNORE une bouffe PROCHE → arbitrage/oscillation (le fix = mieux s'engager/arbitrer, pas juste l'horizon)"
    elif fd > 3.0:
        verdict = "bouffe LOIN pendant le déclin → trajet/économie (resources espacées ; le fix = router les visites / horizon long)"
    else:
        verdict = "intermédiaire → regarder oscillation + distances ci-dessus"
    print(f"\n>>> CAUSE DOMINANTE : {verdict}")
else:
    print("Aucune mort de faim dans ce log (relancer en foresight 0 / vérifier le log).")
