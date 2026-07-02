"""F-pure-3 (plan WM object-centric pur) — PERMANENCE : un slot DEAD-RECKONÉ par l'ego-motion tient-il l'objet à
travers une forte réorientation (sans le re-percevoir), là où un slot FIGÉ échouerait ? (offline, gratuit).
Précondition de l'object permanence / mémoire spatiale.

Sur des fenêtres de W pas : slot init = position objet à t0 ; on le transporte par l'ego-motion VRAIE (torso, sans
re-grounding rétine pendant la fenêtre = comme si l'objet était occulté), et on compare le bearing prédit en fin de
fenêtre au vrai. Contrôle = slot FIGÉ (bearing de t0). Bucketé par rotation nette de la fenêtre.

SUCCÈS = sur forte rotation, DEAD-RECKON corr ≫ FIGÉ corr (le slot persiste à travers la réorientation).

Usage: BUF=retina_eat_a W=30 PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_fpure3_permanence.py
"""
import json, glob, math, os, statistics as st
import torch

BUF = os.environ.get("BUF", "retina_eat_a"); W = int(os.environ.get("W", "30"))
files = sorted(glob.glob(f"godot/data/replay_buffer/{BUF}/episode_*.jsonl") or
               glob.glob(f"data/replay_buffer/{BUF}/episode_*.jsonl"))[:60]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


SY = float(os.environ.get("SY", "-1"))   # signe de rotation
SL = float(os.environ.get("SL", "1"))    # signe translation latérale


def transport(p, dyaw, dfwd, dlat):
    px = p[0] - SL * dlat; pz = p[1] - dfwd
    ca, sa = math.cos(SY * dyaw), math.sin(SY * dyaw)
    return [ca * px - sa * pz, sa * px + ca * pz]


def corr(a, b):
    a = torch.tensor(a); b = torch.tensor(b); a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm(); return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


eps = []
for f in files:
    seq = []
    for line in open(f):
        w = json.loads(line).get("wm", {}); fr = w.get("food_rel0"); t0 = w.get("torso0")
        if not fr or not t0:
            continue
        seq.append((float(fr[0]), float(fr[1]), float(fr[2]), t0[0], t0[1], t0[2]))
    if len(seq) > W + 2:
        eps.append(seq)

# par fenêtre : dead-reckon vs figé vs vrai, à la fin. net rotation = |Σ dyaw|
buckets = {"faible <20°": [], "moyen 20-60°": [], "fort >60°": []}
for seq in eps:
    for s in range(0, len(seq) - W, max(1, W // 2)):
        win = seq[s:s + W + 1]
        if any(f[2] < 0.5 for f in win):     # objet visible tout du long (pour avoir la vérité)
            continue
        slot = [win[0][0], win[0][1]]
        netyaw = 0.0
        for i in range(W):
            a = win[i]; b = win[i + 1]
            dyaw = wrap(b[5] - a[5]); dx, dz = b[3] - a[3], b[4] - a[4]
            dfwd = dx * math.sin(a[5]) + dz * math.cos(a[5]); dlat = dx * math.cos(a[5]) - dz * math.sin(a[5])
            slot = transport(slot, dyaw, dfwd, dlat); netyaw += dyaw
        br_pred = math.atan2(slot[0], slot[1])
        br_true = math.atan2(win[-1][0], win[-1][1])
        br_froz = math.atan2(win[0][0], win[0][1])
        deg = abs(math.degrees(netyaw))
        k = "faible <20°" if deg < 20 else ("moyen 20-60°" if deg < 60 else "fort >60°")
        buckets[k].append((br_pred, br_true, br_froz))

print(f"BUF={BUF} W={W} — PERMANENCE par rotation nette de la fenêtre :")
print(f"{'bucket':>14} | {'n':>5} | {'DEAD-RECKON':>11} | {'FIGÉ':>6}")
strong_dr = strong_fr = None
for k, v in buckets.items():
    if not v:
        print(f"{k:>14} |   0   |     -      |   -"); continue
    dr = corr([x[0] for x in v], [x[1] for x in v]) if len(v) > 2 else float("nan")
    fr = corr([x[2] for x in v], [x[1] for x in v]) if len(v) > 2 else float("nan")
    print(f"{k:>14} | {len(v):>5} | {dr:>+11.2f} | {fr:>+6.2f}")
    if k.startswith("fort"):
        strong_dr, strong_fr = dr, fr
print()
if strong_dr is not None and not math.isnan(strong_dr):
    gain = strong_dr - (strong_fr if strong_fr == strong_fr else 0)
    print(f"SUR FORTE ROTATION : dead-reckon {strong_dr:+.2f} vs figé {strong_fr:+.2f} (gain {gain:+.2f})")
    print(">>> " + ("SUCCÈS : le slot dead-reckoné PERSISTE à travers la réorientation (≫ figé) → permanence viable."
                   if (strong_dr >= 0.6 and strong_dr - strong_fr >= 0.2) else
                   "PARTIEL : la persistance aide peu ici (bearing trop lent ?) — voir gain."))
else:
    print(">>> pas assez de fenêtres à forte rotation (objet visible+statique) pour trancher.")
