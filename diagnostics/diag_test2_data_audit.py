"""TEST 2 (carte clé de voûte) — audit DATA : le replay a-t-il assez de virages-avec-ACQUISITION pour apprendre
au WM à imaginer la perception sous rotation ? (offline, 0 entraînement).

On compte l'événement-clé = la bouffe BALAYÉE devant↔derrière pendant un virage (ce que le rêve doit apprendre) :
  - %frames en virage (|ω|>0.3), |ω| moyen ;
  - crossings derrière→devant et devant→derrière (|bearing| traverse 90°) AVEC bouffe visible ;
  - durée passée "derrière" (où l'encodeur est faible).
Verdict (jugé, pas de seuil dur) : peu de crossings → recollecte SCRIPTÉE de virages nécessaire (un babbling
aléatoire fait PIRE, cf mémoire) ; beaucoup → la donnée existante peut nourrir GATE 3.

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_test2_data_audit.py
"""
import json, glob, math, statistics as st

for buf in ("retina_forage", "retina_wm_a", "retina_eat_a"):
    files = sorted(glob.glob(f"godot/data/replay_buffer/{buf}/episode_*.jsonl") or
                   glob.glob(f"data/replay_buffer/{buf}/episode_*.jsonl"))
    if not files:
        print(f"[{buf}] introuvable"); continue
    nep = 0; nfr = 0; rot_fr = 0; behind_fr = 0; vis_fr = 0
    cross_b2f = 0; cross_f2b = 0; omegas = []
    for f in files:
        prev = None
        for line in open(f):
            r = json.loads(line); w = r.get("wm", {})
            fr = w.get("food_rel0"); cmd = w.get("cmd")
            if not fr or not cmd:
                continue
            nfr += 1
            om = abs(cmd[1]); omegas.append(om)
            if om > 0.3:
                rot_fr += 1
            vis = fr[2] > 0.5
            if vis:
                vis_fr += 1
                brg = math.atan2(fr[0], fr[1]); behind = abs(brg) > math.pi / 2
                if behind:
                    behind_fr += 1
                if prev is not None and prev[0]:  # both visible
                    if prev[1] and not behind:
                        cross_b2f += 1            # derrière → devant (acquisition par virage)
                    elif (not prev[1]) and behind:
                        cross_f2b += 1            # devant → derrière (perte)
                prev = (True, behind)
            else:
                prev = (False, False)
        nep += 1
    print(f"\n[{buf}] {nep} ép, {nfr} frames")
    print(f"  |ω| moyen={st.mean(omegas):.2f}  %virage(|ω|>0.3)={100*rot_fr/nfr:.0f}%")
    print(f"  bouffe visible={100*vis_fr/nfr:.0f}%  dont derrière={100*behind_fr/max(1,vis_fr):.0f}%")
    print(f"  CROSSINGS derrière→devant (acquisition)={cross_b2f}  devant→derrière={cross_f2b}")
    print(f"  → acquisitions/ép ≈ {cross_b2f/max(1,nep):.1f}")
print("\nLecture : l'acquisition-par-virage (derrière→devant, bouffe visible) est l'événement que le WM doit"
      " apprendre à RÊVER. Peu d'événements → recollecte SCRIPTÉE ciblée (cibles 360° + rotations commandées).")
