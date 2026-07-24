"""TEST GRATUIT — une cible qui BOUGE rendrait-elle la prédiction indispensable ?

POURQUOI. Tout ce qu'on a ajouté au monde jusqu'ici décorait une tâche d'ALLER-CHERCHER un point
FIXE. Mesuré le 2026-07-24 : le déplacement du corps est reconstructible à R² 0,985 depuis la
commande SEULE (une droite), et la géométrie seule prédit le retour mieux que géométrie+latent
(0,179 vs 0,149). Autrement dit, ce monde n'a presque aucune dynamique à apprendre, donc `-min_dist`
est près de l'optimum et aucun critique n'a de marge.

Une cible qui FUIT change ça de façon PROUVABLE : viser où la proie EST ne l'attrape pas, il faut
viser où elle SERA. La prédiction devient alors la chose qui décide.

CE QUE CE TEST TRANCHE, AVANT TOUT DÉVELOPPEMENT. On simule analytiquement (aucun Godot, aucun WM)
deux poursuivants aux capacités IDENTIQUES, avec les constantes MESURÉES de notre corps :
  * POURSUITE  : viser la position ACTUELLE de la proie  <- ce que fait `-min_dist`
  * INTERCEPTION : viser la position FUTURE prédite       <- ce qu'un WM+critique permettrait
et on mesure l'écart en fonction de la vitesse de la proie.

  écart nul aux vitesses réalistes  -> l'idée est MORTE pour zéro run, on n'y va pas.
  écart net                         -> on sait AVANT de coder qu'un critique aurait enfin une marge,
                                       ce qu'on n'a JAMAIS pu dire des baies.

⚠️ Contrôle obligatoire : à vitesse de proie NULLE les deux politiques doivent être IDENTIQUES.
Si elles diffèrent, le simulateur est biaisé et le résultat ne vaut rien.

⚠️⚠️ PIÈGE DÉGÉNÉRÉ, RENCONTRÉ PUIS CORRIGÉ (2026-07-24). Une proie qui FUIT converge vers une
trajectoire RADIALE (droit à l'opposé de l'agent) ; contre une cible radiale l'angle d'avance est nul
PAR CONSTRUCTION, donc poursuite et interception COÏNCIDENT mathématiquement. Le premier jet de ce
test utilisait une telle proie et rendait « écart nul » à toutes les vitesses ET à toutes les
agilités — un faux négatif qui m'avait fait conclure à tort que le CORPS était trop agile.
La proie doit donc VAQUER (mouvement transversal conservé), pas fuir.

RÉSULTAT (proie qui vaque, agilité = NOTRE corps actuel, 600 poursuites/point) :
  v_proie/v_agent | poursuite | interception |
      0,0         |  100 %    |   100 %      | contrôle OK
      0,3         |  100 %    |   100 %      | ~0
      0,6         |  100 %    |   100 %      | -7 % de TEMPS
      0,9         |  56,2 %   |   67,5 %     | +11 points de capture
      1,2         |  18,7 %   |   33,0 %     | +14 points (presque le double)
SPÉC. DE CONCEPTION : le levier n'existe que si la proie (a) a du mouvement TRANSVERSAL et (b) est
RAPIDE (>= 0,9x la vitesse de l'agent). Le corps actuel suffit — aucune inertie à ajouter.

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_prey_interception.py
"""
from __future__ import annotations

import argparse
import math

import torch

# Constantes MESURÉES de notre corps (cf CLAUDE.md + mesures 2026-07) — on ne teste pas un corps
# imaginaire, on teste LE NÔTRE.
SPEED = 0.011          # m/tick (mesuré, kin_speed 0.8)
DT = 0.05              # pas de simulation Godot
MAX_TURN = 1.5 * DT    # rad/tick (kin_turn 1.5)
CAPTURE = 1.0          # m (eat_radius)
MAX_TICKS = 2000


def run(n: int, ratio: float, mode: str, seed: int) -> tuple[float, float]:
    """n poursuites indépendantes. → (taux de capture, temps médian de capture)."""
    g = torch.Generator().manual_seed(seed)
    # départ : proie à 3-11 m (comme nos ressources), azimut quelconque, cap agent quelconque
    d0 = 3.0 + 8.0 * torch.rand(n, generator=g)
    a0 = 2 * math.pi * torch.rand(n, generator=g)
    prey = torch.stack([d0 * torch.cos(a0), d0 * torch.sin(a0)], 1)
    pos = torch.zeros(n, 2)
    head = 2 * math.pi * torch.rand(n, generator=g)
    # direction initiale de la proie : quelconque (elle vit sa vie, elle ne fait pas que fuir)
    pdir = 2 * math.pi * torch.rand(n, generator=g)
    pdir = torch.stack([torch.cos(pdir), torch.sin(pdir)], 1)

    caught = torch.zeros(n, dtype=torch.bool)
    t_catch = torch.full((n,), float(MAX_TICKS))
    vp = SPEED * ratio

    for t in range(MAX_TICKS):
        rel = prey - pos
        dist = rel.norm(dim=1)
        newly = (~caught) & (dist <= CAPTURE)
        t_catch[newly] = float(t)
        caught |= newly
        if bool(caught.all()):
            break

        if mode == "pursuit":
            target = prey                                   # viser où elle EST
        else:
            # INTERCEPTION : temps d'atteinte estimé, puis viser où elle sera alors.
            # (deux itérations suffisent à converger ; c'est ce qu'un rollout du WM ferait)
            tau = dist / max(SPEED, 1e-9)
            for _ in range(2):
                fut = prey + pdir * vp * tau.unsqueeze(1)
                tau = (fut - pos).norm(dim=1) / max(SPEED, 1e-9)
            target = prey + pdir * vp * tau.unsqueeze(1)

        want = torch.atan2(target[:, 1] - pos[:, 1], target[:, 0] - pos[:, 0])
        err = torch.remainder(want - head + math.pi, 2 * math.pi) - math.pi
        head = head + err.clamp(-MAX_TURN, MAX_TURN)        # virage BORNÉ : c'est ce qui coûte
        step = torch.stack([torch.cos(head), torch.sin(head)], 1) * SPEED
        pos = torch.where(caught.unsqueeze(1), pos, pos + step)

        # PROIE : elle garde sa direction (inertie) et s'écarte un peu de l'agent quand il approche.
        # C'est ce qui crée du mouvement TRANSVERSAL — sans quoi poursuite et interception
        # coïncideraient trivialement (une proie qui fuit en ligne droite radiale ne teste rien).
        away = (prey - pos) / (prey - pos).norm(dim=1, keepdim=True).clamp_min(1e-6)
        near = (dist < 3.0).float().unsqueeze(1)
        nd = pdir * 0.97 + away * 0.03 + near * away * 0.15
        pdir = nd / nd.norm(dim=1, keepdim=True).clamp_min(1e-6)
        prey = torch.where(caught.unsqueeze(1), prey, prey + pdir * vp)

    rate = float(caught.float().mean())
    med = float(t_catch[caught].median()) if bool(caught.any()) else float("nan")
    return rate, med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"corps MESURÉ : {SPEED} m/tick, virage max {MAX_TURN:.3f} rad/tick, capture {CAPTURE} m, "
          f"{MAX_TICKS} ticks max | {args.n} poursuites par point\n")
    print("  v_proie/v_agent |   POURSUITE (-min_dist)   |    INTERCEPTION (prédiction)  |  écart")
    print("                  | capture   temps médian    | capture   temps médian        |")
    for ratio in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75):
        rp, tp = run(args.n, ratio, "pursuit", args.seed)
        ri, ti = run(args.n, ratio, "intercept", args.seed)
        gain = ri - rp
        flag = ""
        if ratio == 0.0:
            flag = "  <- CONTRÔLE : doit être identique" if abs(gain) < 1e-9 else "  <- ⚠ BIAISÉ"
        print(f"       {ratio:.2f}       |  {rp:5.1%}      {tp:7.0f}      |  {ri:5.1%}      {ti:7.0f}"
              f"         | {gain:+6.1%}{flag}")

    print("\n  LECTURE : un écart de capture NET aux ratios réalistes (0,3-0,6) signifie que viser")
    print("  la position actuelle est démontrablement sous-optimal -> un critique/WM aurait enfin")
    print("  une marge que la géométrie instantanée ne peut pas atteindre. Écart nul -> piste MORTE.")


if __name__ == "__main__":
    main()
