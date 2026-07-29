"""LE SLOT LOCALISE-T-IL VRAIMENT LA RESSOURCE DANS CE MONDE ? (offline, gratuit, zéro entraînement)

POURQUOI. Le gate closed-loop du 2026-07-29 échoue alors que le WM est excellent (A1 99,9 %,
open-loop 0,051 m à h=50). Le diagnostic a montré que ce n'est ni le monde ni un blocage : la
nourriture est VISIBLE au tick 0 dans 11 vies sur 12, un premier repas coûte 3,9 m quand il arrive,
et pourtant 6 vies sur 12 parcourent 16,5 m sans jamais manger. L'échec est donc dans la DÉCISION.

Le suspect le moins cher est nommé dans le harnais lui-même : le canal-slot est GREFFÉ, ses
requêtes-couleur viennent de l'ANCIEN monde. Si le slot situe mal la nourriture, le planner vise à
côté et tout le reste du raisonnement sur l'arbitrage serait bâti sur du sable. On l'écarte AVANT
de théoriser — et on l'écarte offline, sans Godot, sans entraînement.

CE QU'ON COMPARE. La coordonnée rendue par le canal-slot du WM SERVI, contre la position vraie de la
ressource la plus proche, que Godot écrit dans le corpus (`wm.food_rel0` = [dx, dz, visible]). Ce
n'est pas un label externe fabriqué pour l'occasion : c'est la vérité-terrain que le monde a
enregistrée pendant que l'entité vivait, sur les ticks où elle voyait effectivement quelque chose.

CRITÈRES PRÉ-ENREGISTRÉS (barre historique du projet : slot ≤ 1,0 m, mesuré 0,35 m bouffe / 0,84 eau) :
  T1 PRÉCISION ... erreur médiane ≤ 1,0 m sur les ticks où la ressource est visible.
                   Au-delà, le planner vise un fantôme et le gate closed-loop est ininterprétable.
  T2 PORTANCE .... au moins 60 % des ticks visibles doivent être exploitables (slot non dégénéré).
  KILL ........... erreur médiane > 2,0 m ⇒ le canal greffé est INVALIDE pour ce monde ; refaire les
                   requêtes (build_typed_slots) AVANT toute conclusion sur l'arbitrage.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_slot_localise.py \
        --wm data/checkpoints/wm_foret_v2_slot/wm_best.pt --corpus data/replay_buffer/gate_foret_cl
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_slot_localise.py --selfcheck
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.models.command_wm import CommandWorldModel  # noqa: E402

PRECISION_BAR = 1.0     # m — barre historique du projet
KILL_BAR = 2.0          # m — au-delà, le canal greffé est invalide pour ce monde
COVERAGE_BAR = 0.60     # part des ticks visibles où le slot rend une coordonnée exploitable
RETINA_DIM = 144


def load_pairs(corpus: str, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(rétines, positions vraies) sur les ticks où la ressource est VISIBLE.

    On ne garde que les ticks visibles parce que sur les autres il n'y a pas de vérité à comparer :
    juger le slot là où le monde lui-même ne voit rien mesurerait notre propre convention, pas lui.
    """
    ret, tgt = [], []
    for f in sorted(glob.glob(os.path.join(corpus, "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("wm", {}).get(key)
            if not v or len(v) < 3 or v[2] <= 0.5:
                continue
            ret.append(r["obs"]["retina"])
            tgt.append([v[0], v[1]])
    if not ret:
        raise SystemExit(f"aucun tick avec {key} visible dans {corpus}")
    return torch.tensor(ret, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)


def measure(wm_path: str, corpus: str, key: str, slot_idx: int) -> dict:
    payload = torch.load(wm_path, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    if not meta.get("with_slot"):
        raise SystemExit(f"{wm_path} n'a pas de canal-slot")
    wm = CommandWorldModel.from_checkpoint(payload) if hasattr(CommandWorldModel, "from_checkpoint") \
        else None
    if wm is None:
        raise SystemExit("CommandWorldModel.from_checkpoint absent — construire à la main serait faux")
    wm.load_state_dict(payload["model"])
    # 🚨 LE CÔNE, SANS QUOI CE DIAGNOSTIC MESURE SA PROPRE ERREUR (attrapé le 2026-07-29 : la
    # première version rendait 4,72 m et j'ai failli en conclure que le canal greffé était mort).
    # Les angles des rayons du slot sont des buffers PERSISTANTS : load_state_dict vient de
    # restaurer ceux de la rétine 360° du checkpoint, alors que ce monde sert un cône de 120°.
    # serve_planner_command les recalcule au chargement ; un diagnostic qui ne le fait pas lit la
    # rétine du cône avec la table d'angles de la 360° et rend des positions fausses SANS RIEN
    # SIGNALER — il accuserait le slot d'un défaut qui serait le sien. On réplique donc exactement
    # ce que fait le serveur, à partir de la MÊME variable d'environnement.
    fov = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360"))
    if abs(fov - 360.0) > 1e-6 and getattr(wm.slot_encoder, "sin", None) is not None:
        n = wm.slot_encoder.sin.shape[0]
        th = torch.tensor([(k if k <= n // 2 else k - n) * math.radians(fov) / n
                           for k in range(n)], dtype=torch.float32)
        with torch.no_grad():
            wm.slot_encoder.sin.copy_(torch.sin(th))
            wm.slot_encoder.cos.copy_(torch.cos(th))
        print(f"  [cône {fov:.0f}° appliqué : {n} rayons redistribués, comme le fait le serveur]")
    wm.eval()

    retina, truth = load_pairs(corpus, key)
    with torch.no_grad():
        # positions() rend [B, R, 2] : une coordonnée ego par ressource requêtée.
        pos = wm.slot_encoder.positions(retina)
    if pos.dim() != 3 or pos.shape[1] <= slot_idx:
        raise SystemExit(f"slot {slot_idx} absent (sortie {tuple(pos.shape)})")
    pred = pos[:, slot_idx, :]

    err = (pred - truth).norm(dim=1)
    live = (pred.abs().sum(dim=1) > 1e-6)          # slot dégénéré (0,0) = pas de coordonnée rendue
    e = err[live]
    return {
        "n": int(len(err)), "coverage": float(live.float().mean()),
        "median": float(e.median()) if len(e) else float("nan"),
        "p90": float(e.quantile(0.9)) if len(e) else float("nan"),
        "truth_dist": float(truth.norm(dim=1).median()),
    }


def render(name: str, m: dict) -> bool:
    print(f"\n  {name}")
    print(f"    ticks visibles           {m['n']}")
    print(f"    distance VRAIE (médiane) {m['truth_dist']:.2f} m")
    print(f"    portance du slot         {100 * m['coverage']:.0f} %  (barre {100 * COVERAGE_BAR:.0f} %)")
    print(f"    ERREUR médiane           {m['median']:.2f} m  (barre {PRECISION_BAR:.1f} m)")
    print(f"    erreur p90               {m['p90']:.2f} m")
    # LE TÉMOIN NUL, et c'est lui qui rend le verdict inattaquable. Un prédicteur qui répondrait
    # toujours « (0,0), c'est à mes pieds » commet une erreur égale à la distance vraie. Si le slot
    # fait PIRE que ça, il ne porte aucune information exploitable : le planner ferait mieux de
    # foncer droit devant. Comparer à une barre absolue seule laisserait planer le doute ;
    # comparer au témoin nul le lève.
    print(f"    témoin NUL (prédire 0,0) {m['truth_dist']:.2f} m"
          f"  → le slot fait {'PIRE' if m['median'] > m['truth_dist'] else 'mieux'}"
          f" ({m['median'] / m['truth_dist']:.2f}x)")
    if math.isnan(m["median"]) or m["median"] > KILL_BAR:
        print(f"    🛑 KILL : le canal greffé est INVALIDE ici — refaire les requêtes avant"
              " toute conclusion sur l'arbitrage.")
        return False
    ok = m["median"] <= PRECISION_BAR and m["coverage"] >= COVERAGE_BAR
    print(f"    {'✅ le slot localise correctement' if ok else '⚠️  imprécis mais pas absurde'}")
    return ok


def selfcheck() -> int:
    m = {"n": 100, "coverage": 0.9, "median": 0.3, "p90": 0.8, "truth_dist": 4.0}
    assert render("cas BON", m)
    assert not render("cas KILL", dict(m, median=3.1))
    assert not render("cas PORTANCE FAIBLE", dict(m, coverage=0.2))
    print("\nSELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2_slot/wm_best.pt")
    ap.add_argument("--corpus", default="data/replay_buffer/gate_foret_cl")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    print(f"=== LE SLOT LOCALISE-T-IL ? | {a.wm} | {a.corpus} ===")
    ok = True
    for name, key, idx in (("NOURRITURE (slot 0)", "food_rel0", 0), ("EAU (slot 1)", "water_rel0", 1)):
        try:
            ok &= render(name, measure(a.wm, a.corpus, key, idx))
        except SystemExit as exc:
            print(f"\n  {name}\n    ⚠️  {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
