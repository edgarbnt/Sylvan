"""Re-mesure des requêtes-couleur et seuils pour le monde foret_v2_slot.

Le slot actuel utilise des requêtes (0.876, 0.349, 0.333) mesurées sur le monde
wm_objcentric_kin_typed. Mais le monde foret a des BOSQUETS avec buissons-marqueurs
VERTS (PATCH_BUSH_COLOR = (0.47, 0.93, 0.53)) sur la couche rétine. Ces buissons
masquent les baies et sont ce que le slot voit réellement. Les requêtes rouges
ne matchent pas le vert du buisson → cos=0.78 < seuil 0.808 → exclus → position bruitée.

Ce script refait le k-means sur les couleurs des rayons qui touchent la ressource
la plus proche, et re-mesure les marges par type.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python scripts/remeasure_queries_foret.py
"""

from __future__ import annotations

import json
import math
import glob
import sys
import os

import numpy as np

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from sylvan.models.command_wm import CommandWorldModel

NRAY = 36
HALF = NRAY // 2
CONTACT_M = 1.5          # portée de consommation
CORPUS = "data/replay_buffer/gate_foret_cl"
WM_PATH = "data/checkpoints/wm_foret_v2_slot/wm_best.pt"
OUT_PATH = "data/checkpoints/wm_foret_v2_slot/wm_remeasured.pt"


def _ray_angle(k: int) -> float:
    return (k if k <= HALF else k - NRAY) * 120.0 / NRAY


def _nearest_ray(bearing_deg: float) -> int:
    return min(range(NRAY), key=lambda k: abs(bearing_deg - _ray_angle(k)))


def main() -> int:
    print("=== RE-MESURE DES REQUÊTES-COULEUR POUR foret_v2 ===")

    # ── 1. Collecter les couleurs des rayons touchant la ressource ──
    colors_by_drive = {"energy": [], "thirst": [], "damage": []}

    for f in sorted(glob.glob(os.path.join(CORPUS, "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            obs = r.get("obs", {})
            wm = r.get("wm", {})
            energy = obs.get("energy", 60.0)
            thirst = obs.get("thirst", 60.0)
            health = obs.get("health", 100.0)

            # Pour chaque drive, trouver la position de la ressource la plus proche
            for drive, key in [("energy", "food_rel0"), ("thirst", "water_rel0")]:
                v = wm.get(key)
                if not v or len(v) < 3 or v[2] <= 0.5:
                    continue
                bearing = math.degrees(math.atan2(v[0], v[1]))
                if abs(bearing) > 60.0:
                    continue

                # Trouver le rayon pointant vers cette ressource
                best_idx = _nearest_ray(bearing)
                retina = obs.get("retina", [])
                if len(retina) != 144:
                    continue

                d = retina[best_idx * 4]
                ray_dist = d * 10.0 + 0.35
                rgb = retina[best_idx * 4 + 1: best_idx * 4 + 4]

                # On NE FILTRE PAS par hit (profondeur ≈ distance vraie) : le slot voit
                # le RAYON à cette direction, que ce soit la baie ou le BUISSON-MARQUEUR.
                # Dans le monde foret, les buissons verts (PATCH_BUSH_COLOR) masquent les
                # baies sur la couche rétine → ce qu'on veut, c'est la couleur que la rétine
                # voit au bearing de la ressource.
                norm = math.sqrt(rgb[0] ** 2 + rgb[1] ** 2 + rgb[2] ** 2)
                if norm < 0.01:
                    continue
                normalized = (rgb[0] / norm, rgb[1] / norm, rgb[2] / norm)
                colors_by_drive[drive].append(normalized)

    print(f"\nRayons collectés :")
    for drive, cols in colors_by_drive.items():
        print(f"  {drive}: {len(cols)} samples")

    # ── 2. K-means sur chaque drive ──
    new_queries = []
    new_thresholds = []
    drive_order = []

    for drive in ["energy", "thirst", "damage"]:
        samples = colors_by_drive[drive]
        if len(samples) < 5:
            print(f"  {drive}: pas assez d'échantillons ({len(samples)}), saute")
            continue

        X = np.array(samples)

        # K=1 pour chaque drive (un seul prototype par pulsion)
        # prototype = moyenne directionnelle (normalisée)
        prototype = X.mean(axis=0)
        prototype = prototype / np.linalg.norm(prototype)

        # Mesure de la marge : cos q05 intra-groupe
        cos_intra = X @ prototype
        q05 = float(np.percentile(cos_intra, 5))

        # Mesure inter-groupe : cos max avec les autres prototypes
        other_cos = []
        for other_q in new_queries:
            other_cos.append(float(prototype @ other_q))
        if other_cos:
            q995 = float(np.percentile(other_cos, 99.5)) if other_cos else 0.0
            threshold = (q05 + q995) / 2.0
        else:
            q995 = 0.0
            threshold = q05 - 0.05  # fallback

        new_queries.append(prototype.tolist())
        new_thresholds.append(threshold)
        drive_order.append(drive)

        print(f"\n  {drive}:")
        print(f"    Prototype: ({prototype[0]:.4f}, {prototype[1]:.4f}, {prototype[2]:.4f})")
        print(f"    Cos intra q05: {q05:.4f}")
        print(f"    Seuil: {threshold:.4f}")
        if other_cos:
            print(f"    Cos inter max: {max(other_cos):.4f}")

    # ── 3. Appliquer au checkpoint ──
    print(f"\n=== Application au checkpoint {WM_PATH} ===")
    payload = torch.load(WM_PATH, map_location="cpu", weights_only=False)
    meta = payload["meta"]

    # Créer les tenseurs de requêtes (normalisées)
    q_tensor = torch.tensor(new_queries, dtype=torch.float32)
    q_tensor = q_tensor / q_tensor.norm(dim=-1, keepdim=True)
    thr_tensor = torch.tensor(new_thresholds, dtype=torch.float32)

    # Retirer les vestiges de l'ancien attention MLP (score.2.*)
    keys_to_remove = [k for k in payload["model"] if k.startswith("slot_encoder.score.2.")]
    for k in keys_to_remove:
        del payload["model"][k]
    print(f"  Retiré {len(keys_to_remove)} clés obsolètes (attention MLP)")

    # Mettre à jour le state_dict du slot_encoder
    payload["model"]["slot_encoder.color_queries"] = q_tensor

    # Mettre à jour le meta
    meta["query_thr"] = new_thresholds
    meta["food_idx"] = drive_order.index("energy") if "energy" in drive_order else 0
    meta["water_idx"] = drive_order.index("thirst") if "thirst" in drive_order else 1
    meta["hazard_idx"] = drive_order.index("damage") if "damage" in drive_order else 2
    meta["slot_resources"] = len(new_queries)

    # Validate
    print(f"\n  Nouvelles requêtes:")
    for i, (q, thr) in enumerate(zip(new_queries, new_thresholds)):
        print(f"    Slot {i} ({drive_order[i]}): ({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}) thr={thr:.4f}")

    # Sauvegarder
    torch.save(payload, OUT_PATH)
    print(f"\n✅ Checkpoint sauvegardé: {OUT_PATH}")

    # ── 4. Vérification rapide : slot avec nouvelles requêtes ──
    print(f"\n=== Vérification rapide via diag_slot_localise ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
