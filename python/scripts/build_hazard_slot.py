"""BUILD (pas un entraînement) — ajoute un 3ᵉ slot DANGER (vert) au WM object-centric, ZÉRO retrain.

POURQUOI ça marche sans entraîner (sonde diag_hazard_slot.py, 2026-07-15) : le slot lit la RÉTINE BRUTE
et localise par attention-couleur GÉOMÉTRIQUE (le scoreur appris est retiré des logits quand la requête-
couleur est active → les MLP .score sont des poids MORTS). Donc passer slot_resources 2→3 ajoute une
requête verte [0,1,0] et un slot danger localisé — sans toucher aux poids de base ni aux slots bouffe/eau.
C'est la même leçon que slot-1/slot-2 : « le slot était déjà là ».

Charge le WM 2-slots, ré-instancie en 3-slots (color_queries = rouge,bleu,VERT), copie les poids de base
en strict=False (slot_encoder.score.2 = aléatoire mais INUTILISÉ ; color_queries [2,3]→[3,3] remplacé par la
version fraîche), écrit meta.slot_resources=3 + hazard_idx=2, sauve un nouveau checkpoint.

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.build_hazard_slot \
            [--src data/checkpoints/wm_objcentric_kin/wm_best.pt] [--out data/checkpoints/wm_objcentric_kin_haz]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sylvan.models.command_wm import CommandWorldModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--out", default="data/checkpoints/wm_objcentric_kin_haz")
    args = ap.parse_args()

    torch.manual_seed(0)   # score.2 est init aléatoire (INUTILISÉ, readout géométrique) — seed → build reproductible
    payload = torch.load(args.src, map_location="cpu", weights_only=False)
    meta = dict(payload["meta"])
    src_res = meta.get("slot_resources", 1)
    if src_res != 2:
        print(f"⚠️ attendu slot_resources=2 (bouffe,eau), trouvé {src_res} — vérifier la source.")

    # WM 3-slots : rouge(bouffe)=0, bleu(eau)=1, VERT(danger)=2 (color_queries dans SelfSupervisedSlotHead).
    wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                           predictor_arch=meta.get("predictor_arch", "shallow"),
                           with_slot=meta.get("with_slot", False), slot_resources=3)
    # strict=False autorise les clés MANQUANTES/EN-TROP mais PAS une taille ≠ sur une clé commune. Le
    # checkpoint a color_queries [2,3], le module 3-slots [3,3] → on RETIRE la clé du checkpoint pour qu'elle
    # soit « manquante » → la version fraîche [rouge,bleu,VERT] du module est conservée (c'est le but).
    state = dict(payload["model"])
    state.pop("slot_encoder.color_queries", None)
    missing, unexpected = wm.load_state_dict(state, strict=False)
    wm.eval()

    # garde-fou : la 3ᵉ requête DOIT être verte (sinon le slot danger ne localiserait pas le vert).
    cq = wm.slot_encoder.color_queries
    assert cq.shape == (3, 3), f"color_queries {cq.shape} ≠ (3,3)"
    green = cq[2] / cq[2].norm()
    assert torch.allclose(green, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5), f"3ᵉ requête ≠ verte : {cq[2]}"
    # les 2 premières requêtes (bouffe/eau) inchangées
    assert torch.allclose(cq[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(cq[1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    # seuls score.2.* (3ᵉ MLP inutilisé) et color_queries (retiré exprès, remplacé par la version verte)
    # doivent manquer ; tout autre manque = poids de base perdu = build cassé.
    bad_missing = [k for k in missing if "score.2" not in k and "color_queries" not in k]
    assert not bad_missing, f"poids de base manquants (inattendu) : {bad_missing}"
    print(f"[build] chargé strict=False : {len(missing)} manquants (tous score.2, inutilisés), "
          f"{len(unexpected)} inattendus")

    meta["slot_resources"] = 3
    meta["hazard_idx"] = 2
    meta["food_idx"] = meta.get("food_idx", 0)
    meta["water_idx"] = meta.get("water_idx", 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": wm.state_dict(), "meta": meta}, out / "wm_best.pt")
    print(f"[build] WM 3-slots (bouffe=0, eau=1, DANGER=2) sauvé → {out / 'wm_best.pt'}")
    print(f"[build] slot_resources={meta['slot_resources']} hazard_idx={meta['hazard_idx']} — zéro entraînement.")


if __name__ == "__main__":
    main()
