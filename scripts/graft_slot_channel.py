"""GREFFE le CANAL-SLOT d'un WM sur un autre. Zéro entraînement, zéro gradient.

POURQUOI C'EST LÉGITIME, ET PAS UN RACCOURCI. Le canal-slot ne contient AUCUNE connaissance apprise
du monde : c'est de la géométrie (`slot_calib` = le transport fixe (1,−1,−1), `sin`/`cos` = la table
d'angles des 36 rayons) plus les requêtes-couleur. Son readout est géométrique zéro-paramètre — le
petit scoreur MLP est calculé puis intégralement ÉCRASÉ par la branche géométrique (CLAUDE.md §2bis :
« le slot a zéro paramètre appris »). Et il lit la RÉTINE, jamais le latent (`encode_slot` découpe
obs[proprio_dim : proprio_dim+144]) : il ne dépend donc pas des poids du WM sur lequel on le pose.

POURQUOI ON EN A BESOIN. Un WM ré-entraîné sort SANS canal-slot (le warm-start écarte ces clés :
le modèle de base ne les déclare pas). Or `build_typed_slots` — l'outil qui fait tomber le verrou A2
en remplaçant les requêtes CODÉES-MAIN par celles DÉCOUVERTES du vécu — exige un WM qui possède déjà
`slot_encoder.color_queries`. La greffe fournit la STRUCTURE ; c'est `build_typed_slots` qui, ensuite,
en re-dérive les VALEURS depuis le corpus du nouveau monde. On ne recycle donc aucune connaissance
d'apparence de l'ancien monde : uniquement le câblage géométrique.

⚠️ CE QUE LA GREFFE NE FAIT PAS : elle ne rend pas les requêtes valides pour le nouveau monde. Tant
que `build_typed_slots` n'a pas tourné, les requêtes greffées sont celles de l'ANCIEN monde — donc
A2 n'est PAS levé. La greffe est une étape, pas un aboutissement.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python scripts/graft_slot_channel.py \
        --dst data/checkpoints/wm_foret_v1/wm_best.pt \
        --src data/checkpoints/wm_objcentric_kin_typed/wm_best.pt \
        --out data/checkpoints/wm_foret_v1_slot/wm_best.pt
    PYTHONPATH=python env_pytorch_3.12/bin/python scripts/graft_slot_channel.py --selfcheck
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

SLOT_PREFIXES = ("slot_calib", "slot_encoder.")


def slot_tensors(state: dict) -> dict:
    """Les tenseurs du canal-slot, et EUX SEULS."""
    return {k: v for k, v in state.items() if k.startswith(SLOT_PREFIXES)}


def graft(dst_payload: dict, src_payload: dict) -> tuple[dict, list[str]]:
    """Pose le canal-slot de `src` sur `dst`. Renvoie (payload greffé, clés copiées)."""
    src_slots = slot_tensors(src_payload["model"])
    if not src_slots:
        raise SystemExit("la source ne porte AUCUN canal-slot — rien à greffer")
    if slot_tensors(dst_payload["model"]):
        raise SystemExit("la cible porte DÉJÀ un canal-slot — refus d'écraser en silence")

    state = dict(dst_payload["model"])
    state.update({k: v.clone() for k, v in src_slots.items()})
    out = dict(dst_payload)
    out["model"] = state

    src_meta, meta = src_payload.get("meta", {}), dict(dst_payload.get("meta", {}))
    meta["with_slot"] = True
    meta["slot_resources"] = int(src_meta.get("slot_resources", 1))
    # Traçabilité : d'où vient le câblage, et le rappel que les VALEURS restent à re-dériver.
    meta["slot_note"] = (f"canal greffé (géométrie + requêtes) depuis {src_meta.get('name', 'src')} ; "
                         "requêtes NON re-dérivées : lancer build_typed_slots pour lever A2")
    if "query_thr" in src_meta:
        meta["query_thr"] = src_meta["query_thr"]
    out["meta"] = meta
    return out, sorted(src_slots)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", help="WM ré-entraîné, SANS canal-slot")
    ap.add_argument("--src", help="WM porteur du canal-slot à copier")
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if not (a.dst and a.src and a.out):
        raise SystemExit("usage : --dst <wm.pt> --src <wm.pt> --out <wm.pt>")

    dst = torch.load(a.dst, map_location="cpu", weights_only=False)
    src = torch.load(a.src, map_location="cpu", weights_only=False)
    out, copied = graft(dst, src)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"canal-slot greffé : {len(copied)} tenseurs, slot_resources="
          f"{out['meta']['slot_resources']} | proprio_dim={out['meta'].get('proprio_dim')} "
          f"obs_dim={out['meta'].get('obs_dim')}")
    for k in copied:
        print(f"    {k} {tuple(out['model'][k].shape)}")
    print(f"-> {a.out}")
    print("⚠️  requêtes encore celles de la SOURCE : lancer build_typed_slots pour lever A2.")
    return 0


def selfcheck() -> int:
    # la greffe copie le canal, et RIEN d'autre
    dst = {"model": {"encoder.net.0.weight": torch.randn(4, 8), "predictor.w": torch.randn(3)},
           "meta": {"obs_dim": 278, "proprio_dim": 133}}
    src = {"model": {"encoder.net.0.weight": torch.randn(4, 8),      # ne doit PAS écraser la cible
                     "slot_calib": torch.tensor([1.0, -1.0, -1.0]),
                     "slot_encoder.color_queries": torch.randn(3, 3),
                     "slot_encoder.sin": torch.randn(36)},
           "meta": {"slot_resources": 3, "name": "src_wm"}}
    enc_before = dst["model"]["encoder.net.0.weight"].clone()
    out, copied = graft(dst, src)
    assert len(copied) == 3, copied
    assert torch.equal(out["model"]["encoder.net.0.weight"], enc_before), \
        "la greffe ne doit JAMAIS toucher les poids appris de la cible"
    assert torch.equal(out["model"]["slot_calib"], src["model"]["slot_calib"])
    assert out["meta"]["slot_resources"] == 3 and out["meta"]["with_slot"] is True
    assert out["meta"]["proprio_dim"] == 133, "la meta de la CIBLE doit survivre à la greffe"
    print("  [ok] greffe : 3 tenseurs de canal copiés, poids appris de la cible INTACTS, meta cible préservée")

    # refuse d'écraser un canal existant (une greffe silencieuse serait indétectable ensuite)
    try:
        graft(out, src)
        raise AssertionError("aurait dû refuser d'écraser un canal existant")
    except SystemExit:
        print("  [ok] refuse d'écraser un canal-slot déjà présent")

    # le canal est bien de la GÉOMÉTRIE lisant la rétine : mêmes positions sur les deux WM
    from sylvan.models.command_wm import CommandWorldModel
    from sylvan.models.perception_head import RETINA_DIM
    torch.manual_seed(0)
    a_wm = CommandWorldModel(obs_dim=133 + RETINA_DIM + 1, proprio_dim=133, with_slot=True, slot_resources=2)
    b_wm = CommandWorldModel(obs_dim=132 + RETINA_DIM + 1, proprio_dim=132, with_slot=True, slot_resources=2)
    b_sd = dict(b_wm.state_dict())
    a_sd = dict(a_wm.state_dict())
    for k, v in slot_tensors(b_sd).items():
        a_sd[k] = v.clone()
    a_wm.load_state_dict(a_sd)
    retina = torch.rand(5, RETINA_DIM)
    obs_a = torch.cat([torch.randn(5, 133), retina, torch.rand(5, 1)], dim=1)
    obs_b = torch.cat([torch.randn(5, 132), retina, torch.rand(5, 1)], dim=1)
    assert torch.allclose(a_wm.encode_slot(obs_a), b_wm.encode_slot(obs_b), atol=1e-6), \
        "le slot doit dépendre de la RÉTINE seule, pas de la proprio ni des poids du WM"
    print("  [ok] slot greffé sur un WM à proprio 133 = MÊMES positions que la source à 132 "
          "(il ne lit que la rétine)")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
