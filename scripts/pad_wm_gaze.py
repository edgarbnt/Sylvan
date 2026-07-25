"""PRÉPARE UN CHECKPOINT WM (proprio 132) POUR UN WARM-START À REGARD ACTIF (proprio 133).

POURQUOI CET OUTIL EXISTE. Le retrain du monde-forêt se fait avec le REGARD, donc une proprioception
de 133 dimensions et une observation WM de 278 (proprio ++ rétine(144) ++ énergie) au lieu de 277.
`load_state_dict` refuse un tel écart de forme — même en `strict=False`, une taille qui diffère lève.
Sans cet outil, le retrain partirait de ZÉRO et jetterait tout ce que le WM a déjà appris de la
rétine et de la dynamique.

🚨 POURQUOI UN ZERO-PAD NAÏF SERAIT FAUX. L'idiome maison (constants.py) est « la vision est ajoutée
EN DERNIER pour que les warm-starts complètent de zéros ». Ici ce n'est PAS le cas : l'angle de tête
est ajouté à la fin de la PROPRIOCEPTION, donc à l'indice 132 — c'est-à-dire AU MILIEU de
l'observation du WM, juste avant la rétine. Compléter à droite décalerait la rétine et l'énergie
d'un cran : chaque poids appris regarderait le mauvais canal. On INSÈRE donc à l'indice 132, en
poussant les 145 dimensions suivantes d'un rang, ce qui préserve l'alignement appris.

Le poids inséré est NUL : au premier pas, le modèle chargé se comporte EXACTEMENT comme avant sur les
dimensions qu'il connaissait, et le regard part sans influence — c'est l'entraînement qui décidera de
lui en donner une. Un warm-start ne doit rien changer avant la première mise à jour.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python scripts/pad_wm_gaze.py \
        data/checkpoints/wm_objcentric_kin/wm_best.pt /tmp/wm_kin_gaze_init.pt
    PYTHONPATH=python env_pytorch_3.12/bin/python scripts/pad_wm_gaze.py --selfcheck
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

GAZE_INDEX = 132          # l'angle de tête est appendé APRÈS les 132 dims historiques
RETINA_PLUS_ENERGY = 145  # ce qui suit dans l'observation WM et doit être DÉCALÉ, pas écrasé


def _insert_zero(t: torch.Tensor, dim: int, index: int) -> torch.Tensor:
    """Insère une tranche NULLE à `index` le long de `dim` (les suivantes sont décalées)."""
    shape = list(t.shape)
    shape[dim] = 1
    zero = torch.zeros(shape, dtype=t.dtype)
    return torch.cat([t.narrow(dim, 0, index), zero,
                      t.narrow(dim, index, t.shape[dim] - index)], dim=dim)


def pad_state_dict(sd: dict, obs_dim: int) -> tuple[dict, list[str]]:
    """Élargit de obs_dim à obs_dim+1 tout tenseur indexé par l'observation. Renvoie (sd, touchés)."""
    out, touched = {}, []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        dims = [d for d, n in enumerate(v.shape) if n == obs_dim]
        if not dims:
            out[k] = v
            continue
        w = v
        for d in dims:                      # un biais n'a qu'une dim ; un poids peut en avoir une seule aussi
            w = _insert_zero(w, d, GAZE_INDEX)
        out[k] = w
        touched.append(f"{k} {tuple(v.shape)} -> {tuple(w.shape)}")
    return out, touched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if not a.src or not a.dst:
        raise SystemExit("usage : pad_wm_gaze.py <src.pt> <dst.pt>")

    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    meta = dict(ck.get("meta", {}))
    obs_dim, proprio_dim = meta.get("obs_dim"), meta.get("proprio_dim")
    if proprio_dim != GAZE_INDEX:
        raise SystemExit(f"proprio_dim {proprio_dim} != {GAZE_INDEX} — ce checkpoint n'est pas un WM "
                         "d'avant le regard, rien à élargir")

    sd, touched = pad_state_dict(ck["model"], obs_dim)
    ck["model"] = sd
    meta["obs_dim"] = obs_dim + 1
    meta["proprio_dim"] = proprio_dim + 1
    meta["padded_for_gaze_from"] = str(a.src)      # traçabilité : d'où vient ce checkpoint
    ck["meta"] = meta

    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.dst)
    print(f"obs {obs_dim} -> {obs_dim + 1}, proprio {proprio_dim} -> {proprio_dim + 1} | "
          f"{len(touched)} tenseurs élargis :")
    for t in touched:
        print(f"    {t}")
    print(f"-> {a.dst}")
    return 0


def selfcheck() -> int:
    # une colonne NULLE est insérée au bon endroit, et rien d'autre ne bouge
    w = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5)
    got = _insert_zero(w, 1, 2)
    assert got.shape == (2, 6)
    assert torch.equal(got[:, :2], w[:, :2]), "les colonnes AVANT l'insertion doivent être intactes"
    assert torch.equal(got[:, 3:], w[:, 2:]), "les colonnes APRÈS doivent être DÉCALÉES, pas écrasées"
    assert float(got[:, 2].abs().sum()) == 0.0, "la colonne insérée doit être nulle"
    print("  [ok] insertion : avant intact, après décalé, colonne insérée nulle")

    # sur une forme réelle de WM : (256, 277) -> (256, 278), rétine+énergie décalées d'un cran
    enc = torch.randn(256, 277)
    pad = _insert_zero(enc, 1, GAZE_INDEX)
    assert pad.shape == (256, 278)
    assert torch.equal(pad[:, :GAZE_INDEX], enc[:, :GAZE_INDEX])
    assert torch.equal(pad[:, GAZE_INDEX + 1:], enc[:, GAZE_INDEX:])
    assert pad[:, GAZE_INDEX].abs().sum() == 0.0
    assert enc.shape[1] - GAZE_INDEX == RETINA_PLUS_ENERGY, "rétine+énergie = 145 dims après le regard"
    print(f"  [ok] encoder (256,277) -> (256,278) : les {RETINA_PLUS_ENERGY} dims de rétine+énergie "
          "sont DÉCALÉES, jamais écrasées")

    # un warm-start neutre : à entrée identique (gaze=0), la sortie est INCHANGÉE
    x = torch.randn(4, 277)
    x_gaze = torch.cat([x[:, :GAZE_INDEX], torch.zeros(4, 1), x[:, GAZE_INDEX:]], dim=1)
    assert torch.allclose(x @ enc.T, x_gaze @ pad.T, atol=1e-5)
    print("  [ok] à regard nul, le modèle élargi rend EXACTEMENT la même sortie qu'avant")

    # pad_state_dict ne touche QUE les tenseurs indexés par l'observation
    sd = {"encoder.net.0.weight": torch.randn(256, 277),
          "obs_head.network.2.weight": torch.randn(277, 128),
          "obs_head.network.2.bias": torch.randn(277),
          "predictor.net.0.weight": torch.randn(128, 130),      # ne doit PAS bouger
          "epoch": 7}
    out, touched = pad_state_dict(sd, 277)
    assert len(touched) == 3, touched
    assert out["predictor.net.0.weight"].shape == (128, 130), "un tenseur sans obs_dim doit être intact"
    assert out["encoder.net.0.weight"].shape == (256, 278)
    assert out["obs_head.network.2.weight"].shape == (278, 128)
    assert out["obs_head.network.2.bias"].shape == (278,)
    assert out["epoch"] == 7, "les entrées non-tenseurs doivent passer telles quelles"
    print(f"  [ok] pad_state_dict n'élargit que les {len(touched)} tenseurs indexés par l'observation")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
