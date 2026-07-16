"""BUILD (pas un entraînement) — remplace les requêtes-couleur MAIN du WM vivant par les requêtes
APPRISES du soulagement vécu (volet P6, docs/design_purete_hjepa.md §P6). Gabarit build_hazard_slot :
le readout du slot est géométrique zéro-paramètre → changer les 9 nombres du buffer
`slot_encoder.color_queries` est le SEUL geste ; WM GELÉ, tous les autres poids intacts.

Refuse un ckpt de requêtes dont les gates offline (G-q + G-slot-parité) n'ont pas passé.

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.build_learned_queries \
            [--queries data/checkpoints/drive_queries/queries_best.pt] \
            [--src data/checkpoints/wm_objcentric_kin_haz/wm_best.pt] \
            [--out data/checkpoints/wm_objcentric_kin_lq]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.train_drive_queries import DRIVES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/checkpoints/drive_queries/queries_best.pt")
    ap.add_argument("--src", default="data/checkpoints/wm_objcentric_kin_haz/wm_best.pt")
    ap.add_argument("--out", default="data/checkpoints/wm_objcentric_kin_lq")
    args = ap.parse_args()

    qk = torch.load(args.queries, map_location="cpu", weights_only=True)
    assert qk.get("gates_pass"), "gates offline P6 non passés — build refusé (§P6)"
    Q = qk["queries"]
    assert tuple(Q.shape) == (3, 3) and list(qk["drives"]) == list(DRIVES), (Q.shape, qk["drives"])
    Q = Q / Q.norm(dim=-1, keepdim=True)

    payload = torch.load(args.src, map_location="cpu", weights_only=False)
    meta = dict(payload["meta"])
    # l'ordre des slots du WM vivant (food=0, water=1, danger=2) DOIT matcher l'ordre des drives
    assert (meta.get("food_idx"), meta.get("water_idx"), meta.get("hazard_idx")) == (0, 1, 2), meta
    state = dict(payload["model"])
    old = state["slot_encoder.color_queries"]
    assert old.shape == Q.shape, (old.shape, Q.shape)
    for k, d in enumerate(DRIVES):
        cos = float(Q[k] @ old[k])
        print(f"[build-lq] {d:6s} : main=[{old[k][0]:.2f} {old[k][1]:.2f} {old[k][2]:.2f}] → "
              f"apprise=[{Q[k][0]:+.3f} {Q[k][1]:+.3f} {Q[k][2]:+.3f}] (cos={cos:.4f})")
    state["slot_encoder.color_queries"] = Q

    meta["queries"] = "learned_from_consequence_P6"
    meta["queries_ckpt"] = str(args.queries)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "meta": meta}, out / "wm_best.pt")
    print(f"[build-lq] WM à requêtes APPRISES sauvé → {out / 'wm_best.pt'} (zéro retrain ; "
          f"seuls les 9 nombres du buffer ont changé)")


if __name__ == "__main__":
    main()
