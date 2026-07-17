"""Diag GRATUIT — le WM GELÉ encaisse-t-il des apparences VARIÉES ? (docs/design_monde_incremental.md)

Gate le bump « apparences variées » (le seul ingrédient de monde qui touche le WM gelé). Question
falsifiable : perturber les rétines RÉELLES du corpus (jitter de teinte intra-type, bruit de texture
par-rayon, désaturation) à magnitude graduée, et mesurer si le LATENT du WM bouge — comparé à sa
dérive NATURELLE tick-à-tick (l'étalon honnête : le latent bouge déjà à chaque pas de vie normale).

- **Sujet = le LATENT** (`wm.encoder(obs)`) : c'est le substrat sur lequel tout repose. S'il est
  ~invariant à l'apparence → le bump est CHEAP (généralise, retrain léger de la requête seule).
  S'il explose à faible perturbation → recollecte du WM = EXPENSIVE.
- **Secondaire = le SLOT** (position + visibilité) : il utilise les requêtes-couleur codées-main
  (seuil 0.55) → attendu plus fragile ; c'est la pièce qu'on remplace de toute façon. Reporté à
  part pour ne PAS confondre « requête fragile » (connu) et « substrat fragile » (la vraie question).

⚠️ HONNÊTETÉ : une perturbation RGB synthétique ≠ un rendu texturé réel. Ce diag BORNE le risque
gratuitement (courbe de dégradation du modèle gelé) ; il ne remplace pas un check open-loop sur le
vrai bump. Mais si le latent est robuste à un jitter substantiel, le bump est probablement cheap ;
s'il est fragile, on l'a appris pour zéro coût.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_wm_appearance_robustness.py [--selfcheck]
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics as st
from pathlib import Path

import torch

from scripts.appearance_synth import (DESAT, HUE_DEG, MODERATE, NRAY, TEX_SIGMA, hue_matrix,
                                      perturb, touch_mask)
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.perception_head import RETINA_DIM

WM_CKPT = "data/checkpoints/wm_objcentric_kin_haz/wm_best.pt"
RUNS = ["data/replay_buffer/critic_kin_g24as1", "data/replay_buffer/critic_kin_judge1",
        "data/replay_buffer/critic_kin_pure2"]


def load_wm() -> CommandWorldModel:
    payload = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                           predictor_arch=meta.get("predictor_arch", "shallow"),
                           with_slot=meta.get("with_slot", False),
                           slot_resources=meta.get("slot_resources", 1))
    wm.load_state_dict(payload["model"])
    wm.eval()
    return wm, meta


def _open(run: Path):
    p = run / "ep_0000.jsonl.gz"
    return gzip.open(p, "rt", errors="ignore") if p.exists() else open(run / "ep_0000.jsonl",
                                                                       errors="ignore")


def load_obs(wm_meta: dict, n_max: int = 3000) -> torch.Tensor:
    """Construit l'obs WM [N, obs_dim] = proprio(132) + retina(144) + energie(1), depuis le BC."""
    pdim, odim = wm_meta["proprio_dim"], wm_meta["obs_dim"]
    rows = []
    per = max(1, n_max // len(RUNS))
    for run in RUNS:
        got = 0
        for line in _open(Path(run)):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pro, ret = r["obs"]["proprio"], r["wm"]["retina0"]
            if len(pro) != pdim or len(ret) != RETINA_DIM:
                continue
            rows.append(pro + ret + [float(r["obs"]["energy"]) / 100.0])
            got += 1
            if got >= per:
                break
    x = torch.tensor(rows, dtype=torch.float32)
    assert x.shape[1] == odim, (x.shape, odim)
    return x


@torch.no_grad()
def rel_drift(z: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Dérive relative L2 par échantillon : ‖z2−z‖ / ‖z‖ (le latent bouge de combien, en %)."""
    return (z2 - z).norm(dim=-1) / (z.norm(dim=-1) + 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    wm, meta = load_wm()
    pdim = meta["proprio_dim"]
    obs = load_obs(meta, args.n)
    ret_lo, ret_hi = pdim, pdim + RETINA_DIM
    gen = torch.Generator().manual_seed(0)
    print(f"[wm-robust] WM={Path(WM_CKPT).parent.name} obs_dim={meta['obs_dim']} "
          f"latent={wm.encoder(obs[:1]).shape[-1]} | {len(obs)} ticks")

    with torch.no_grad():
        z0 = wm.encoder(obs)
    # ÉTALON : dérive latente NATURELLE entre ticks consécutifs (une vie normale bouge déjà le latent)
    d_nat = float(rel_drift(z0[:-1], z0[1:]).median())
    # SLOT de référence (position + visibilité) sur les rétines originales
    ret0 = obs[:, ret_lo:ret_hi]
    with torch.no_grad():
        pos0 = wm.slot_encoder.positions(ret0)              # [N, R, 2]
        vis0 = wm.slot_encoder.visibility(ret0) > 1e-6      # [N, R]

    print(f"\n[wm-robust] ÉTALON dérive latente NATURELLE tick-à-tick (méd) = {d_nat:.4f}")
    print(f"[wm-robust] {'perturbation':<22}{'lat.drift/nat':>14}{'slot Δpos(m)':>14}{'vis gardée':>12}")
    print(f"[wm-robust] {'-' * 62}")
    results: dict[tuple[str, int], float] = {}
    for kind, mags in (("hue", HUE_DEG), ("texture", TEX_SIGMA), ("desat", DESAT)):
        for j, mag in enumerate(mags):
            pobs = obs.clone()
            pret = perturb(obs[:, ret_lo:ret_hi], kind, mag, gen)
            pobs[:, ret_lo:ret_hi] = pret
            with torch.no_grad():
                z = wm.encoder(pobs)
                pos = wm.slot_encoder.positions(pret)
                vis = wm.slot_encoder.visibility(pret) > 1e-6
            d = float(rel_drift(z0, z).median())
            ratio = d / (d_nat + 1e-9)
            both = vis0 & vis
            dpos = float((pos - pos0)[both].norm(dim=-1).median()) if bool(both.any()) else float("nan")
            visk = float((vis[vis0]).float().mean()) if bool(vis0.any()) else float("nan")
            results[(kind, j)] = ratio
            tag = "  ← modéré" if j == MODERATE else ""
            print(f"[wm-robust] {kind}={mag:<16.2f}{ratio:>13.2f}x{dpos:>14.3f}{100 * visk:>11.0f}%{tag}")

    print(f"\n[wm-robust] === VERDICT (BUT = robustesse du LATENT, pas le slot) ===")
    mod = [results[(k, MODERATE)] for k in ("hue", "texture", "desat")]
    worst = max(mod)
    print(f"[wm-robust] au niveau MODÉRÉ (hue 20° / σ 0.05 / desat 0.4), dérive latente = "
          f"{', '.join(f'{r:.1f}x' for r in mod)} × la dérive naturelle (pire = {worst:.1f}x)")
    if worst <= 1.0:
        print("[wm-robust] ✅ CHEAP : à perturbation réaliste, le latent bouge MOINS qu'un pas de vie "
              "normal → le WM gelé est ~invariant à l'apparence. Le bump = retrain LÉGER de la "
              "requête seule (WM intouché). Diag open-loop sur le vrai bump reste dû (caveat).")
    elif worst <= 3.0:
        print("[wm-robust] ⚠️ PARTIEL : le latent bouge de 1-3× la dérive naturelle → décision owner. "
              "Piste : diag open-loop réel avant de trancher recollecte vs retrain requête.")
    else:
        print("[wm-robust] ❌ EXPENSIVE : le latent explose (>3× naturel) à perturbation modérée → le "
              "substrat gelé NE généralise PAS aux apparences variées → recollecte + ré-entraînement "
              "WM requis avant de rouvrir la reconnaissance des types. Coût confirmé pour ZÉRO run.")
    print("[wm-robust] (slot Δpos/vis = fragilité de la REQUÊTE codée-main, la pièce déjà à remplacer "
          "— informe la difficulté du retrain requête, pas le verdict substrat.)")


def selfcheck() -> None:
    from scripts.appearance_synth import selfcheck as _synth_selfcheck
    assert NRAY == 36 and RETINA_DIM == 144
    _synth_selfcheck()                                                 # invariants du perturbateur partagé
    assert callable(hue_matrix) and callable(touch_mask)               # importés OK
    print("[selfcheck] OK — perturbateur partagé (appearance_synth) + dims WM 144/36")


if __name__ == "__main__":
    main()
