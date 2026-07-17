"""G0 (GRATUIT, décisif) du chantier CANAL OBSTACLE / affordance physique.

Doc : docs/design_obstacle_affordance.md §Gates G0. Zéro run, zéro Godot, zéro entraînement — rejeu
d'un corpus PROPRE existant + injection SYNTHÉTIQUE d'un obstacle DEVANT dans la rétine.

Question à TRANCHER (choisit la voie de build) : le WM GELÉ conditionne-t-il son DÉPLACEMENT prédit
sur la PERCEPTION (la rétine), ou seulement sur la commande (vx, ω) ? La tête déplacement =
MetricsPredictionHead(cat[latent(128), commande(2)]) → la rétine ne peut atteindre le déplacement que
VIA le latent (encodeur→RSSM), et sa CIBLE d'entraînement est de la cinématique corporelle pure
(monde sans obstacle) → couplage rétine→déplacement probablement FAIBLE. On le MESURE, plus deux
bornes de représentabilité (le latent qui nourrit la tête déplacement bouge-t-il ? le slot requêté
bouge-t-il ?), pour LOCALISER le coût d'une éventuelle voie A.

Trois issues pré-enregistrées (docs/design_obstacle_affordance.md §Verdict G0) :
  - Δ_disp significatif                → voie A quasi-gratuite (le WM répond déjà) ;
  - Δ_disp ≈ 0 mais Δ_latent/Δ_slot fort → voie A possible mais COÛTEUSE ; voie B (prédicteur séparé,
    jumeau lunette danger) lit le même latent bien moins cher → RECOMMANDÉE ;
  - Δ_disp ≈ 0 ET Δ_latent ≈ 0         → voie A exige un retrain encodeur → voie B tranchée.

Contrôles pré-enregistrés : obstacle DEVANT vs CÔTÉ vs DERRIÈRE (réponse avant doit dominer) ;
placebo LOINTAIN (même couleur à ~10 m → Δ≈0 si la réponse est sensible à la PROXIMITÉ) ; null
déterministe (obs identique → Δ=0 exact, sanity).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_obstacle_g0.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "python")

from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE  # noqa: E402
from sylvan.models.slot_head import DEPTH_OFFSET, NRAY, RANGE  # noqa: E402

WM_CKPT = "data/checkpoints/wm_objcentric_kin_typed/wm_best.pt"
CLEAN_RUNS = [
    "data/replay_buffer/critic_kin_typcorp",
    "data/replay_buffer/critic_kin_g24as1", "data/replay_buffer/critic_kin_g24as2",
    "data/replay_buffer/critic_kin_g24bs1", "data/replay_buffer/critic_kin_g24bs2",
    "data/replay_buffer/critic_kin_pure1", "data/replay_buffer/critic_kin_pure2",
]
N_FRAMES = 256                    # obs réelles échantillonnées (front dégagé)
VX = 0.65                         # commande avant fixe (régime propre du corps)
HORIZON = 8                       # court : l'injection n'affecte que t0, le reste free-run
D_NEAR = 0.15                     # depth normalisé → ~1.85 m devant (obstacle proche)
D_FAR = 0.95                      # depth normalisé → ~9.85 m (placebo lointain, même couleur)
FRONT_RAYS = [NRAY - 1, 0, 1]     # -10°, 0°, +10° = un « mur » devant
SIDE_RAYS = [NRAY // 4 - 1, NRAY // 4, NRAY // 4 + 1]           # ~90° (côté)
BEHIND_RAYS = [NRAY // 2 - 1, NRAY // 2, NRAY // 2 + 1]         # ~180° (derrière)
GRAY = (0.5, 0.5, 0.5)            # obstacle NEUTRE (pas un drive) — le vrai cas
GREEN = (0.0, 1.0, 0.0)          # couleur REQUÊTÉE (hazard) — ancre de représentabilité du readout
PROPRIO_DIM = 132
RETINA_OFF = PROPRIO_DIM          # rétine = obs[132:276]


def _open(run: Path):
    p = run / "ep_0000.jsonl.gz"
    if p.exists():
        return gzip.open(p, "rt", errors="ignore")
    return open(run / "ep_0000.jsonl", errors="ignore")


def load_clean_front_obs(n: int) -> torch.Tensor:
    """Obs réelles [n, 277] dont le RAYON AVANT (ray 0) est un MISS → injection = intervention propre."""
    out = []
    for run in CLEAN_RUNS:
        rp = Path(run)
        if not (rp / "ep_0000.jsonl.gz").exists() and not (rp / "ep_0000.jsonl").exists():
            continue
        for line in _open(rp):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            w = r.get("wm", {})
            ret = w.get("retina0")
            if ret is None or len(ret) != NRAY * 4:
                continue
            if ret[0] < 0.999:                          # front DÉJÀ occupé → on saute (veut un front clair)
                continue
            obs = list(r["obs"]["proprio"]) + list(ret) + [float(r["obs"]["energy"]) / 100.0]
            if len(obs) != PROPRIO_DIM + NRAY * 4 + 1:
                continue
            out.append(obs)
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return torch.tensor(out, dtype=torch.float32)


def inject(obs: torch.Tensor, rays: list[int], depth: float, rgb: tuple[float, float, float]) -> torch.Tensor:
    """Clone de obs [B,277] avec un obstacle (depth, rgb) posé sur `rays`."""
    o = obs.clone()
    for k in rays:
        b = RETINA_OFF + 4 * k
        o[:, b] = depth
        o[:, b + 1] = rgb[0]
        o[:, b + 2] = rgb[1]
        o[:, b + 3] = rgb[2]
    return o


def load_wm() -> CommandWorldModel:
    pl = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    m = pl["meta"]
    wm = CommandWorldModel(
        obs_dim=m["obs_dim"], proprio_dim=m["proprio_dim"],
        predictor_arch=m.get("predictor_arch", "shallow"),
        with_slot=m["with_slot"], slot_resources=m["slot_resources"],
    )
    wm.load_state_dict(pl["model"])
    wm.eval()
    return wm, m


@torch.no_grad()
def rollout(wm: CommandWorldModel, obs: torch.Tensor):
    b = obs.shape[0]
    cmds = torch.zeros(b, HORIZON, 2)
    cmds[..., 0] = VX
    out = wm.rollout_open_loop(obs, cmds)
    disp = out["predicted_displacement"] / DISPLACEMENT_SCALE          # [B,T,3] (d_fwd,d_lat,d_yaw) réels
    d_fwd0 = disp[:, 0, 0]                                             # déplacement AVANT pas-0 (mètres)
    path = disp[..., :2].norm(dim=-1).sum(dim=1)                       # longueur de chemin cumulée [B]
    lat0 = out["predicted_latents"][:, 0]                             # [B,128] latent nourrissant la tête déplacement
    slots0 = out["slots"][:, 0] if "slots" in out else out["slot"][:, 0].unsqueeze(1)  # [B,R,2] readout requêté t0
    return d_fwd0, path, lat0, slots0


def main() -> None:
    torch.manual_seed(0)
    wm, m = load_wm()
    obs = load_clean_front_obs(N_FRAMES)
    print(f"[g0-obs] WM={WM_CKPT} obs_dim={m['obs_dim']} slots={m['slot_resources']} "
          f"(idx food/water/hazard={m.get('food_idx')}/{m.get('water_idx')}/{m.get('hazard_idx')})")
    print(f"[g0-obs] {obs.shape[0]} frames à front dégagé ; commande avant vx={VX}, horizon={HORIZON}")
    if obs.shape[0] < 16:
        print("[g0-obs] ❌ trop peu de frames — corpus absent ?")
        return

    d0_clean, path_clean, lat_clean, slot_clean = rollout(wm, obs)

    def probe(name: str, rays, depth, rgb):
        o = inject(obs, rays, depth, rgb)
        d0, path, lat, slot = rollout(wm, o)
        ratio = float((d0 / d0_clean.clamp(min=1e-9)).median())        # 1=ignore, <1=ralentit, ~0=bloque
        d_disp_mm = float((d0_clean - d0).abs().median()) * 1000.0     # |Δ déplacement avant| pas-0 (mm)
        cos_lat = float(torch.cosine_similarity(lat, lat_clean, dim=-1).median())
        d_lat = 1.0 - cos_lat                                          # 0=latent inchangé ; >0 = obstacle atteint le latent
        d_slot = float((slot - slot_clean).norm(dim=-1).max(dim=-1).values.median())  # bouge du readout requêté (m)
        d_path_mm = float((path_clean - path).abs().median()) * 1000.0
        print(f"  {name:22s} d_fwd0 {float(d0.median())*1000:6.2f}mm  ratio {ratio:5.2f}  "
              f"|Δdisp| {d_disp_mm:6.2f}mm  Δlatent(1-cos) {d_lat:7.4f}  Δslot {d_slot:5.2f}m  "
              f"|Δpath8| {d_path_mm:6.2f}mm")
        return dict(ratio=ratio, d_disp=d_disp_mm, d_lat=d_lat, d_slot=d_slot)

    print(f"[g0-obs] baseline : d_fwd0 clean = {float(d0_clean.median())*1000:.2f} mm "
          f"(path/{HORIZON} = {float(path_clean.median())*1000:.1f} mm)")
    print("\n[g0-obs] === SONDES (obstacle injecté) ===")
    null = probe("NULL (obs identique)", [], D_NEAR, GRAY)             # sanity : doit être ~0 partout
    front = probe("DEVANT proche (gris)", FRONT_RAYS, D_NEAR, GRAY)
    far = probe("DEVANT LOINTAIN (gris)", FRONT_RAYS, D_FAR, GRAY)     # placebo proximité
    side = probe("CÔTÉ proche (gris)", SIDE_RAYS, D_NEAR, GRAY)
    behind = probe("DERRIÈRE proche (gris)", BEHIND_RAYS, D_NEAR, GRAY)
    green = probe("DEVANT proche (VERT)", FRONT_RAYS, D_NEAR, GREEN)   # couleur requêtée → ancre readout

    # --- VERDICT (seuils pré-enregistrés, docs/design_obstacle_affordance.md §Verdict G0) ---
    # « Δ_disp significatif » = le déplacement avant CHUTE (ratio < 0.7 = >30% de frein) ET domine le côté.
    # « représentable » = le latent bouge nettement (Δlatent > 10× le null) OU le slot requêté répond.
    resp_disp = front["ratio"] < 0.70 and front["d_disp"] > side["d_disp"] and front["d_disp"] > far["d_disp"]
    lat_floor = max(null["d_lat"], 1e-6)
    representable = (front["d_lat"] > 10 * lat_floor) or (green["d_slot"] > 0.2)
    print("\n[g0-obs] === VERDICT G0 (sanity null : ratio≈1, Δ≈0) ===")
    print(f"[g0-obs] sanity null : ratio {null['ratio']:.3f} Δdisp {null['d_disp']:.3f}mm Δlatent {null['d_lat']:.5f}"
          f" → {'✅ déterministe' if null['d_disp'] < 1e-3 and null['d_lat'] < 1e-5 else '⚠️ non-nul'}")
    print(f"[g0-obs] contrôle spatial : Δdisp DEVANT {front['d_disp']:.2f}mm vs CÔTÉ {side['d_disp']:.2f}mm "
          f"vs DERRIÈRE {behind['d_disp']:.2f}mm vs LOINTAIN {far['d_disp']:.2f}mm")
    print(f"[g0-obs] représentabilité : Δlatent DEVANT {front['d_lat']:.4f} (null {null['d_lat']:.5f}) ; "
          f"Δslot VERT-devant {green['d_slot']:.2f} m")
    if resp_disp:
        print("[g0-obs] ➜ VOIE A QUASI-GRATUITE : le WM FREINE déjà face à l'obstacle "
              f"(ratio {front['ratio']:.2f} < 0.70) → fine-tune léger de la tête déplacement.")
    elif representable:
        print("[g0-obs] ➜ VOIE B RECOMMANDÉE : le déplacement IGNORE l'obstacle "
              f"(ratio {front['ratio']:.2f} ≈ 1) MAIS il ATTEINT le latent "
              f"(Δlatent {front['d_lat']:.4f} ; Δslot vert {green['d_slot']:.2f} m) → voie A = re-collecte+"
              "fine-tune COÛTEUX ; un prédicteur séparé lit le même latent/rétine bien moins cher (jumeau danger).")
    else:
        print("[g0-obs] ➜ VOIE B TRANCHÉE : l'obstacle N'ATTEINT MÊME PAS le latent "
              f"(Δlatent {front['d_lat']:.4f} ≈ null) → voie A exigerait un retrain ENCODEUR complet.")


if __name__ == "__main__":
    main()
