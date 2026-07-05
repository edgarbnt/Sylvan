"""diag_survcost_sanity — sanité GRATUITE du coût survie refill-aware du planner (gate B0 → code).

Vérifie `_survival_extension` (phase 2 analytique) sur des scénarios synthétiques dont l'issue est
connue, puis (--smoke) un appel `plan()` de bout en bout avec le VRAI WM promu (wm_objcentric_s1)
en mode survival vs designed (shapes/clé de retour, aucun crash, commandes dans la grille).

Lancer :  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_survcost_sanity.py [--smoke]
"""

from __future__ import annotations

import argparse
import math
import os

import torch

from sylvan.control.planning.command_planner import _survival_extension

# Régime « éco de vie » (baseline_multidrive_slot.sh) : drain 0.05/pas /100, refill 40/100.
DRAIN, RESTORE, SPD, CAP, MARGIN_W = 0.0005, 0.4, 0.02, 3000.0, 200.0


def ext(df, dw, e, t, dist_fw=5.0, steps_p1=None, alive=None, turn_f=None, turn_w=None):
    n = len(df)
    z = lambda v: torch.tensor(v, dtype=torch.float32)
    sf, sw = _survival_extension(
        z(df), z(dw), z(e), z(t),
        torch.ones(n) if alive is None else z(alive),
        torch.zeros(n) if steps_p1 is None else z(steps_p1),
        dist_fw, DRAIN, RESTORE, SPD, CAP, MARGIN_W,
        turn_f=None if turn_f is None else z(turn_f),
        turn_w=None if turn_w is None else z(turn_w),
    )
    return torch.maximum(sf, sw)   # sémantique historique des scénarios (le choix d'ordre = caller)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="appel plan() bout-en-bout avec le WM promu")
    args = ap.parse_args()

    # A. ARBITRAGE : énergie basse (0.10 → mort dans 200 pas) — le candidat qui finit PRÈS DE LA
    #    BOUFFE doit battre celui qui finit près de l'eau (l'aller-vers-l'eau-d'abord tue).
    s = ext(df=[0.5, 6.0], dw=[6.0, 0.5], e=[0.10, 0.10], t=[0.90, 0.90])
    assert s[0] > s[1] + 500, f"A: près-bouffe doit dominer nettement, {s.tolist()}"
    # A' symétrique : soif basse → près-de-l'eau domine.
    s = ext(df=[0.5, 6.0], dw=[6.0, 0.5], e=[0.90, 0.90], t=[0.10, 0.10])
    assert s[1] > s[0] + 500, f"A': près-eau doit dominer, {s.tolist()}"

    # B. COMMITTMENT (tie-break) : tout le monde survit au cap → la marge classe le candidat qui
    #    ARRIVE PLUS TÔT (plus près de la ressource urgente) devant — gradient lisse, pas de knife-edge.
    s = ext(df=[1.0, 2.0, 4.0], dw=[4.0, 4.0, 4.0], e=[0.5, 0.5, 0.5], t=[0.9, 0.9, 0.9], dist_fw=3.0)
    assert s[0] > s[1] > s[2], f"B: la marge doit décroître avec la distance, {s.tolist()}"
    assert float(s.min()) >= CAP, f"B: tous survivent au cap, {s.tolist()}"

    # C. MYOPIE PUNIE : ressource déplétée inatteignable → le score = pas-vécus < cap (mort simulée).
    s = ext(df=[6.0], dw=[0.5], e=[0.05], t=[0.90])
    assert float(s[0]) < 300, f"C: mort attendue ~100 pas, {s.tolist()}"

    # D. Mort en phase 1 (alive=0) : le temps reste figé aux pas déjà vécus.
    s = ext(df=[1.0], dw=[1.0], e=[0.5], t=[0.5], steps_p1=[42.0], alive=[0.0])
    assert abs(float(s[0]) - 42.0) < 1e-4, f"D: temps figé attendu 42, {s.tolist()}"

    # E. TEMPS DE VIRAGE (fix post-KILL) : mêmes distances, mais le candidat 2 finit DOS à la bouffe
    #    (π/0.015 ≈ 209 pas de virage) → celui qui s'est déjà tourné doit gagner.
    s = ext(df=[2.0, 2.0], dw=[6.0, 6.0], e=[0.30, 0.30], t=[0.9, 0.9],
            turn_f=[0.0, 209.0], turn_w=[0.0, 0.0])
    assert s[0] > s[1], f"E: le candidat déjà tourné doit gagner, {s.tolist()}"

    # F. ANTI-PLAT (fix post-KILL, std=0.000 à distance) : drive URGENT (e=0.25) + bouffe LOIN :
    #    l'ordre eau-d'abord meurt (9 m entre ressources) → le max = bouffe-d'abord, dont la marge de
    #    1ʳᵉ arrivée doit maintenant DIFFÉRENCIER deux fins d'arc distantes de 1 m (plus d'absorption).
    s = ext(df=[4.5, 5.5], dw=[4.0, 4.0], e=[0.25, 0.25], t=[0.85, 0.85], dist_fw=9.0)
    assert float((s[0] - s[1]).abs()) > 1.0, f"F: le score ne doit plus être plat à distance, {s.tolist()}"
    assert s[0] > s[1], f"F: la fin d'arc plus proche de la bouffe urgente doit gagner, {s.tolist()}"

    print("[sanity] OK — arbitrage (A/A'), committment-marge (B), myopie punie (C), mort phase-1 (D), "
          "virage (E), anti-plat (F)")

    if args.smoke:
        os.environ["SYLVAN_PLANNER_DRAIN"] = str(DRAIN)
        os.environ["SYLVAN_PLANNER_RESTORE"] = str(RESTORE)
        from sylvan.control.planning.command_planner import CommandPlanner
        from sylvan.models.command_wm import CommandWorldModel
        payload = torch.load("data/checkpoints/wm_objcentric_s1/wm_best.pt",
                             map_location="cpu", weights_only=False)  # same load path as serve_planner_command
        meta = payload["meta"]
        wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                               predictor_arch=meta.get("predictor_arch", "shallow"),
                               with_slot=meta.get("with_slot", False),
                               slot_resources=meta.get("slot_resources", 1))
        wm.load_state_dict(payload["model"])
        wm.eval()
        torch.manual_seed(0)
        obs = torch.zeros(meta["obs_dim"])
        obs[-1] = 0.4                       # énergie normalisée
        water = [0.0] * 36
        water[9] = 0.7                      # eau visible à ~90° droite
        for mode in ("designed", "survival"):
            os.environ["SYLVAN_PLANNER_COST"] = mode
            planner = CommandPlanner(wm)
            out = planner.plan(obs, radar=[0.0] * 12, water_radar=water, energy=0.4, thirst=0.2)
            vx, om = out["command"]
            assert 0.5 <= vx <= 0.8 and -0.65 <= om <= 0.65, out
            print(f"[smoke] {mode:9s} -> reason={out['reason']} cmd=({vx:.2f},{om:+.2f})"
                  + (f" pred_steps_alive={out.get('pred_steps_alive'):.0f}" if mode == "survival" else ""))
        print("[smoke] OK — plan() designed et survival répondent avec le WM promu")


if __name__ == "__main__":
    main()
