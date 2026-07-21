"""TEST GRATUIT — les candidats du planner divergent-ils assez pour qu'une DÉCISION soit possible ?

LA QUESTION. L'audit de péremption a conclu que le mur n'était pas les constantes mais l'HORIZON
D'IMAGINATION : le WM déroule 80 pas = 0,8 m dans un monde où les ressources sont à 2-8 m, donc les
33+ candidats finiraient quasi ex-æquo (le code note « marge relative 0,003-0,005 »,
command_planner.py:745-760, comme cause de l'échec du critique appris). Avant de payer un horizon
plus long ou une abstraction temporelle, il faut VÉRIFIER cette conclusion — elle m'arrange, donc
elle est suspecte (§2 : « une conclusion qui arrange = suspecte »).

CE QUE MESURE LA SONDE, et pourquoi c'est gratuit et décisif. Le corps cinématique obéit EXACTEMENT
à (vx, ω) — constantes MESURÉES en Phase 1 : 0,0100 m/pas à vx=0,75 et 0,0150 rad/pas à |ω|=0,6.
On peut donc dérouler chaque candidat ANALYTIQUEMENT, sans WM ni Godot. C'est une BORNE SUPÉRIEURE :
le WM ne peut pas produire plus de divergence que la géométrie n'en autorise.

On mesure DEUX choses, et c'est leur écart qui tranche :
  (a) la divergence GÉOMÉTRIQUE — de combien les candidats séparent la distance finale à la cible.
      C'est le signal DISPONIBLE.
  (b) la divergence du SCORE de la queue survie — ce que le planner utilise VRAIMENT.
      C'est le signal UTILISÉ.

  - (a) petit  => l'horizon est bien le mur ; allonger l'imagination est la vraie voie.
  - (a) grand ET (b) petit => l'information EST là et c'est le COÛT qui la détruit. L'horizon
    ne serait alors PAS le mur, et ma conclusion d'audit serait à réviser.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_candidate_divergence.py \
        [--runs data/replay_buffer/X ...] [--selfcheck]
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from guards import _ticks  # noqa: E402
from sylvan.control.planning.command_planner import _survival_extension  # noqa: E402

# Cinématique MESURÉE (Phase 1) : m/pas = vx x K_V, rad/pas = omega x K_W.
K_V, K_W = 0.0100 / 0.75, 0.0150 / 0.6
# Grilles EXACTES du planner (command_planner.py:52-61), pivot OFF (défaut).
VX, OM = (0.55, 0.65, 0.75), (-0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6)
SEG_VX, SEG_OM, SPLITS = (0.6, 0.75), (-0.6, -0.3, 0.0, 0.3, 0.6), (40,)
# Coût servi par les harnais vivants.
DRAIN, DRAIN_T, RESTORE, CAP, MARGIN_W, TURN_RATE = 0.0005, 0.00035, 0.4, 3000.0, 200.0, 0.015
SPD_SERVED = 0.02          # nominal_speed RÉELLEMENT servi (périmé mais load-bearing, cf. KILL)


def candidates(h: int) -> torch.Tensor:
    """[N, h, 2] — reproduit _build_candidates (constant + 2-segments, pivot OFF)."""
    seqs = [[[vx, om]] * h for vx in VX for om in OM]
    seen = set()
    for split in SPLITS:
        if not (0 < split < h):
            continue
        for vx1 in SEG_VX:
            for om1 in SEG_OM:
                for vx2 in SEG_VX:
                    for om2 in SEG_OM:
                        k = (split, vx1, om1, vx2, om2)
                        if (vx1, om1) == (vx2, om2) or k in seen:
                            continue
                        seen.add(k)
                        seqs.append([[vx1, om1]] * split + [[vx2, om2]] * (h - split))
    return torch.tensor(seqs, dtype=torch.float32)


def rollout(cands: torch.Tensor, tx: float, tz: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Déroule la cinématique EXACTE. Cible en ego (tx, tz) ; renvoie (distance finale, |bearing| final).

    Repère du planner : l'indice 1 est l'AVANT (cos_brg = slot[...,1]/dist, command_planner.py:555).
    """
    n, h, _ = cands.shape
    x = torch.zeros(n)
    z = torch.zeros(n)
    yaw = torch.zeros(n)
    for k in range(h):
        vx, om = cands[:, k, 0] * K_V, cands[:, k, 1] * K_W
        yaw = yaw + om
        x = x + vx * torch.sin(yaw)
        z = z + vx * torch.cos(yaw)
    dx, dz = tx - x, tz - z
    dist = torch.sqrt(dx ** 2 + dz ** 2)
    fwd = dx * torch.sin(yaw) + dz * torch.cos(yaw)
    lat = dx * torch.cos(yaw) - dz * torch.sin(yaw)
    return dist, torch.atan2(lat, fwd).abs()


def tail_scores(df: torch.Tensor, dw: torch.Tensor, bf: torch.Tensor, bw: torch.Tensor,
                e: float, t: float, dist_fw: float) -> torch.Tensor:
    """Le score que le planner utilise réellement (max des deux ordres, comme dans plan())."""
    s_f, s_w = _survival_extension(
        df, dw, torch.full_like(df, e), torch.full_like(df, t),
        torch.ones_like(df), torch.zeros_like(df),
        dist_fw, DRAIN, RESTORE, SPD_SERVED, CAP, MARGIN_W,
        turn_f=bf / TURN_RATE, turn_w=bw / TURN_RATE, gamma=0.0, drain_t=DRAIN_T)
    return torch.maximum(s_f, s_w)


def sample_states(runs: list[str], limit: int = 400) -> list[dict]:
    """États RÉELS (distances + bearings ego des deux ressources, niveaux) tirés des corpus."""
    out: list[dict] = []
    for run in runs:
        T = _ticks(run)
        for i, tk in enumerate(T):
            p = tk.get("plan")
            if not p or not p.get("food") or not p.get("water"):
                continue
            f, w = p["food"], p["water"]
            df, dw = math.hypot(*f), math.hypot(*w)
            if df <= 0 or dw <= 0:
                continue
            out.append({"f": f, "w": w, "df": df, "dw": dw,
                        "e": float(tk["obs"]["energy"]) / 100.0,
                        "t": float(tk["obs"]["thirst"]) / 100.0,
                        "dfw": math.hypot(f[0] - w[0], f[1] - w[1])})
            if len(out) >= limit:
                return out
    return out


def analyse(states: list[dict], h: int) -> dict:
    cands = candidates(h)
    geo_rel, sc_rel, agree = [], [], []
    for s in states:
        df, bf = rollout(cands, s["f"][0], s["f"][1])
        dw, bw = rollout(cands, s["w"][0], s["w"][1])
        sc = tail_scores(df, dw, bf, bw, s["e"], s["t"], s["dfw"])
        # (a) divergence GÉOMÉTRIQUE : étendue de la distance finale, en % de la distance initiale
        geo_rel.append(float(df.max() - df.min()) / s["df"])
        # (b) divergence du SCORE : étendue relative à |score| médian
        med = float(sc.median().abs()) + 1e-9
        sc_rel.append(float(sc.max() - sc.min()) / med)
        # le meilleur score est-il bien le candidat qui s'approche le plus ?
        agree.append(int(int(sc.argmax()) == int(df.argmin())))
    return {"h": h, "n_cand": cands.shape[0], "travel_m": h * 0.0100,
            "geo_rel": st.median(geo_rel), "sc_rel": st.median(sc_rel),
            "agree": 100.0 * sum(agree) / max(len(agree), 1)}


def _selfcheck() -> None:
    c = candidates(80)
    assert c.shape[0] == len(VX) * len(OM) + (len(SEG_VX) * len(SEG_OM)) ** 2 - len(SEG_VX) * len(SEG_OM)
    # aller tout droit doit couvrir h x 0.0100 m a vx=0.75
    straight = torch.tensor([[[0.75, 0.0]] * 100], dtype=torch.float32)
    d, _ = rollout(straight, 0.0, 10.0)
    assert abs((10.0 - float(d[0])) - 1.0) < 1e-3, f"trajet attendu 1.00 m, obtenu {10.0 - float(d[0]):.3f}"
    # un demi-tour doit S'ELOIGNER d'une cible devant
    back = torch.tensor([[[0.75, -0.6]] * 100 + [[0.75, 0.0]] * 0], dtype=torch.float32)
    db, _ = rollout(back, 0.0, 10.0)
    assert float(db[0]) > float(d[0]), "tourner devrait moins approcher qu'aller tout droit"
    print(f"selfcheck OK — {c.shape[0]} candidats, cinématique exacte (1.00 m en 100 pas à vx=0.75)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["data/replay_buffer/arbgrad_graded_s1_r40_fa0"])
    ap.add_argument("--horizons", nargs="*", type=int, default=[80, 120, 240, 480, 960])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
        return
    states = sample_states(a.runs, a.limit)
    if not states:
        raise SystemExit("aucun état exploitable (il faut des ticks avec plan.food ET plan.water)")
    print(f"{len(states)} états RÉELS | distance médiane à la bouffe "
          f"{st.median([s['df'] for s in states]):.2f} m\n")
    print(f"{'horizon':>8} {'trajet':>8} {'cand.':>6} | {'(a) divergence GÉO':>20} "
          f"| {'(b) divergence SCORE':>21} | {'argmax=le+proche':>17}")
    print("-" * 92)
    for h in a.horizons:
        r = analyse(states, h)
        print(f"{r['h']:8d} {r['travel_m']:7.2f}m {r['n_cand']:6d} | "
              f"{100 * r['geo_rel']:19.1f}% | {100 * r['sc_rel']:20.2f}% | {r['agree']:16.0f}%")
    print("\n(a) = étendue de la distance finale entre candidats, en % de la distance initiale."
          "\n(b) = étendue du score de la queue survie, en % du score médian."
          "\n(a) grand + (b) petit => l'info EST là et le COÛT la détruit (l'horizon n'est PAS le mur).")


if __name__ == "__main__":
    main()
