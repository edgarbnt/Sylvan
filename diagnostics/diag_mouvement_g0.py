"""G0 GRATUIT — le mouvement propre des objets est-il seulement LISIBLE ?

Gates pré-inscrits dans `docs/design_mouvement_objets.md`. Zéro entraînement lourd, zéro Godot.

CONTEXTE. Le WM croit les objets immobiles (`transport_slot` = ego-motion seule). Sur 80 pas de
rêve, ça accumule ~1,8 m d'erreur pour une bouche de 1,0 m, et le ralenti qui en découle coûte
−64,5 points de capture. La voie NAÏVE est déjà réfutée : estimer la vitesse par différences finies
sur le slot servi donne 0,0751 m/pas, soit 3,3x PIRE que de supposer l'objet immobile (0,0230) —
le bruit du slot écrase le mouvement et la différenciation l'amplifie.

L'ASYMÉTRIE QU'ON EXPLOITE ICI. À UN pas, bruit et mouvement dérivent à la même vitesse
(autocorrélations 0,948 contre 0,936) — d'où l'échec naïf. Mais le MOUVEMENT S'ACCUMULE
linéairement alors que le BRUIT SATURE (il est borné par la dispersion de l'erreur). À un décalage
plus long, le rapport peut donc s'inverser. C'est exactement ce que G0-A mesure.

  G0-A  SNR(Δ) = médiane |déplacement VRAI sur Δ| ÷ médiane |changement d'ERREUR du slot sur Δ|
        PASS  : il existe un Δ avec SNR >= 2,0
        STOP  : SNR < 1,0 à TOUS les Δ ⇒ le slot servi ne peut pas porter le mouvement, et le
                chantier REDIRIGE vers la PRÉCISION DE LA PERCEPTION.

  G0-B  L'information est-elle dans le CAPTEUR ? Tête sur deux rétines séparées de Δ + ego-motion,
        étiquettes VÉRITÉ-TERRAIN en ORACLE D'ÉVAL (plafond), découpe PAR ÉPISODE.
        PASS  : erreur angulaire médiane <= 45° (hasard = 90°)
        STOP  : > 70°

⚠️ `food_rel0` est un oracle d'ÉVAL. Il sert à MESURER la faisabilité, jamais à entraîner le vivant.
⚠️ Découpe PAR ÉPISODE, jamais aléatoire (facteur 6 mesuré le 2026-08-02).

Usage :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        diagnostics/diag_mouvement_g0.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st

import torch

from sylvan.models.command_wm import CommandWorldModel

SWITCH_M = 0.5   # saut de position => changement de cible, on coupe la série
BAR_SNR = 2.0
STOP_SNR = 1.0
BAR_ANG = 45.0
STOP_ANG = 70.0


def to_world(rel: tuple[float, float], torso: list[float]) -> tuple[float, float]:
    c, s = math.cos(torso[2]), math.sin(torso[2])
    return (torso[0] + rel[1] * s + rel[0] * c, torso[1] + rel[1] * c - rel[0] * s)


def load(runs: list[str], wm: CommandWorldModel, fov: float) -> list[dict]:
    """Séries par épisode : position VRAIE, position LUE par le slot, torse, rétine."""
    eps = []
    for run in runs:
        for f in sorted(glob.glob(f"{run}/*.jsonl")):
            R, T, F = [], [], []
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                w, ob = r.get("wm"), r.get("obs")
                if not w:
                    continue
                ret = w.get("retina0")
                t = w.get("torso0") or (ob or {}).get("torso")
                fr = w.get("food_rel0") or [0.0, 0.0, 0.0]
                if not ret or len(ret) != 144 or not t:
                    continue
                bear = abs(math.degrees(math.atan2(fr[0], fr[1])))
                vis = fr[2] > 0.5 and bear <= fov / 2
                R.append(ret)
                T.append(t)
                F.append((fr[0], fr[1], vis))
            if len(R) < 80:
                continue
            X = torch.tensor(R)
            with torch.no_grad():
                P = wm.slot_encoder.positions(X)[:, 0, :]
            true_w = [to_world((F[i][0], F[i][1]), T[i]) for i in range(len(T))]
            slot_w = [to_world((float(P[i, 0]), float(P[i, 1])), T[i]) for i in range(len(T))]
            eps.append({"true": true_w, "slot": slot_w, "vis": [f[2] for f in F],
                        "torso": T, "retina": R})
    return eps


def g0a(eps: list[dict], lags: list[int]) -> tuple[float, int]:
    print("G0-A — le mouvement est-il lisible À TRAVERS le slot ?\n")
    print(f"  {'décalage Δ':>11} {'déplacement VRAI':>18} {'changement ERREUR':>19} {'SNR':>7}")
    best, best_lag = 0.0, 0
    for L in lags:
        disp, derr = [], []
        for e in eps:
            tw, sw, vis = e["true"], e["slot"], e["vis"]
            for i in range(len(tw) - L):
                if not (vis[i] and vis[i + L]):
                    continue
                d = math.hypot(tw[i + L][0] - tw[i][0], tw[i + L][1] - tw[i][1])
                if d > SWITCH_M * L:          # changement de cible : on écarte
                    continue
                e0 = (sw[i][0] - tw[i][0], sw[i][1] - tw[i][1])
                e1 = (sw[i + L][0] - tw[i + L][0], sw[i + L][1] - tw[i + L][1])
                disp.append(d)
                derr.append(math.hypot(e1[0] - e0[0], e1[1] - e0[1]))
        if len(disp) < 30:
            continue
        md, me = st.median(disp), st.median(derr)
        snr = md / max(me, 1e-9)
        if snr > best:
            best, best_lag = snr, L
        flag = "✅" if snr >= BAR_SNR else ("~" if snr >= STOP_SNR else "  ")
        print(f"  {flag}{L:>9} {md:>15.4f} m {me:>17.4f} m {snr:>7.2f}")
    print(f"\n  meilleur SNR = {best:.2f} au décalage Δ={best_lag}  "
          f"(barre {BAR_SNR}, stop sous {STOP_SNR})")
    return best, best_lag


def g0b(eps: list[dict], lag: int, seed: int) -> float:
    print(f"\nG0-B — l'information est-elle dans le CAPTEUR ? (Δ={lag}, étiquettes PARFAITES)\n")
    X, Y, E = [], [], []
    for k, e in enumerate(eps):
        tw, vis, T, R = e["true"], e["vis"], e["torso"], e["retina"]
        for i in range(len(tw) - lag):
            if not (vis[i] and vis[i + lag]):
                continue
            dx, dz = tw[i + lag][0] - tw[i][0], tw[i + lag][1] - tw[i][1]
            if math.hypot(dx, dz) > SWITCH_M * lag or math.hypot(dx, dz) < 1e-4:
                continue
            c, s = math.cos(T[i][2]), math.sin(T[i][2])
            ego = [T[i + lag][0] - T[i][0], T[i + lag][1] - T[i][1], T[i + lag][2] - T[i][2]]
            X.append(R[i] + R[i + lag] + ego)
            Y.append([dx * c - dz * s, dx * s + dz * c])   # déplacement de la proie, repère t
            E.append(k)
    if len(X) < 200:
        print(f"  ❌ trop peu d'échantillons ({len(X)})")
        return 90.0
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    E = torch.tensor(E)
    uniq = E.unique()
    cut = uniq[int(0.8 * len(uniq))]
    tr, te = E < cut, E >= cut
    print(f"  {len(X)} échantillons · train {int(tr.sum())} · test {int(te.sum())} "
          f"· {len(uniq)} épisodes (découpe PAR ÉPISODE)")
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    for _ in range(1500):
        b = torch.randint(0, int(tr.sum()), (256,))
        loss = torch.nn.functional.mse_loss(net(((X[tr] - mu) / sd)[b]), Y[tr][b])
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        pred = net((X[te] - mu) / sd)
    a = torch.atan2(pred[:, 0], pred[:, 1])
    b = torch.atan2(Y[te][:, 0], Y[te][:, 1])
    err = (a - b).abs()
    err = torch.minimum(err, 2 * math.pi - err) * 180 / math.pi
    med = float(err.median())
    print(f"  erreur angulaire médiane = {med:.1f}°   (hasard = 90°, barre {BAR_ANG}°)")
    return med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=[f"data/replay_buffer/sp2_ref{s}" for s in (1, 2, 3)])
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2_slot/wm_best.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import os
    fov = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360"))
    payload = torch.load(args.wm, map_location="cpu", weights_only=False)
    wm = CommandWorldModel.from_checkpoint(payload)
    wm.eval()
    wm.requires_grad_(False)
    eps = load(args.runs, wm, fov)
    print(f"{len(eps)} épisodes · cône {fov:.0f}°\n")

    snr, lag = g0a(eps, [1, 2, 5, 10, 20, 30, 45, 60])
    ang = g0b(eps, lag if lag else 20, args.seed)

    print("\n" + "=" * 74)
    if snr < STOP_SNR:
        print("🛑 G0-A STOP pré-enregistré — le mouvement n'est lisible à AUCUN décalage.")
        print("   Le slot servi ne peut pas porter le mouvement des objets : son bruit domine")
        print("   à toutes les échelles. ⇒ Le chantier REDIRIGE vers la PRÉCISION DE LA")
        print("   PERCEPTION, comme la réserve du design l'annonçait.")
    elif snr < BAR_SNR:
        print(f"~ G0-A ZONE GRISE (SNR {snr:.2f} à Δ={lag}) — lisible mais fragile.")
    else:
        print(f"✅ G0-A PASSÉ — SNR {snr:.2f} au décalage Δ={lag}.")
    if ang <= BAR_ANG:
        print(f"✅ G0-B PASSÉ — {ang:.0f}° : l'information EST dans la rétine.")
    elif ang > STOP_ANG:
        print(f"🛑 G0-B STOP — {ang:.0f}° : l'information n'est pas dans la rétine.")
    else:
        print(f"~ G0-B ZONE GRISE — {ang:.0f}°.")


if __name__ == "__main__":
    main()
