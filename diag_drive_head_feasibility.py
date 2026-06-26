"""diag_drive_head_feasibility.py — GATE GRATUIT de la VOIE APPRISE (§1), avant tout build de la tête drive-dynamics.
Le WM est aveugle au repas (diag_eat_dynamics : 1% de la bosse). Question : une TÊTE RAPIDE DÉDIÉE, sur la perception
du SLOT (WM gelé), peut-elle apprendre « slot atteint la bouffe → énergie remonte » ? = condition pour débloquer le
critique model-based (§3 : l'eat-dynamics est une tête rapide sur le slot, pas un head WM).

Données : retina_wm_a/b (retina0 → slot_head ; food_rel0 = vérité-terrain perception ; energy ; cmd ; ate).
État (par transition t→t+1) : [slot_dist, energy, vx, om]. Cible : ΔE = energy_{t+1} − energy_t (repas ≈ +32, sinon −drain).
Tête = petit MLP, perte MSE AVEC SUR-PONDÉRATION des repas (sinon l'événement rare est noyé — c'est tout l'intérêt
d'une tête DÉDIÉE vs le readout joint du WM). held-out PAR ÉPISODE.

SUCCÈS : la tête capte la bosse (à held-out : ΔE prédit aux repas > 40% du réel ET discrimine repas vs non-repas)
→ le signal repas EST dans le slot → tête drive-dynamics FAISABLE → critique model-based débloquable.
KILL : même une tête dédiée ne capte pas (capté < ~20%, pas de discrim) → le signal n'est pas dans la représentation
slot → repenser (latent du WM ? autre feature ?) AVANT de builder.

Variante food_rel0 = BORNE HAUTE (perception parfaite) : isole « signal présent ? » de « slot_head assez précis ? ».
Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_drive_head_feasibility.py
"""
import os, glob, json, math
import torch
import torch.nn as nn
from sylvan.models.command_wm import CommandWorldModel

BUFS = os.environ.get("BUFS", "retina_wm_a retina_wm_b").split()
WM_CKPT = os.environ.get("WM_CKPT", "data/checkpoints/wm_objcentric_s1/wm_best.pt")
RETINA_DIM = 144
JUMP = 5.0
EAT_W = 50.0  # sur-pondération des transitions repas dans la perte (tête dédiée → ne pas noyer l'événement rare)


def load_wm():
    ck = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    m = ck["meta"]
    wm = CommandWorldModel(obs_dim=m["obs_dim"], proprio_dim=m["proprio_dim"],
                           predictor_arch=m.get("predictor_arch", "shallow"),
                           with_slot=True, slot_resources=m.get("slot_resources", 1))
    wm.load_state_dict(ck["model"]); wm.eval()
    return wm


@torch.no_grad()
def slot_dist(wm, retina):
    p = wm.slot_encoder.positions(torch.tensor(retina, dtype=torch.float32))[0, :]
    return float(math.hypot(float(p[0]), float(p[1])))


def build(wm, use_truth):
    """retourne liste d'épisodes, chacun = (X[n,4], y[n], eat[n] bool)."""
    eps = []
    for buf in BUFS:
        for f in sorted(glob.glob(f"data/replay_buffer/{buf}/*.jsonl")):
            rows = [json.loads(l) for l in open(f)]
            X, y, eat = [], [], []
            for i in range(len(rows) - 1):
                w = rows[i].get("wm", {})
                if not w.get("retina0") or not w.get("cmd") or not w.get("food_rel0"):
                    continue
                e0 = float(rows[i]["obs"].get("energy", 0.0))
                e1 = float(rows[i + 1]["obs"].get("energy", 0.0))
                if use_truth:
                    fr = w["food_rel0"]; d = math.hypot(float(fr[0]), float(fr[1]))
                else:
                    d = slot_dist(wm, w["retina0"])
                vx, om = float(w["cmd"][0]), float(w["cmd"][1])
                X.append([d, e0 / 100.0, vx, om]); y.append(e1 - e0); eat.append((e1 - e0) > JUMP)
            if len(y) > 50:
                eps.append((torch.tensor(X), torch.tensor(y), torch.tensor(eat)))
    return eps


def run(eps, tag):
    ntr = max(1, int(0.8 * len(eps)))
    Xtr = torch.cat([eps[i][0] for i in range(ntr)]); ytr = torch.cat([eps[i][1] for i in range(ntr)])
    etr = torch.cat([eps[i][2] for i in range(ntr)])
    Xte = torch.cat([eps[i][0] for i in range(ntr, len(eps))]); yte = torch.cat([eps[i][1] for i in range(ntr, len(eps))])
    ete = torch.cat([eps[i][2] for i in range(ntr, len(eps))])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    wtr = torch.where(etr, torch.tensor(EAT_W), torch.tensor(1.0))
    Xn = (Xtr - mu) / sd
    for _ in range(400):
        opt.zero_grad()
        pred = net(Xn).squeeze(-1)
        loss = (wtr * (pred - ytr / 100.0) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        pte = net((Xte - mu) / sd).squeeze(-1) * 100.0
    eat_real = yte[ete].mean().item() if ete.any() else float("nan")
    eat_pred = pte[ete].mean().item() if ete.any() else float("nan")
    non_pred = pte[~ete].mean().item()
    cap = eat_pred / eat_real if eat_real else float("nan")
    discrim = eat_pred - non_pred
    sd_eat = Xte[ete, 0].median().item() if ete.any() else float("nan")
    sd_non = Xte[~ete, 0].median().item()
    print(f"[{tag}] held-out repas test n={int(ete.sum())} | slot_dist médian aux repas={sd_eat:.2f} (non-repas {sd_non:.2f})")
    print(f"        ΔE réel(repas)={eat_real:+.1f}  ΔE prédit(repas)={eat_pred:+.2f}  capté={100*cap:.0f}%  | discrim={discrim:+.2f}")
    return cap, discrim


def main():
    wm = load_wm()
    print("=== (A) tête sur le SLOT APPRIS (slot_head, ce qu'on déploierait) ===")
    capA, dA = run(build(wm, use_truth=False), "slot")
    print("\n=== (B) BORNE HAUTE : tête sur food_rel0 (perception parfaite) ===")
    capB, dB = run(build(wm, use_truth=True), "truth")
    ok = capA > 0.40 and dA > 8.0
    print(f"\n>>> VERDICT (tête sur slot appris) : capté={100*capA:.0f}%(>40%?) discrim={dA:+.1f}(>8?) → "
          f"{'PASS — tête drive-dynamics FAISABLE → critique model-based débloquable' if ok else 'FAIL/KILL — signal repas pas (assez) dans le slot → repenser AVANT build'}")
    if not ok and capB > 0.40:
        print("    NB : la borne haute (food_rel0) PASSE → le signal EXISTE ; c'est la précision du slot_head qui limite (piste : latent du WM, ou slot plus précis au contact).")


if __name__ == "__main__":
    main()
