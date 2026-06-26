"""diag_drive_head_latent.py — VARIANTE LATENT du gate voie-apprise (§1, suite de diag_drive_head_feasibility).
La tête sur le scalaire slot_dist captait 14% (borne haute food_rel0 = 46% → signal présent mais slot_dist lossy).
Ici : la tête lit le LATENT du WM (128-d, ce que le critique model-based utiliserait) → capte-t-elle MIEUX la bosse ?
Cible : ΔE = energy_{t+1} − energy_t. held-out PAR ÉPISODE. Sous-pondération repas (rare). Non-repas sous-échantillonnés.

SUCCÈS : capté > 40% ET discrim > 8 → le latent porte le signal repas → tête drive-dynamics sur le latent FAISABLE.
Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_drive_head_latent.py
"""
import os, glob, json
import torch
import torch.nn as nn
from sylvan.models.command_wm import CommandWorldModel

BUFS = os.environ.get("BUFS", "retina_wm_a retina_wm_b").split()
WM_CKPT = os.environ.get("WM_CKPT", "data/checkpoints/wm_objcentric_s1/wm_best.pt")
JUMP = 5.0
EAT_W = 50.0
NON_KEEP = 4000  # sous-échantillon de non-repas (les repas, tous gardés)


def load_wm():
    ck = torch.load(WM_CKPT, map_location="cpu", weights_only=False); m = ck["meta"]
    wm = CommandWorldModel(obs_dim=m["obs_dim"], proprio_dim=m["proprio_dim"],
                           predictor_arch=m.get("predictor_arch", "shallow"),
                           with_slot=True, slot_resources=m.get("slot_resources", 1))
    wm.load_state_dict(ck["model"]); wm.eval(); return wm, m


@torch.no_grad()
def latents_batch(wm, obs, cmd):
    out = wm.rollout_open_loop(obs, cmd.unsqueeze(1))  # [B,obs_dim],[B,1,2]
    return out["predicted_latents"][:, 0, :]  # [B, latent]


def main():
    wm, m = load_wm()
    torch.manual_seed(0)
    # collecte (wm_obs, cmd, ΔE, eat, ep)
    obs_l, cmd_l, dy_l, eat_l, ep_l = [], [], [], [], []
    ep = 0
    for buf in BUFS:
        for f in sorted(glob.glob(f"data/replay_buffer/{buf}/*.jsonl")):
            rows = [json.loads(l) for l in open(f)]
            had = False
            for i in range(len(rows) - 1):
                w = rows[i].get("wm", {})
                if not w.get("retina0") or not w.get("cmd"):
                    continue
                e0 = float(rows[i]["obs"].get("energy", 0.0)); e1 = float(rows[i + 1]["obs"].get("energy", 0.0))
                de = e1 - e0; eat = de > JUMP
                if not eat and torch.rand(1).item() > NON_KEEP / 80000.0:
                    continue
                obs_l.append(rows[i]["obs"]["proprio"] + w["retina0"] + [e0 / 100.0])
                cmd_l.append([float(w["cmd"][0]), float(w["cmd"][1])])
                dy_l.append(de); eat_l.append(eat); ep_l.append(ep); had = True
            if had:
                ep += 1
    obs = torch.tensor(obs_l, dtype=torch.float32); cmd = torch.tensor(cmd_l, dtype=torch.float32)
    dy = torch.tensor(dy_l); eat = torch.tensor(eat_l); epi = torch.tensor(ep_l)
    print(f"transitions={len(dy)} (repas={int(eat.sum())}) sur {ep} épisodes")
    # latents en batch
    lat = torch.cat([latents_batch(wm, obs[i:i+2048], cmd[i:i+2048]) for i in range(0, len(obs), 2048)])
    # split par épisode 80/20
    ntr = max(1, int(0.8 * ep)); tr = epi < ntr; te = ~tr
    mu, sd = lat[tr].mean(0), lat[tr].std(0) + 1e-6
    net = nn.Sequential(nn.Linear(lat.shape[1], 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xtr = (lat[tr] - mu) / sd; ytr = dy[tr] / 100.0; wtr = torch.where(eat[tr], torch.tensor(EAT_W), torch.tensor(1.0))
    for _ in range(500):
        opt.zero_grad(); pred = net(Xtr).squeeze(-1)
        (wtr * (pred - ytr) ** 2).mean().backward(); opt.step()
    with torch.no_grad():
        pte = net((lat[te] - mu) / sd).squeeze(-1) * 100.0
    ete = eat[te]
    er = dy[te][ete].float().mean().item(); ep_ = pte[ete].mean().item(); npd = pte[~ete].mean().item()
    cap = ep_ / er if er else float("nan"); discrim = ep_ - npd
    print(f"[latent] held-out repas n={int(ete.sum())} | ΔE réel={er:+.1f}  ΔE prédit={ep_:+.2f}  capté={100*cap:.0f}%  discrim={discrim:+.2f}")
    ok = cap > 0.40 and discrim > 8.0
    print(f"\n>>> VERDICT (tête sur LATENT) : capté={100*cap:.0f}%(>40%?) discrim={discrim:+.1f}(>8?) → "
          f"{'PASS — le latent porte le signal repas → tête drive-dynamics FAISABLE' if ok else 'PARTIEL/FAIL — le latent ne suffit pas non plus → repenser la représentation au contact'}")


if __name__ == "__main__":
    main()
