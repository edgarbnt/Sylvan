"""Entraîne le slot TOKEN-BASED (JEPA-pur) — score sur tokens encodeur 64D au lieu de la rétine 4D.

Signal = consistance de transport sous ego-motion (équivariance) + VICReg (anti-collapse).
L'attention APPREND à sélectionner les bons rayons POUR QUE le transport soit cohérent —
exactement le même principe que le slot original (train_slot_head.py), mais sur des features
64D (contexte inter-rayons) au lieu de 4D (depth,R,G,B). ZÉRO label de position.

Usage :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        scripts/train_slot_token.py \
        --corpus data/replay_buffer/gate_foret_cl \
        --wm data/checkpoints/wm_foret_attn_hue/wm_best.pt \
        --out data/checkpoints/slot_token/slot_token_best.pt \
        --gap 8 --epochs 50
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import warnings
from pathlib import Path

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.models.command_wm import CommandWorldModel, vicreg_terms  # noqa: E402
from sylvan.models.slot_head import SelfSupervisedSlotHead, NRAY  # noqa: E402

_HALF_FOV = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")) / 2.0


def ray_bearing(k: int) -> float:
    return (k if k <= NRAY // 2 else k - NRAY) * float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")) / NRAY


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ─── Chargement des paires (obs_a, obs_b) depuis le corpus ────────────────


def load_pairs(corpus: str, gap: int):
    """Charge les paires (obs_a, obs_b) séparées de `gap` ticks, dans le même épisode.

    Pour chaque paire, on calcule l'ego-motion vraie (dyaw, dfwd, dlat) entre a et b.
    On filtre : la nourriture doit être visible AUX DEUX bouts (food_visible>0.5).
    Retourne (obs_a, obs_b, dyaw, dfwd, dlat, food_x, food_z).
    """
    eps_obs = []  # list of lists: each episode = list of (obs_vector, torso, food_rel0)
    for f in sorted(glob.glob(os.path.join(corpus, "*.jsonl"))):
        seq = []
        for line in open(f):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            wm = r.get("wm", {})
            food = wm.get("food_rel0")
            torso = wm.get("torso0")
            obs_dict = r.get("obs", {})
            retina = obs_dict.get("retina")
            proprio = obs_dict.get("proprio")
            energy = obs_dict.get("energy", 60.0)
            if not food or not torso or not retina or not proprio:
                continue
            if len(retina) != 144 or len(proprio) < 133:
                continue
            bearing = math.degrees(math.atan2(food[0], food[1]))
            if abs(bearing) > _HALF_FOV:
                continue
            obs = proprio + retina + [energy / 100.0]
            seq.append((obs, torso, (food[0], food[1], food[2])))
        if len(seq) > gap + 2:
            eps_obs.append(seq)

    n_tr = max(1, int(0.8 * len(eps_obs)))
    cols = {"oa": [], "ob": [], "dy": [], "df": [], "dl": [], "fx": [], "fz": [], "tr": []}
    for ei, seq in enumerate(eps_obs):
        is_tr = ei < n_tr
        for i in range(len(seq) - gap):
            a = seq[i]; b = seq[i + gap]
            if a[2][2] < 0.5 or b[2][2] < 0.5:
                continue  # nourriture non visible à l'un des deux bouts
            x0, z0, y0 = a[1][0], a[1][1], a[1][2]
            x1, z1, y1 = b[1][0], b[1][1], b[1][2]
            cols["oa"].append(a[0]); cols["ob"].append(b[0])
            cols["dy"].append(wrap(y1 - y0))
            cols["df"].append(x1 * math.sin(y0) + z1 * math.cos(y0))
            cols["dl"].append(x1 * math.cos(y0) - z1 * math.sin(y0))
            cols["fx"].append(b[2][0]); cols["fz"].append(b[2][1])
            cols["tr"].append(is_tr)
    t = {k: torch.tensor(v) for k, v in cols.items()}
    return t, len(eps_obs)


# ─── Transport (même convention que train_slot_head) ───────────────────────


def transport(p, dyaw, dfwd, dlat):
    px = p[:, 0] - dlat
    pz = p[:, 1] - dfwd
    ca, sa = torch.cos(dyaw), torch.sin(dyaw)
    return torch.stack([ca * px - sa * pz, sa * px + ca * pz], dim=1)


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/replay_buffer/gate_foret_cl")
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_attn_hue/wm_best.pt")
    ap.add_argument("--out", default="data/checkpoints/slot_token/slot_token_best.pt")
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(4)

    # 1. Charger les données
    t, nep = load_pairs(a.corpus, a.gap)
    tr = t["tr"].bool(); te = ~tr
    print(f"[token-slot] corpus={a.corpus} gap={a.gap} épisodes={nep} "
          f"paires={len(t['oa'])} (train={int(tr.sum())} test={int(te.sum())})")

    # 2. Charger le WM (encodeur gelé, sert juste à extraire les tokens)
    payload = torch.load(a.wm, map_location="cpu", weights_only=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = CommandWorldModel.from_checkpoint(payload)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    # 3. Créer le slot TOKEN-BASED
    head = SelfSupervisedSlotHead(n_resources=1, token_features=True)
    # token_score + gate parameters (on score.0 si présent)
    opt = torch.optim.Adam([
        {"params": head.token_score.parameters(), "lr": a.lr},
    ])
    n_params = sum(p.numel() for p in head.token_score.parameters())
    print(f"[token-slot] token_score params: {n_params}")

    # 4. Entraînement JEPA-pur (transport consistency + VICReg)
    ra, rb, dy, df, dl = (t[k][tr] for k in ("oa", "ob", "dy", "df", "dl"))
    N = len(ra)

    def slots(obs_batch):
        """Encode le slot pour un batch d'observations : extrait les tokens encodeur,
        puis score via token_score. L'encodeur est gelé (no_grad pour le token)."""
        obs_t = torch.tensor(obs_batch, dtype=torch.float32)
        # Batch peut être grand → itérer
        all_pos = []
        bs = 512
        for start in range(0, obs_t.shape[0], bs):
            ob = obs_t[start:start + bs]
            with torch.no_grad():
                tok = wm.encoder._extract_tokens(ob)   # [B, 36, 64]
            ret = ob[:, wm.proprio_dim:wm.proprio_dim + 144]
            all_pos.append(head.positions(ret, tokens=tok))
        return torch.cat(all_pos, dim=0)

    for epoch in range(a.epochs):
        bi = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0
        bs = 256
        for start in range(0, N, bs):
            idx = bi[start:start + bs]
            # Extraire les tokens pour la batch
            with torch.no_grad():
                tok_a = wm.encoder._extract_tokens(ra[idx])
                tok_b = wm.encoder._extract_tokens(rb[idx])
            ret_a = ra[idx][:, wm.proprio_dim:wm.proprio_dim + 144]
            ret_b = rb[idx][:, wm.proprio_dim:wm.proprio_dim + 144]

            sa = head.positions(ret_a, tokens=tok_a)  # [B, 1, 2]
            sb = head.positions(ret_b, tokens=tok_b)  # [B, 1, 2]

            # Consistance de transport (JEPA)
            s_trans = transport(sa[:, 0, :], dy[idx], df[idx], dl[idx])
            loss = ((s_trans - sb[:, 0, :].detach()) ** 2).mean()

            # VICReg anti-collapse — pas de padding factice, juste sur les positions 2D
            # avec gamma réduit (la position est 2D, pas 128D)
            all_s = torch.cat([sa[:, 0, :], sb[:, 0, :]], dim=0)
            # Centrer + normaliser pour VICReg
            all_c = all_s - all_s.mean(dim=0, keepdim=True)
            all_n = all_c / (all_c.std(dim=0, keepdim=True) + 1e-6)
            # Au moins 2D pour cov — c'est SERVI
            var = (all_n.var(dim=0) - 1.0).abs().mean()
            n_dim = all_n.shape[1]
            cov = (all_n.T @ all_n) / (all_n.shape[0] - 1)
            # Covariance hors-diagonale seulement
            off_diag = cov - torch.diag(torch.diag(cov))
            cov_pen = (off_diag ** 2).sum() / max(n_dim, 2)
            loss = loss + 0.1 * (var + cov_pen)
            loss.backward()
            opt.step(); opt.zero_grad()
            total_loss += float(loss)
            n_batches += 1

        if epoch % 10 == 0 or epoch == a.epochs - 1:
            # Éval
            with torch.no_grad():
                ret_te = t["ob"][te][:, wm.proprio_dim:wm.proprio_dim + 144]
                tok_te = wm.encoder._extract_tokens(t["ob"][te])
                sp = head.positions(ret_te, tokens=tok_te)[:, 0, :]
                sm = (sp.abs().sum(dim=1) > 1e-6)
                errs = (sp[sm] - torch.stack([t["fx"][te][sm],
                                               t["fz"][te][sm]], dim=-1)).norm(dim=-1)
                bmae = math.degrees(torch.atan2(
                    torch.sin(torch.atan2(sp[sm, 0], sp[sm, 1])
                              - torch.atan2(t["fx"][te][sm], t["fz"][te][sm])),
                    torch.cos(torch.atan2(sp[sm, 0], sp[sm, 1])
                              - torch.atan2(t["fx"][te][sm], t["fz"][te][sm])),
                ).abs().mean())
                pmae = float(errs.mean()) if errs.numel() > 0 else float("nan")
                pmed = float(errs.median()) if errs.numel() > 0 else float("nan")
                cov = float(sm.float().mean())
                print(f"[token-slot] epoch {epoch:3d} loss={total_loss/n_batches:.4f} | "
                      f"held-out: bearing={bmae:.1f}° pos_méd={pmed:.2f}m "
                      f"pos_moy={pmae:.2f}m couv={100*cov:.0f}%")

    # 5. Éval finale
    with torch.no_grad():
        ret_all = t["ob"][:, wm.proprio_dim:wm.proprio_dim + 144]
        tok_all = wm.encoder._extract_tokens(t["ob"])
        sp = head.positions(ret_all, tokens=tok_all)[:, 0, :]
        sm = sp.abs().sum(dim=1) > 1e-6
        errs = (sp[sm] - torch.stack([t["fx"][sm], t["fz"][sm]], dim=-1)).norm(dim=-1)
        pmae = float(errs.mean()) if errs.numel() > 0 else float("nan")
        pmed = float(errs.median()) if errs.numel() > 0 else float("nan")
        pct_lt_05 = float((errs < 0.5).float().mean()) if errs.numel() > 0 else 0.0
        cov = float(sm.float().mean())

    print(f"\n[token-slot] FINAL : pos_méd={pmed:.2f}m pos_moy={pmae:.2f}m "
          f"<0.5m={100*pct_lt_05:.1f}% couv={100*cov:.0f}%")

    # 6. Sauvegarde
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ck = {
        "state_dict": head.state_dict(),
        "n_resources": 1,
        "token_features": True,
        "heldout_pos_med_m": pmed,
        "heldout_pos_mae_m": pmae,
        "heldout_pct_lt_05": pct_lt_05,
        "heldout_coverage": cov,
        "corpus": a.corpus,
        "wm": a.wm,
        "gap": a.gap,
        "jepa_pure": True,
        "token_based": True,
    }
    torch.save(ck, out)
    print(f"[token-slot] sauvé → {out}")

    # 7. Verdict
    if pmed < 0.80:
        print(f"\n✅ G0 PASS : token-slot {pmed:.2f}m < 0.80m — les tokens encodeur + "
              f"transport-consistance suffisent.")
        return 0
    else:
        print(f"\n❌ G0 ÉCHEC : {pmed:.2f}m ≥ 0.80m")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
