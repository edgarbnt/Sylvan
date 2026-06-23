"""Phase 4: train the command-space world model on WM-collect rollouts.

Usage:
    python -m scripts.train_wm_command --runs DIR [DIR ...] --out data/checkpoints/wm_command_v1 \
        [--epochs 20] [--seq-len 64] [--batch-size 16] [--stride 4]

Episodes are split train/val BY EPISODE (no window leakage). Saves wm_best.pt (best val
total loss) + wm_latest.pt, with the val episode list so eval_wm_command uses held-out data.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sylvan.buffer.wm_dataset import (
    CommandSequenceDataset,
    collate_command_samples,
    list_wm_episodes,
)
from sylvan.constants import DEFAULT_PROPRIO_DIM
from sylvan.models.command_wm import (
    DEFAULT_LOSS_WEIGHTS,
    CommandWorldModel,
    compute_command_wm_losses,
    representation_health,
)

LOSS_KEYS = ("latent", "proprio", "radar", "energy", "displacement", "done", "vic_var", "vic_cov")
HEALTH_KEYS = ("lat_std", "lat_std_min", "eff_rank", "offdiag")


def _auc(score: torch.Tensor, label: torch.Tensor) -> float:
    s, l = score.flatten(), label.flatten()
    o = torch.argsort(s); rk = torch.empty_like(s); rk[o] = torch.arange(1, len(s) + 1, dtype=s.dtype, device=s.device)
    np_, nn_ = l.sum().item(), (1 - l).sum().item()
    return float("nan") if np_ == 0 or nn_ == 0 else (rk[l == 1].sum().item() - np_ * (np_ + 1) / 2) / (np_ * nn_)


def _nearest_hit_bearing(obs, proprio_dim, n_ray=36):
    """Cible COLOR-AGNOSTIC (§3) pour la perte clé-de-voûte : bearing égocentrique du PLUS PROCHE objet perçu,
    dérivé de la rétine de l'obs (36 rayons × [depth,R,G,B] ; miss → depth≈1.0). Convention = food_rel0/atan2 :
    rayon k → bearing k·(2π/36) (0=devant, +=droite), wrap (-π,π]. Retourne (target[...,2]=(cos,sin), mask[...])."""
    ret = obs[..., proprio_dim:proprio_dim + 4 * n_ray].reshape(*obs.shape[:-1], n_ray, 4)
    depth = ret[..., 0]
    hit = depth < 0.99
    masked = torch.where(hit, depth, torch.full_like(depth, 2.0))
    ang = masked.argmin(dim=-1).float() * (2.0 * math.pi / n_ray)
    ang = (ang + math.pi) % (2.0 * math.pi) - math.pi
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1), hit.any(dim=-1).float()


def run_epoch(model, loader, device, optimizer=None, scheduled_sampling_prob=0.5, weights=None,
              latent_loss_mode="mse", vicreg=(0.0, 0.0, 1.0), w_food=0.0, w_rollout=0.0, w_bearing=0.0,
              w_bearing_tf=0.0, mirror=None):
    training = optimizer is not None
    model.train(training)
    sums = {k: 0.0 for k in ("loss", *LOSS_KEYS, "food", "rollout", "bearing", "bearing_tf")}
    health_sums = {k: 0.0 for k in HEALTH_KEYS}
    food_scores, food_labels = [], []
    count = 0
    for batch in loader:
        obs = batch.obs.to(device)
        cmd = batch.command.to(device)
        nxt = batch.next_obs.to(device)
        disp = batch.displacement.to(device)
        done = batch.done.to(device)
        eatw = batch.eat_weight.to(device)
        es = batch.eat_soon.to(device)
        # AUGMENTATION MIROIR gauche↔droite (fix PROPRE de l'asymétrie du WM) : miroite ~50% du batch EN PLACE
        # (stochastique, PAS de doublement → zéro surcoût mémoire) — obs/next via la carte 277 (proprio symmetry
        # + rétine ray↔(36−k) + énergie) ; commande omega NÉGÉE ; déplacement d_lat/d_yaw NÉGÉS ; done/eat inchangés.
        # Le WM apprend la symétrie sagittale du CORPS → plus de biais gauche. Même principe que ppo/symmetry (moteur).
        if mirror is not None and training:
            mp, msg = mirror
            idx = (torch.rand(obs.shape[0], device=obs.device) < 0.5).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                obs[idx] = obs[idx][..., mp] * msg
                nxt[idx] = nxt[idx][..., mp] * msg
                cmd[idx, :, 1] = -cmd[idx, :, 1]
                disp[idx, :, 1] = -disp[idx, :, 1]
                disp[idx, :, 2] = -disp[idx, :, 2]
        outputs = model(
            obs,
            cmd,
            scheduled_sampling_prob=scheduled_sampling_prob if training else 1.0,
        )
        losses = compute_command_wm_losses(
            outputs,
            next_obs=nxt,
            displacement=disp,
            done=done,
            eat_weight=eatw,
            model=model,
            proprio_dim=DEFAULT_PROPRIO_DIM,
            weights=weights,
            latent_loss_mode=latent_loss_mode,
            vicreg_var=vicreg[0],
            vicreg_cov=vicreg[1],
            vicreg_gamma=vicreg[2],
        )
        total = losses["loss"]
        food_loss = torch.zeros((), device=device)
        rollout_loss = torch.zeros((), device=device)
        bearing_loss = torch.zeros((), device=device)
        bearing_tf_loss = torch.zeros((), device=device)
        # 3a′ : presse la REPRÉSENTATION — bearing du plus proche objet lu sur les latents TEACHER-FORCED
        # (outputs["latents"]) → force l'encodeur/to_latent à garder le bearing-fin (plafond mesuré REPR ~+0.2).
        if w_bearing_tf > 0.0 and getattr(model, "bearing_head", None) is not None:
            btgt, bmask = _nearest_hit_bearing(obs, DEFAULT_PROPRIO_DIM)
            bpred = model.bearing_head(outputs["latents"])
            bpred = bpred / (bpred.norm(dim=-1, keepdim=True) + 1e-6)
            bearing_tf_loss = (((bpred - btgt) ** 2).sum(-1) * bmask).sum() / (bmask.sum() + 1e-6)
            if training:
                total = total + w_bearing_tf * bearing_tf_loss
        # Un SEUL rollout open-loop COMPLET (= exactement ce que le planner/gate verront), partagé par :
        #   (a) FIDÉLITÉ DYNAMIQUE [w_rollout] — le rêve doit SUIVRE la trajectoire latente réelle
        #       (teacher-forced, stop-grad = cible JEPA). Corrige l'exposure-bias qui fige le rêve en
        #       espace latent RICHE (eff_rank haut via VICReg) : sans ça, nourri de ses prédictions le
        #       prédicteur dérive dès t=1 (cos rêve↔réel 0.59 mesuré). MSE ancre direction ET magnitude
        #       (le collapse que MSE seul permettrait est barré par VICReg). BLUEPRINT §13.
        #   (b) food-aware [w_food] — force le rêve à transporter la bouffe.
        if (w_rollout > 0.0) or (w_food > 0.0 and getattr(model, "food_head", None) is not None) or (w_bearing > 0.0 and getattr(model, "bearing_head", None) is not None):
            ctx = torch.enable_grad() if training else torch.no_grad()
            with ctx:
                dream = model.dream_latents(obs[:, 0, :], cmd)   # [B,T,L]
                if w_rollout > 0.0:
                    rollout_loss = F.mse_loss(dream, outputs["latents"].detach())
                    if training:
                        total = total + w_rollout * rollout_loss
                if w_food > 0.0 and getattr(model, "food_head", None) is not None:
                    food_logit = model.food_head(dream).squeeze(-1)         # [B,T] (es déjà au scope)
                    pw = ((1 - es).sum() / (es.sum() + 1e-6)).clamp(1.0, 50.0)
                    food_loss = F.binary_cross_entropy_with_logits(food_logit, es, pos_weight=pw)
                    if training:
                        total = total + w_food * food_loss
                    else:
                        food_scores.append(torch.sigmoid(food_logit).detach().flatten())
                        food_labels.append(es.flatten())
                if w_bearing > 0.0 and getattr(model, "bearing_head", None) is not None:
                    # CLÉ DE VOÛTE : force le rêve à transporter le bearing du plus proche objet perçu (§3-pur).
                    btgt, bmask = _nearest_hit_bearing(obs, DEFAULT_PROPRIO_DIM)
                    bpred = model.bearing_head(dream)
                    bpred = bpred / (bpred.norm(dim=-1, keepdim=True) + 1e-6)
                    bearing_loss = (((bpred - btgt) ** 2).sum(-1) * bmask).sum() / (bmask.sum() + 1e-6)
                    if training:
                        total = total + w_bearing * bearing_loss
        if training:
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for k in (*LOSS_KEYS,):
            sums[k] += float(losses[k])
        sums["loss"] += float(total)
        sums["food"] += float(food_loss)
        sums["rollout"] += float(rollout_loss)
        sums["bearing"] += float(bearing_loss)
        sums["bearing_tf"] += float(bearing_tf_loss)
        if not training:  # repr-health is a val-only diagnostic (no_grad), BLUEPRINT §13
            for k, v in representation_health(outputs["latents"]).items():
                health_sums[k] += v
        count += 1
    out = {k: v / max(1, count) for k, v in sums.items()}
    if not training:
        out.update({k: v / max(1, count) for k, v in health_sums.items()})
        if food_scores:
            out["food_auc"] = _auc(torch.cat(food_scores), torch.cat(food_labels))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Phase-4 command-space world model.")
    ap.add_argument("--runs", nargs="+", required=True, help="Run dirs with wm-block JSONL episodes.")
    ap.add_argument("--out", required=True, help="Checkpoint output directory.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    # Le ROCm de la box plante (HIP invalid device function) → CPU par défaut, comme le PPO.
    ap.add_argument("--device", default="cpu")
    # Phase B (JEPA-ification): override loss weights to shift reconstruction → latent prediction.
    # Unset → DEFAULT_LOSS_WEIGHTS (= validated wm_command_v2, default run is byte-for-byte unchanged).
    for k in DEFAULT_LOSS_WEIGHTS:
        ap.add_argument(f"--w-{k}", type=float, default=None, help=f"Loss weight for '{k}' (default {DEFAULT_LOSS_WEIGHTS[k]}).")
    ap.add_argument("--predictor-arch", choices=["shallow", "deep"], default="shallow",
                    help="'deep' muscles the JEPA latent predictor (Phase B step 1.1).")
    ap.add_argument("--latent-loss", choices=["mse", "cosine"], default="mse",
                    help="'cosine' = scale-invariant latent loss (Phase B step 1.1).")
    ap.add_argument("--vicreg-var", type=float, default=0.0, help="VICReg variance weight (Phase B step 2; 0=off).")
    ap.add_argument("--vicreg-cov", type=float, default=0.0, help="VICReg covariance weight (Phase B step 2; 0=off).")
    ap.add_argument("--vicreg-gamma", type=float, default=1.0, help="VICReg variance hinge target std.")
    ap.add_argument("--w-food", type=float, default=0.0, help="🅑 poids de la perte auxiliaire food-aware sur "
                    "les latents RÊVÉS (0=off, défaut → run inchangé). Force le rêve à transporter la bouffe.")
    ap.add_argument("--w-rollout", type=float, default=0.0, help="FIDÉLITÉ DU RÊVE (chantier archi 2026-06-19) : "
                    "poids de la perte qui aligne le rollout open-loop sur la trajectoire latente réelle "
                    "(teacher-forced, stop-grad). Corrige l'exposure-bias qui fige le rêve en espace riche. 0=off.")
    ap.add_argument("--w-bearing", type=float, default=0.0, help="CLÉ DE VOÛTE (2026-06-21) : poids de la perte "
                    "auxiliaire 'bearing du plus proche objet perçu À TRAVERS le rollout' (cible color-agnostic "
                    "depuis la rétine → §3-pur). Force le rêve open-loop à transporter la perception sous rotation "
                    "(manque mesuré : rêve corr +0.08, cf diag_wm_rotation). Tête NON sauvée. 0=off.")
    ap.add_argument("--w-bearing-tf", type=float, default=0.0, help="3a′ (2026-06-23) : même perte bearing mais sur "
                    "les latents TEACHER-FORCED (presse la REPRÉSENTATION/encodeur — plafond mesuré REPR ~+0.2). "
                    "Complète --w-bearing (rêve). Tête bearing partagée, NON sauvée. 0=off.")
    ap.add_argument("--init-from", default=None, help="warm-start : charge les poids d'un checkpoint WM "
                    "(strict=False → tolère l'absence de food_head). Évite de ré-apprendre la dynamique de zéro.")
    ap.add_argument("--mirror-augment", action="store_true", help="AUGMENTATION MIROIR gauche↔droite : double "
                    "chaque batch avec sa version miroitée → le WM apprend la symétrie sagittale du corps (fix "
                    "PROPRE de l'asymétrie du rêve, supprime le besoin de la béquille d'inférence). WM-rétine 277.")
    args = ap.parse_args()
    vicreg = (args.vicreg_var, args.vicreg_cov, args.vicreg_gamma)
    if args.vicreg_var or args.vicreg_cov:
        print(f"[train_wm_command] VICReg actif: var={args.vicreg_var} cov={args.vicreg_cov} gamma={args.vicreg_gamma}")

    weights = {**DEFAULT_LOSS_WEIGHTS}
    for k in DEFAULT_LOSS_WEIGHTS:
        v = getattr(args, f"w_{k}")
        if v is not None:
            weights[k] = v
    if weights != DEFAULT_LOSS_WEIGHTS:
        print(f"[train_wm_command] poids JEPA-shift: {weights}")

    device = torch.device(args.device)
    episodes = list_wm_episodes([Path(d) for d in args.runs])
    if not episodes:
        raise SystemExit("Aucun épisode trouvé dans --runs")
    rng = random.Random(args.seed)
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * args.val_frac))
    val_eps, train_eps = episodes[:n_val], episodes[n_val:]
    print(f"[train_wm_command] {len(train_eps)} épisodes train / {len(val_eps)} val | device={device}")

    train_ds = CommandSequenceDataset(train_eps, args.seq_len, args.stride)
    val_ds = CommandSequenceDataset(val_eps, args.seq_len, max(args.stride, 16))
    print(f"[train_wm_command] {len(train_ds)} fenêtres train / {len(val_ds)} val (seq_len={args.seq_len})")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_command_samples, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_command_samples, num_workers=2,
    )

    obs_dim = train_ds.episodes[0]["obs"].shape[-1]
    model = CommandWorldModel(
        obs_dim=obs_dim, proprio_dim=DEFAULT_PROPRIO_DIM, predictor_arch=args.predictor_arch,
        with_food_head=args.w_food > 0.0,
        with_bearing_head=args.w_bearing > 0.0 or args.w_bearing_tf > 0.0,
    ).to(device)
    if args.w_food > 0.0:
        print(f"[train_wm_command] AUXILIAIRE food-aware 🅑 actif: w_food={args.w_food} (tête NON sauvée)")
    if args.w_bearing > 0.0:
        print(f"[train_wm_command] AUXILIAIRE bearing-through-rollout (clé de voûte) actif: w_bearing={args.w_bearing} (tête NON sauvée)")
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        miss_non_food = [k for k in missing if not k.startswith(("food_head", "bearing_head"))]
        print(f"[train_wm_command] WARM-START depuis {args.init_from} "
              f"(missing hors food_head={len(miss_non_food)}, unexpected={len(unexpected)})")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mirror = None
    if args.mirror_augment:
        from sylvan.control.ppo.symmetry import _build_proprio_maps
        pperm, psign = _build_proprio_maps()
        perm = list(range(obs_dim)); sign = [1.0] * obs_dim
        for i in range(min(DEFAULT_PROPRIO_DIM, len(pperm))):
            perm[i] = pperm[i]; sign[i] = psign[i]
        n_ray = (obs_dim - DEFAULT_PROPRIO_DIM - 1) // 4          # rétine = (obs-proprio-énergie)/4 rayons
        for k in range(n_ray):
            src = (n_ray - k) % n_ray                              # miroir azimutal G↔D (validé à l'inférence)
            for j in range(4):
                perm[DEFAULT_PROPRIO_DIM + 4 * k + j] = DEFAULT_PROPRIO_DIM + 4 * src + j
        mirror = (torch.tensor(perm, device=device),
                  torch.tensor(sign, dtype=torch.float32, device=device))
        print(f"[train_wm_command] AUGMENTATION MIROIR active ({n_ray} rayons rétine miroités) → WM symétrique")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "obs_dim": obs_dim,
        "proprio_dim": DEFAULT_PROPRIO_DIM,
        "seq_len": args.seq_len,
        "val_episodes": [str(p) for p in val_eps],
        "loss_weights": weights,
        "predictor_arch": args.predictor_arch,
        "latent_loss": args.latent_loss,
        "vicreg": vicreg,
        "w_rollout": args.w_rollout,
        "w_bearing": args.w_bearing,
        "w_bearing_tf": args.w_bearing_tf,
        "mirror_augment": bool(args.mirror_augment),
    }
    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer, weights=weights,
                       latent_loss_mode=args.latent_loss, vicreg=vicreg, w_food=args.w_food,
                       w_rollout=args.w_rollout, w_bearing=args.w_bearing, w_bearing_tf=args.w_bearing_tf, mirror=mirror)
        va = run_epoch(model, val_loader, device, weights=weights,
                       latent_loss_mode=args.latent_loss, vicreg=vicreg, w_food=args.w_food,
                       w_rollout=args.w_rollout, w_bearing=args.w_bearing, w_bearing_tf=args.w_bearing_tf)
        line = " ".join(f"{k}={va[k]:.4f}" for k in ("loss", *LOSS_KEYS))
        if args.w_rollout > 0.0:
            line += f" rollout={va['rollout']:.4f}"
        if args.w_food > 0.0:
            line += f" food={va['food']:.4f} food_auc={va.get('food_auc', float('nan')):.3f}"
        if args.w_bearing > 0.0:
            line += f" bearing={va['bearing']:.4f}"
        if args.w_bearing_tf > 0.0:
            line += f" bearing_tf={va['bearing_tf']:.4f}"
        # JEPA-ness: share of the (weighted) loss carried by latent prediction vs the recon terms.
        jepa_num = weights["latent"] * va["latent"]
        jepa_den = weights["proprio"] * va["proprio"] + weights["radar"] * va["radar"]
        jepa_ratio = jepa_num / (jepa_den + 1e-12)
        health = " ".join(f"{k}={va[k]:.3f}" for k in HEALTH_KEYS)
        print(
            f"[epoch {epoch:02d}] train_loss={tr['loss']:.4f} | val {line} | "
            f"jepa_ratio={jepa_ratio:.2f} {health} | {time.time()-t0:.0f}s",
            flush=True,
        )
        # NE PAS sauver la tête auxiliaire food_head (aide d'entraînement) → structure de checkpoint
        # INCHANGÉE → tous les loaders existants marchent sans modif. À l'inférence = ValueHead séparée.
        sd = {k: v for k, v in model.state_dict().items() if not k.startswith(("food_head", "bearing_head"))}
        payload = {"model": sd, "meta": meta, "epoch": epoch, "val_loss": va["loss"]}
        torch.save(payload, out / "wm_latest.pt")
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(payload, out / "wm_best.pt")
            print(f"[epoch {epoch:02d}] -> wm_best.pt (val_loss {best_val:.4f})", flush=True)


if __name__ == "__main__":
    main()
