"""diag_mode1_critic_fit — le critique Mode-1 PEUT-IL apprendre les retours de survie ?

CONTEXTE : Gate-2 (1er run RL) a plateauté au niveau BC ; `value_loss` MORT PLAT ~24k sur
6 itérations → le critique n'apprend pas → avantages bruités → PPO surplace. Hypothèse (§2 :
PAS un plafond de capacité) = défaut de CONDITIONNEMENT : les retours de survie sont NON
normalisés (~centaines) → MSE énorme → gradient énorme → `clip_grad_norm=0.5` PARTAGÉ
(acteur+critique) étrangle tout l'update → le critique avance d'un pas minuscule.

TEST GRATUIT (aucun entraînement RL, aucun Godot) : fitter OFFLINE la `value_head` sur les
retours Monte-Carlo d'un buffer Gate-2 déjà collecté, sous 3 régimes. On mesure le BUT = R²
(fraction de variance des retours EXPLIQUÉE par le critique), pas le `value_loss` brut (proxy
trompeur : 24k peut être « bien » ou « nul » selon l'échelle des cibles).

FALSIFIABLE :
  (A) retours BRUTS,      lr 1e-4, grad-clip 0.5   → mime l'online ; R² attendu ≈ 0 si hypothèse vraie.
  (B) retours NORMALISÉS, lr 1e-4, grad-clip 0.5   → si R² > 0.5 : le critique SAIT fitter → CONDITIONNEMENT.
  (C) retours BRUTS,      lr 1e-3, SANS clip        → isole si (lr+clip) sont l'étranglement.
VERDICT : (B) et/ou (C) >> (A)  → BUG écarté, fix = NORMALISATION retours/valeur (± découpler le grad-clip).
          les 3 ≈ 0             → représentation/bug → creuser (features, gradient, value_coef).

Lancer : PYTHONPATH=python env_pytorch_3.12/bin/python diag_mode1_critic_fit.py \
             [--buffer data/checkpoints/mode1_ppo_gate2/iter_005/buffer] [--epochs 300] [--selfcheck]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sylvan.control.mode1.policy import DriveSymmetricPolicy
from sylvan.control.mode1.rollout_mode1 import _iter_lines, _split_episodes

GAMMA = 0.99
SEED = 0


def _mc_returns(ep: list[dict], gamma: float) -> list[float]:
    """Retour-à-venir Monte-Carlo (reward-to-go), cible du critique INDÉPENDANTE du critique.
    Bootstrap 0 en fin d'épisode (mort ET troncation) : cible propre pour un test de FIT (on teste
    si le critique SAIT fitter une cible cohérente, pas l'exactitude du bootstrap GAE)."""
    g = 0.0
    out = [0.0] * len(ep)
    for t in reversed(range(len(ep))):
        g = float(ep[t]["reward"]) + gamma * g
        out[t] = g
    return out


def _load_dataset(buffer_dir: Path):
    """buffer command-space → (obs empaquetée [N,P+D*38], retours MC [N]). Réutilise la plomberie
    de rollout_mode1 (mêmes _iter_lines/_split_episodes) + build_tokens/pack_obs de la politique."""
    pol = DriveSymmetricPolicy(proprio_dim=132)  # instance pour build_tokens/pack_obs (poids inutilisés ici)
    from sylvan.control.mode1.obs import build_tokens

    lines = list(_iter_lines(buffer_dir))
    episodes = _split_episodes(lines)
    obs_rows, ret_rows = [], []
    for ep in episodes:
        rets = _mc_returns(ep, GAMMA)
        for tr, g in zip(ep, rets):
            proprio, tokens, _ = build_tokens(
                {"proprio": tr["proprio"], "retina": tr["retina"],
                 "energy": tr["energy"], "thirst": tr["thirst"]}
            )
            obs_rows.append(pol.pack_obs(proprio, tokens))
            ret_rows.append(g)
    obs = torch.stack(obs_rows).to(torch.float32)
    returns = torch.tensor(ret_rows, dtype=torch.float32)
    return obs, returns, len(episodes)


def _fit_critic(obs, target, *, lr: float, grad_clip: float | None, epochs: int) -> tuple[float, float]:
    """Fit FULL-BATCH d'un critique FRAIS (value_head + encodeur partagé) sur `target`.
    Renvoie (value_loss_final, R²) où R² = 1 - MSE/Var(target) = fraction de variance expliquée."""
    torch.manual_seed(SEED)  # même init pour comparer les régimes à armes égales
    pol = DriveSymmetricPolicy(proprio_dim=132)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    var = float(target.var(unbiased=False).item()) + 1e-8
    final_loss = float("nan")
    for _ in range(epochs):
        pred = pol.value(obs)                       # [N] via value_head(_features(unpack(obs)))
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(pol.parameters(), grad_clip)
        opt.step()
        final_loss = float(loss.item())
    r2 = 1.0 - final_loss / var
    return final_loss, r2


def selfcheck() -> None:
    pol = DriveSymmetricPolicy(proprio_dim=132)
    from sylvan.control.mode1.obs import build_tokens
    payload = {"proprio": [0.1] * 132, "retina": [0.2] * 144, "energy": 80.0, "thirst": 70.0}
    proprio, tokens, _ = build_tokens(payload)
    assert proprio.shape[-1] == 132, f"proprio={proprio.shape}"
    assert tokens.shape[-1] == 38, f"tok width={tokens.shape}"
    packed = pol.pack_obs(proprio, tokens)
    assert packed.shape[-1] == 132 + tokens.shape[-2] * 38, f"packed={packed.shape}"
    v = pol.value(packed.unsqueeze(0))
    assert v.shape == (1,), f"value shape={v.shape}"
    # MC returns : monotone décroissant sur reward>0 constant (reward-to-go)
    ep = [{"reward": 1.0}] * 5
    rets = _mc_returns(ep, 0.99)
    assert rets[0] > rets[-1] > 0, f"MC returns pas décroissants: {rets}"
    print("[selfcheck] OK — dims (proprio132/tok38/packed), value head, MC returns")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default="data/checkpoints/mode1_ppo_gate2/iter_005/buffer")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    torch.manual_seed(SEED)
    obs, returns, n_ep = _load_dataset(Path(args.buffer))
    n = obs.shape[0]
    r_min, r_max = float(returns.min()), float(returns.max())
    r_mean, r_std = float(returns.mean()), float(returns.std(unbiased=False))
    print(f"buffer={args.buffer}")
    print(f"N transitions={n} | épisodes={n_ep} | obs_dim={obs.shape[1]}")
    print(f"retours MC (γ={GAMMA}): min={r_min:.1f} max={r_max:.1f} mean={r_mean:.1f} std={r_std:.1f}  "
          f"→ échelle {'GRANDE (non-normalisée)' if r_std > 20 else 'petite'}")

    returns_norm = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

    print(f"\nFit critique OFFLINE, {args.epochs} epochs full-batch (critique FRAIS, même seed) :")
    print(f"{'régime':<48}{'value_loss':>14}{'R² (BUT)':>12}")
    print("-" * 74)
    la, r2a = _fit_critic(obs, returns, lr=1e-4, grad_clip=0.5, epochs=args.epochs)
    print(f"{'(A) BRUTS   lr1e-4 clip0.5  [= online]':<48}{la:>14.2f}{r2a:>12.3f}")
    lb, r2b = _fit_critic(obs, returns_norm, lr=1e-4, grad_clip=0.5, epochs=args.epochs)
    print(f"{'(B) NORMALISÉS lr1e-4 clip0.5':<48}{lb:>14.4f}{r2b:>12.3f}")
    lc, r2c = _fit_critic(obs, returns, lr=1e-3, grad_clip=None, epochs=args.epochs)
    print(f"{'(C) BRUTS   lr1e-3 SANS clip':<48}{lc:>14.2f}{r2c:>12.3f}")
    # (D) LE FIX PROPOSÉ : scaling de récompense par constante (÷ ~std), régime online INCHANGÉ
    # (lr 1e-4, clip 0.5). Constante seulement (pas de soustraction de moyenne : le biais du
    # value_head l'absorbe) → GAE reste cohérent (tout est scalé linéairement).
    scale = 1.0 / 250.0
    ld, r2d = _fit_critic(obs, returns * scale, lr=1e-4, grad_clip=0.5, epochs=args.epochs)
    print(f"{'(D) ×(1/250) lr1e-4 clip0.5  [= fix proposé]':<48}{ld:>14.4f}{r2d:>12.3f}")

    print("\n--- VERDICT ---")
    best_alt = max(r2b, r2c)
    if r2a < 0.15 and best_alt > 0.5:
        print(f"CONDITIONNEMENT confirmé : online (A) R²={r2a:.3f} ≈ 0, mais le critique SAIT fitter "
              f"(B/C R²={best_alt:.3f}). Le bug est écarté.")
        print("→ FIX = normalisation des retours/valeur (running mean/std ; dénorm pour GAE)"
              + (" + découpler le grad-clip critique" if r2c > r2b + 0.1 else "") + ", puis relancer Gate-2.")
    elif best_alt < 0.15:
        print(f"Les 3 régimes échouent (meilleur R²={best_alt:.3f}) → PAS un simple conditionnement : "
              "creuser représentation/gradient/value_coef (features insuffisantes ? value_head sans grad ?).")
    else:
        print(f"Résultat intermédiaire : A={r2a:.3f} B={r2b:.3f} C={r2c:.3f} → interpréter (normalisation aide "
              "partiellement ; possible mélange conditionnement + capacité).")


if __name__ == "__main__":
    main()
