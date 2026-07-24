"""LE JUGE DU CRITIQUE — classement INTRA-ÉTAT des 21 candidats, contre les issues RÉELLES.

POURQUOI CELUI-CI ET PAS UNE AUC. Le planner ne compare pas des états entre eux : il classe 117
candidats DANS LE MÊME ÉTAT. Juger un critique au R²/AUC POOLÉ est le bug de mesure derrière les 3
échecs historiques. Ici on pose la seule question qui décide :

    à un fork conséquent, le critique classe-t-il les commandes MIEUX que `-min_dist`
    (le terme qu'il est censé remplacer), en confrontant chaque classement aux repas
    RÉELLEMENT obtenus (mesurés en Godot déterministe) ?

MÉTHODE. On part de l'état RÉEL au fork (log BC tronqué au tick k). On rêve chaque candidat dans le
WM gelé (`rollout_open_loop` → trajectoire de slot, exactement la mécanique du planner), puis :
  * score CRITIQUE   = moyenne de V(token) sur la trajectoire rêvée   (agrégat `mean`, comme le planner)
  * score ANALYTIQUE = −min(distance du slot) sur la trajectoire rêvée (le coût actuel)
et on confronte les deux classements aux issues mesurées.

VERDICT (pré-enregistré) :
  * le critique GAGNE s'il fait au moins aussi bien que l'analytique sur le top-1 (le candidat
    qu'il choisirait obtient-il le max mesuré ?) ET a une AUC intra-état supérieure.
  * il PERD s'il choisit un candidat strictement moins bon que celui de l'analytique.
L'AUC intra-état est calculée sur les 21 candidats de CE fork (issue binaire max vs non-max).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_critic_intra_state.py \
      --fork data/forks/s8_k1200 --outcomes data/forks/s8_k1200_outcomes.txt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from sylvan.critic_corpus import auc, token
from sylvan.models.command_wm import CommandWorldModel
from scripts.train_meal_critic import MealCritic

VXS = [0.55, 0.65, 0.75]
OMS = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]


def load_fork_state(path: Path) -> tuple[torch.Tensor, float]:
    """Dernière ligne du log BC tronqué = l'état AU fork. → (obs [1,277], énergie)."""
    last = None
    with open(path / "ep_0000.jsonl") as fh:
        for line in fh:
            last = line
    r = json.loads(last)
    e = float(r["obs"]["energy"])
    obs = torch.tensor(
        r["obs"]["proprio"] + r["wm"]["retina0"] + [e / 100.0], dtype=torch.float32
    ).unsqueeze(0)
    return obs, e


def load_outcomes(path: Path) -> list[int]:
    """Table du probe → repas mesurés, dans l'ordre (vx, om) de VXS × OMS."""
    out = []
    for line in open(path):
        m = re.match(r"^\s+(0\.\d+)\s+([+-]\d\.\d)\s+(\d+)\s*$", line)
        if m:
            out.append(int(m.group(3)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", default="data/forks/s8_k1200")
    ap.add_argument("--outcomes", default="data/forks/s8_k1200_outcomes.txt")
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--critic", default="data/checkpoints/meal_critic/critic_best.pt")
    ap.add_argument("--critic-type", choices=["token", "latent", "td"], default="token")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--horizon", type=int, default=80)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)

    obs0, energy = load_fork_state(Path(args.fork))
    outcomes = load_outcomes(Path(args.outcomes))
    if len(outcomes) != 21:
        raise SystemExit(f"attendu 21 issues mesurées, trouvé {len(outcomes)} "
                         f"(le probe est-il terminé ?)")

    pl = torch.load(args.wm, map_location="cpu", weights_only=False)
    meta = pl["meta"]
    wm = CommandWorldModel(
        obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
        predictor_arch=meta.get("predictor_arch", "shallow"),
        with_slot=True, slot_resources=meta.get("slot_resources", 1),
    )
    wm.load_state_dict(pl["model"])
    wm.eval()

    ck = torch.load(args.critic, map_location="cpu", weights_only=False)
    if args.critic_type == "td":
        from scripts.train_td_critic import TDValueHead
        critic = TDValueHead(ck["latent_dim"], ck.get("hidden", 256))
    elif args.critic_type == "latent":
        from sylvan.models.value_head import ValueHead
        critic = ValueHead(ck["latent_dim"], ck.get("hidden", 256))
    else:
        critic = MealCritic()
    critic.load_state_dict(ck["model"])
    critic.eval()

    cmds = torch.tensor([[vx, om] for vx in VXS for om in OMS], dtype=torch.float32)   # [21,2]
    seq = cmds.unsqueeze(1).expand(-1, args.horizon, -1).contiguous()                  # commande TENUE
    with torch.no_grad():
        out = wm.rollout_open_loop(obs0.expand(21, -1).contiguous(), seq)
        slot = out["slot"]                                                             # [21, T, 2]
        if args.critic_type == "td":
            # FORME TD-MPC : la valeur est TERMINALE (au BOUT du rêve court), jamais moyennée.
            # C'est elle qui fait entrer l'information de long horizon dans un plan de court horizon.
            v_critic = (args.gamma ** args.horizon) * critic(out["predicted_latents"][:, -1])
        elif args.critic_type == "latent":
            # MÊME entrée qu'à l'entraînement : les latents RÊVÉS open-loop.
            v_critic = critic.value(out["predicted_latents"]).mean(dim=1)
        else:
            lvl = torch.full(slot.shape[:2], energy / 100.0)
            v_critic = critic.value(token(lvl, slot)).mean(dim=1)                      # agrégat mean
        v_analytic = -slot.norm(dim=-1).min(dim=1).values                              # -min_dist

    y = torch.tensor(outcomes, dtype=torch.float32)
    best = float(y.max())
    is_best = (y == best).float()
    print(f"fork {args.fork} | énergie {energy:.1f} | issues mesurées : "
          f"{int(is_best.sum())}/21 candidats atteignent le max ({best:.0f} repas)")
    print()
    print("  vx    om    repas   V_critique   -min_dist")
    for i, (vx, om) in enumerate([(a, b) for a in VXS for b in OMS]):
        print(f"  {vx:.2f}  {om:+.1f}     {outcomes[i]}      {float(v_critic[i]):.4f}      "
              f"{float(v_analytic[i]):+.3f}")

    print()
    for name, s in ((f"CRITIQUE ({args.critic_type})", v_critic), ("ANALYTIQUE (-min_dist)", v_analytic)):
        top = int(s.argmax())
        a = auc(s, is_best)
        print(f"  {name:24s} : top-1 = ({[(x, o) for x in VXS for o in OMS][top]}) -> "
              f"{outcomes[top]} repas | AUC intra-état {a:.3f}")

    t_c, t_a = int(v_critic.argmax()), int(v_analytic.argmax())
    a_c, a_a = auc(v_critic, is_best), auc(v_analytic, is_best)
    base = float(is_best.mean())
    print()
    # ⚠️ LE TOP-1 SEUL NE PROUVE RIEN quand la plupart des candidats atteignent le max : ici un tirage
    # AU HASARD « gagne » avec probabilité `base`. Le verdict exige donc AUSSI un classement meilleur
    # (AUC intra-état), conformément au pré-enregistrement. Sans ça on relaie de la chance.
    print(f"  (taux de base : {100 * base:.0f} % des candidats atteignent le max -> un choix ALÉATOIRE "
          f"'gagne' le top-1 {100 * base:.0f} % du temps ; le top-1 seul n'est pas une preuve)")
    if outcomes[t_c] >= outcomes[t_a] and a_c > a_a:
        print("  -> LE CRITIQUE GAGNE : top-1 au moins aussi bon ET meilleur classement.")
    elif a_c < 0.5:
        print(f"  -> LE CRITIQUE ÉCHOUE : son classement est ANTI-corrélé aux bonnes issues "
              f"(AUC {a_c:.3f} < 0,5). Un top-1 favorable ici serait de la chance.")
    else:
        print(f"  -> PAS DE GAIN démontré : classement {a_c:.3f} vs {a_a:.3f} pour l'analytique.")


if __name__ == "__main__":
    main()
