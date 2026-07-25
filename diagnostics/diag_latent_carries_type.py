"""PRÉCONDITION — le latent porte-t-il le TYPE de la proie VISÉE ?

POURQUOI CETTE SONDE AVANT D'ENTRAÎNER. Le monde v7 rend la valeur d'une proie ARBITRAIRE : elle
dépend de son type (sa teinte), et rien dans la géométrie ne la prédit. C'est la seule condition
mesurée où un critique est NÉCESSAIRE (formule 49,5 % vs appris 69,7 % de la marge oracle).
Mais un critique ne peut apprendre la table type→valeur que s'il PERÇOIT le type. Or on a mesuré :
  * le latent porte l'apparence à la profondeur 0 (maturité lisible à R² 0,65),
  * et cette information SE DÉGRADE le long du rêve (0,556 → 0,160 entre profondeur 0 et 79).
Si le type n'est pas lisible, le critique est condamné et il faut le savoir AVANT de payer un
entraînement — c'est la discipline qui a permis de tuer quatre pistes pour zéro run.

CE QU'ON MESURE. La vérité-terrain est calculée depuis la RÉTINE elle-même (le type EST la teinte) :
on prend les rayons « nourriture » les plus PROCHES — ceux que le slot cible — et on lit leur teinte.
Puis on demande : cette teinte est-elle lisible dans le LATENT du WM ?

  précision ≈ hasard (25 % pour 4 types) -> le latent n'expose pas le type -> critique condamné.
  précision nettement > hasard           -> le critique peut apprendre la table. On peut payer.

Le WM est GELÉ : on mesure ce qu'il contient déjà, on ne le ré-entraîne pas (§3).

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_latent_carries_type.py
"""
from __future__ import annotations

import argparse

import torch
from torch import nn

from sylvan.critic_corpus import load_bc_corpora
from sylvan.models.command_wm import CommandWorldModel

# Palette MESURÉE du monde v7 (food_manager.TYPE_COLORS) — toutes dans le cône bouffe.
TYPE_COLORS = torch.tensor([[0.90, 0.10, 0.10], [0.80, 0.60, 0.15],
                            [0.90, 0.10, 0.45], [0.85, 0.55, 0.35]])
# 🚨 La palette de RÉFÉRENCE doit être celle RÉELLEMENT SERVIE. Étiqueter le monde-forêt avec la
# palette v7 introduit du bruit d'étiquette : deux teintes voisines se retrouvent sous le même
# prototype, et une précision plafonnée par l'étiquetage se lirait comme un encodeur aveugle. On
# choisit donc la palette qui EXPLIQUE le corpus (même mécanisme que info_matrix.pick_palette).
FORET_V1_HUES = torch.tensor([[0.90, 0.12, 0.10], [0.90, 0.55, 0.08],
                              [0.85, 0.10, 0.45], [0.80, 0.42, 0.42]])


def pick_reference(picked: torch.Tensor) -> tuple[torch.Tensor, str]:
    """La palette dont les teintes rendues sont le plus PROCHES : mesurée, jamais supposée."""
    best, name = None, ""
    for cand_name, pal in (("v7", TYPE_COLORS), ("foret_v1", FORET_V1_HUES)):
        ref = pal / pal.norm(dim=-1, keepdim=True)
        score = float((picked @ ref.T).max(dim=1).values.median())
        if best is None or score > best:
            best, name, chosen = score, cand_name, ref
    return chosen, f"{name} (cos médian {best:.4f})"


def target_type_from_retina(retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Type de la proie CIBLÉE (= le rayon nourriture le plus proche), lu depuis la rétine.

    → (type [N] dans 0..3, valide [N]). Vérité-terrain honnête : le type EST la teinte rendue, on ne
    triche pas en lisant un état caché du monde.
    """
    r = retina.reshape(len(retina), 36, 4)
    depth, rgb = r[..., 0], r[..., 1:4]
    n = rgb.norm(dim=-1, keepdim=True) + 1e-6
    u = rgb / n
    is_food = (u[..., 0] > 0.55) & (rgb.norm(dim=-1) > 1e-3)      # même critère que le slot
    d = torch.where(is_food, depth, torch.full_like(depth, 9e9))
    nearest = d.argmin(dim=1)
    valid = is_food.any(dim=1)
    picked = u[torch.arange(len(u)), nearest]                      # teinte NORMALISÉE du rayon visé
    ref, why = pick_reference(picked[valid] if valid.any() else picked)
    typ = (picked @ ref.T).argmax(dim=1)                           # type = teinte de référence la plus proche
    return typ, valid, why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=["data/replay_buffer/critic_bosq_typ31"])
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)

    obs, energy, cmds, bounds = load_bc_corpora(args.corpus)
    pl = torch.load(args.wm, map_location="cpu", weights_only=False)
    meta = pl["meta"]
    wm = CommandWorldModel(
        obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
        predictor_arch=meta.get("predictor_arch", "shallow"),
        # L'architecture d'encodeur se LIT dans la meta : la reconstruire au défaut ferait échouer le
        # chargement sur un WM à attention, ou pire, sonderait un modèle différent de celui entraîné.
        retina_attention=meta.get("retina_attention", False),
        with_slot=True, slot_resources=meta.get("slot_resources", 1),
    )
    wm.load_state_dict(pl["model"])
    wm.eval()
    P = meta["proprio_dim"]

    idx = torch.arange(0, len(energy) - 2, args.stride)
    typ, valid, why = target_type_from_retina(obs[idx, P:P + 144])
    idx, typ = idx[valid], typ[valid]
    # 🚨 DEUX SITES, ET C'EST TOUT L'ENJEU DE LA MESURE. `predicted_latents[:, 0]` n'est PAS la
    # sortie de l'encodeur : c'est to_latent(GRU(encoder(obs), cmd)) — donc DÉJÀ passé par le
    # goulot récurrent. Sonder ce seul site ne peut pas distinguer « l'encodeur jette l'apparence »
    # de « le GRU/to_latent la jette ». Les deux appellent des corrections opposées, donc on mesure
    # les deux : ENCODEUR (ce que la perception extrait) et LATENT (ce que la dynamique conserve).
    with torch.no_grad():
        enc = torch.cat([wm.encoder(obs[idx[i:i + 256]]) for i in range(0, len(idx), 256)])
        lat = torch.cat([
            wm.rollout_open_loop(obs[idx[i:i + 256]],
                                 cmds[idx[i:i + 256]].unsqueeze(1).expand(-1, max(2, args.depth + 1), -1).contiguous()
                                 )["predicted_latents"][:, args.depth]
            for i in range(0, len(idx), 256)])

    counts = torch.bincount(typ, minlength=4).float()
    majority = float(counts.max() / counts.sum())
    print(f"  {len(idx)} états avec une proie en vue | répartition des types "
          f"{[int(c) for c in counts]} | majorité {majority:.1%} (hasard à battre)")
    print(f"  palette de référence retenue : {why}")

    def probe(rep: torch.Tensor, site: str) -> None:
        ntr = int(0.7 * len(rep))
        x = (rep - rep[:ntr].mean(0)) / rep[:ntr].std(0).clamp_min(1e-6)
        for name, net in (("LINÉAIRE", nn.Linear(x.shape[1], 4)),
                          ("MLP", nn.Sequential(nn.Linear(x.shape[1], 256), nn.SiLU(),
                                                nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 4)))):
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            lossf = nn.CrossEntropyLoss()
            for _ in range(3000):
                i = torch.randperm(ntr)[:512]
                opt.zero_grad()
                lossf(net(x[:ntr][i]), typ[:ntr][i]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                acc = float((net(x[ntr:]).argmax(1) == typ[ntr:]).float().mean())
            print(f"  {site:26s} dim {x.shape[1]:4d} — {name:9s} : précision held-out {acc:.1%}")

    probe(enc, "ENCODEUR (perception)")
    probe(lat, f"LATENT RSSM (prof. {args.depth})")

    print("\n  LECTURE — les deux sites ne disent PAS la même chose :")
    print("    ENCODEUR ≈ majorité      -> la PERCEPTION jette l'apparence (fix = pression sur l'encodeur).")
    print("    ENCODEUR bon, LATENT bas -> c'est le goulot RÉCURRENT qui la jette (fix = tout autre).")
    print("    les deux bons            -> le type est disponible, le critique peut apprendre la table.")


if __name__ == "__main__":
    main()
