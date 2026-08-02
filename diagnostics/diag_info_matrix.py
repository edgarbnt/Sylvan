"""MATRICE DE SURVIE DE L'INFORMATION — une commande, un tableau : où l'information MEURT.

LIGNES = propriétés du monde. COLONNES = étages du pipeline (rétine brute -> encodeur -> latent
rêvé d0 -> latent dH -> slot -> token du planner). Chaque case = la part RÉCUPÉRABLE de l'information
(R² pour un continu, précision + majorité pour une catégorie), sonde LINÉAIRE et sonde MLP, en
held-out PAR ÉPISODE. Toute chute entre deux colonnes est une piste, et on sait quel module accuser.

GRATUITE : aucun entraînement du substrat, on lit un corpus déjà collecté et un WM GELÉ.
NON-RÉGRESSION : la relancer après chaque retrain et comparer les colonnes — une case qui baisse est
une régression du substrat, visible avant de dépenser une heure d'A/B.

Design : docs/design_outil_matrice_information.md. Mécanique de mesure : python/sylvan/info_matrix.py.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_info_matrix.py \
      --corpus data/replay_buffer/critic_bosq_ripe11 [--depths 0 20 79] [--rows type vue]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # guards.py (idiome du repo)

from guards import sanity, scaffold_banner                              # noqa: E402
from sylvan.critic_corpus import load_bc_corpora, meal_flags, residual_label  # noqa: E402
from sylvan.info_matrix import (                                        # noqa: E402
    PALETTES, PROPERTIES, PROPERTY_BY_KEY, Cell, build_stages, column_offset, measure_cell,
    measure_palette, pick_palette, positional_split, sample_at, sample_starts,
)
from sylvan.models.command_wm import CommandWorldModel                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHI = os.path.join(ROOT, "tools", "archi_hud", "architecture.json")
# Substrat SERVI par les harnais forêt (scripts/gates_foret_v2.sh, collect_foret_v1.sh).
# ⚠️ 2026-08-02 : pointait `wm_objcentric_kin` (obs 277, ancien monde) — un WM que le monde
# servi ne peut plus alimenter, et que cet outil ne pouvait de toute façon pas charger.
LIVE_WM = "data/checkpoints/wm_foret_v2_slot/wm_best.pt"
DROP_ALERT = 0.15          # chute entre deux étages voisins au-delà de laquelle on pointe le module
MLP_GAP_ALERT = 0.10       # sur ce projet le MLP n'a jamais battu franchement le linéaire : suspect


def declared_wm() -> str | None:
    """Ce que la CARTE VIVANTE déclare pour le world-model. On ne rouvre PAS un troisième endroit où
    lire l'état du projet (spec §5) : on lit la carte, et on CRIE si elle diverge du checkpoint sondé."""
    try:
        with open(ARCHI, encoding="utf-8") as fh:
            for m in json.load(fh)["modules"]:
                if m["id"] == "world_model":
                    return m.get("code")
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    return None


def load_wm(path: str) -> CommandWorldModel:
    """Charge le WM DÉCRIT PAR SON CHECKPOINT — jamais une architecture supposée.

    ⚠️ CORRIGÉ LE 2026-08-02. Cette fonction construisait le modèle à la main sans passer
    `retina_attention` et en FORÇANT `with_slot=True`. Deux pannes silencieuses en découlaient :
      1. un WM à encodeur d'ATTENTION (toute la génération forêt) ne se chargeait pas du tout —
         l'outil phare du projet était donc aveugle au substrat réellement servi ;
      2. sur un WM SANS canal-slot, le forçage créait un slot_encoder à poids ALÉATOIRES, et les
         colonnes « slot » et « token planner » rendaient du BRUIT présenté comme une mesure.
    `from_checkpoint` lit l'architecture dans le meta ; l'absence de slot est désormais dite, pas
    compensée (cf. `build_stages`, qui omet les colonnes correspondantes)."""
    pl = torch.load(path, map_location="cpu", weights_only=False)
    wm = CommandWorldModel.from_checkpoint(pl)
    wm.eval()                                   # WM GELÉ : on mesure ce qu'il contient déjà
    return wm


def fmt(cell: Cell | None, kind: str) -> str:
    """Une case = sonde LINÉAIRE puis sonde MLP (largeur 13, alignée sur l'en-tête)."""
    if cell is None:
        return f"{'—':^13}"
    if kind == "cat":
        return f"{100 * cell.lin:5.1f} {100 * cell.mlp:5.1f}%"
    return f"{cell.lin:+6.3f}{cell.mlp:+7.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", nargs="+", default=["data/replay_buffer/critic_bosq_ripe11"])
    ap.add_argument("--wm", default=LIVE_WM)
    ap.add_argument("--depths", nargs="+", type=int, default=[0, 20, 79],
                    help="profondeurs de rêve sondées (0 = perception, la suite = dégradation)")
    ap.add_argument("--rows", nargs="+", default=None, help=f"sous-ensemble : {[p.key for p in PROPERTIES]}")
    ap.add_argument("--stride", type=int, default=24, help="1 état retenu sur N (ticks voisins redondants)")
    ap.add_argument("--mlp-steps", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--frac-train", type=float, default=0.7)
    ap.add_argument("--meal-horizon", type=int, default=200,
                    help="K de la cible « repas dans les K ticks » (défaut = celui de "
                         "diag_critic_beyond_geometry, pour rester comparable)")
    ap.add_argument("--palette", choices=["auto", "teinte", "luminosite"], default="auto")
    ap.add_argument("--target", choices=["predit", "percu"], default="predit",
                    help="predit = le latent à la profondeur d est jugé sur ce qui est VRAI à t+d "
                         "(question JEPA) ; percu = tout est jugé sur t (question de la mémoire)")
    ap.add_argument("--split", choices=["episode", "positional"], default="episode",
                    help="episode = honnête (défaut) ; positional = convention des mesures "
                         "historiques, elle FUIT — pour comparer des chiffres à des chiffres")
    ap.add_argument("--no-guards", action="store_true", help="sauter la vérification du corpus")
    ap.add_argument("--json", default=None, help="écrit la matrice en JSON (suivi de non-régression)")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)                    # déterminisme (et le WM tourne en CPU de toute façon)
    t0 = time.time()

    # ---- GARDES (diagnostics/guards.py) : jamais de verdict sur un corpus dégénéré -------------
    print(scaffold_banner())
    if not args.no_guards:
        for c in args.corpus:
            s = sanity(c)
            flag = "✅" if not s["anomalies"] else "🚨"
            print(f"[guards] {flag} {c} : {s['ticks']} ticks | {s['consommations']} conso | "
                  f"immobile {100 * s['immobile_frac']:.0f} %")
            for a in s["anomalies"]:
                print(f"           - {a}")
            if s["anomalies"]:
                raise SystemExit("corpus dégénéré — aucune matrice ne serait interprétable")

    # ---- CARTE VIVANTE (tools/archi_hud) : le checkpoint sondé est-il celui que la carte déclare ?
    decl = declared_wm()
    if decl and decl != args.wm:
        print(f"[carte] ⚠️  la carte d'archi déclare world_model = {decl}\n"
              f"        la matrice sonde              {args.wm}  (substrat SERVI, cf CLAUDE.md)")

    obs, energy, cmds, bounds = load_bc_corpora(args.corpus)
    wm = load_wm(args.wm)
    horizon = max(2, max(args.depths) + 1)
    starts, tr = sample_starts(bounds, args.stride, horizon, args.frac_train)
    if int(tr.sum()) < 50 or int((~tr).sum()) < 50:
        raise SystemExit(f"échantillon trop maigre (train {int(tr.sum())}, held-out {int((~tr).sum())})")

    stages = build_stages(wm, obs, cmds, starts, energy, args.depths)
    n_ep = len(bounds) - 1
    print(f"\ncorpus {' + '.join(os.path.basename(c) for c in args.corpus)} | {len(energy)} ticks | "
          f"{n_ep} épisodes | WM {args.wm}")
    if args.split == "episode":
        split_label = (f"split PAR ÉPISODE : train {int(tr.sum())} / held-out {int((~tr).sum())}")
    else:
        split_label = (f"split POSITIONNEL ⚠️ (fuite assumée, convention historique) : "
                       f"{100 * args.frac_train:.0f} % premières lignes de CHAQUE ligne, en train")
    tgt = ("vérité-terrain à t+d pour le latent rêvé à la profondeur d (question JEPA : le rêve "
           "prédit-il ?)" if args.target == "predit"
           else "vérité-terrain à t pour TOUTES les colonnes (question mémoire : ça se souvient ?)")
    print(f"états sondés {len(starts)} (stride {args.stride}) | {split_label}\ncible : {tgt}")

    # ---- Constantes MESURÉES vs DÉCLARÉES, appliqué à la palette du monde ----------------------
    retina0 = obs[starts][:, wm.proprio_dim:wm.proprio_dim + 144]
    if args.palette == "auto":
        pname, palette, stat = pick_palette(retina0)
    else:
        pname, palette = args.palette, PALETTES[args.palette]
        stat = measure_palette(retina0, palette)
    warn = "  ⚠️ des rayons-bouffe qu'aucun type n'explique" if stat["hors_palette"] > 0.10 else ""
    print(f"palette MESURÉE sur le corpus : « {pname} » | écart médian aux couleurs déclarées "
          f"{stat['ecart_median']:.3f} | hors palette {100 * stat['hors_palette']:.1f} %{warn}")

    unknown = set(args.rows or []) - set(PROPERTY_BY_KEY)
    if unknown:
        raise SystemExit(f"propriété inconnue {sorted(unknown)} ; connues : {list(PROPERTY_BY_KEY)}")
    rows = [PROPERTY_BY_KEY[k] for k in args.rows] if args.rows else PROPERTIES
    cols = stages.names
    matrix: dict[str, dict[str, Cell]] = {}

    # Vérité-terrain à chaque décalage utile. La VALIDITÉ est l'INTERSECTION sur tous les décalages :
    # sans ça, chaque colonne serait notée sur un jeu d'états différent et la matrice ne serait plus
    # comparable colonne à colonne — or c'est exactement ce qu'on lui demande.
    offsets = sorted({column_offset(c, args.target) for c in stages.names})
    # Cible du critique : « un repas dans les K ticks à venir », bornée à l'épisode. L'étiquetage est
    # repris de sylvan.critic_corpus — c'est LA convention du projet, on ne la ré-invente pas ici.
    meal = residual_label(meal_flags(energy, bounds), bounds, args.meal_horizon)
    samples = {d: sample_at(wm, obs, energy, starts, d, palette, meal) for d in offsets}
    print(f"cible « repas » : {100 * float(meal[starts].mean()):.1f} % d'états positifs "
          f"(horizon {args.meal_horizon} ticks)")

    for prop in rows:
        truth = {d: prop.extract(samples[d]) for d in offsets}
        valid = torch.ones(len(starts), dtype=torch.bool)
        for _, v in truth.values():
            valid &= v
        n_ok = int(valid.sum())
        if n_ok < 100:
            print(f"  · {prop.label} : IGNORÉE — {n_ok} états valides à TOUTES les profondeurs")
            continue
        y = truth[0][0][valid]
        y = y.long() if prop.kind == "cat" else y.float()
        if prop.kind == "cat":
            share = float(torch.bincount(y, minlength=prop.n_classes).max()) / n_ok
            if share > 0.95:
                # Le monde de CE corpus ne rend pas la propriété (une seule apparence servie) : la
                # sonder donnerait 100 % partout et ferait croire que tout va bien. Un « n/a » franc
                # vaut mieux qu'une ligne verte qui ne mesure rien.
                print(f"  · {prop.label} : n/a — une seule apparence servie ({100 * share:.0f} % "
                      f"d'une même classe), ce monde ne porte pas cette propriété")
                continue
        vtr = tr[valid] if args.split == "episode" else positional_split(n_ok, args.frac_train)
        matrix[prop.key] = {}
        cov = 100 * n_ok / len(valid)
        extra = ""
        if prop.kind == "cat":
            # Majorité POOLÉE en plus de la baseline held-out : c'est le repère historique du projet.
            counts = torch.bincount(y, minlength=prop.n_classes)
            extra = (f" | classes {[int(c) for c in counts]} | majorité poolée "
                     f"{100 * float(counts.max() / counts.sum()):.1f} %")
            # DÉCALAGE DE DISTRIBUTION train/held-out. Une catégorie peut être CONFONDUE avec la vie
            # (les types sont re-tirés à la repousse) : le split honnête par épisode fait alors face
            # à d'autres classes qu'à l'entraînement, et une précision basse ne dit plus « l'info est
            # absente » mais « les classes ont changé ». Le taire serait la faute du §2.
            ptr = torch.bincount(y[vtr], minlength=prop.n_classes).float()
            pte = torch.bincount(y[~vtr], minlength=prop.n_classes).float()
            tv = 0.5 * float((ptr / ptr.sum().clamp_min(1) - pte / pte.sum().clamp_min(1)).abs().sum())
            if tv > 0.20:
                extra += (f"\n      ⚠️ classes DÉCALÉES entre train et held-out (distance totale "
                          f"{tv:.2f}) — la précision sous-estime ce que porte la représentation")
        print(f"  · {prop.label} : {n_ok} états valides ({cov:.0f} % du corpus sondé){extra}", flush=True)
        for col in cols:
            x = stages.reps[col][valid].float()
            yc = truth[column_offset(col, args.target)][0][valid]
            yc = yc.long() if prop.kind == "cat" else yc.float()
            matrix[prop.key][col] = measure_cell(prop, x, yc, vtr, args.mlp_steps, args.hidden)

    # ---- LE TABLEAU ---------------------------------------------------------------------------
    w = 32
    head = f"{'propriété du monde':<{w}} {'baseline':>9} " + " ".join(f"{c:^13}" for c in cols)
    print("\n" + "=" * len(head))
    print("MATRICE DE SURVIE DE L'INFORMATION      (chaque case : sonde LINÉAIRE puis sonde MLP)")
    print("  catégorie -> précision held-out %   ·   continu -> R² held-out")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for prop in rows:
        if prop.key not in matrix:
            continue
        cells = matrix[prop.key]
        base = cells[cols[0]].baseline
        b = f"{100 * base:8.1f}%" if prop.kind == "cat" else f"{base:+9.3f}"
        print(f"{prop.label:<{w}} {b:>9} " + " ".join(fmt(cells[c], prop.kind) for c in cols))
    print("=" * len(head))

    # ---- LECTURE : la chute, et le module à accuser --------------------------------------------
    # Les chutes sont lues le long des ARÊTES RÉELLES du pipeline, pas de gauche à droite : le slot
    # est une branche séparée qui part de la rétine, pas la suite du latent. Chaîner les colonnes
    # accuserait le mauvais module — précisément ce que l'outil existe pour éviter.
    print("\nOÙ L'INFORMATION MEURT (chute le long des arêtes RÉELLES du pipeline)")
    for prop in rows:
        if prop.key not in matrix:
            continue
        cells = matrix[prop.key]
        drops = sorted(((cells[p].best - cells[c].best, p, c) for p, c in stages.edges),
                       key=lambda t: -t[0])
        shown = [d for d in drops if d[0] >= DROP_ALERT] or drops[:1]
        print(f"  {prop.label}")
        for d, src, dst in shown:
            size = f"{100 * d:5.1f} pt" if prop.kind == "cat" else f"{d:6.3f}"
            mark = "  ⚠️ information DÉTRUITE ici" if d >= DROP_ALERT else ""
            print(f"      {src:>13} -> {dst:<13} : −{size}{mark}")
        # L'écart MLP − linéaire n'est interprétable que sur les représentations APPRISES : sur une
        # colonne à 2 ou 5 dimensions (slot, token) une non-linéarité est attendue, pas suspecte.
        gaps = [(cells[c].mlp - cells[c].lin, c) for c in cols
                if c == "encodeur" or c.startswith("latent")]
        if gaps:
            g, gc = max(gaps, key=lambda t: t[0])
            if g >= MLP_GAP_ALERT:
                print(f"      ⚠️ MLP − linéaire = {g:+.3f} en « {gc} » : info PRÉSENTE mais mal "
                      f"exposée\n         (sur ce projet le MLP ne bat jamais franchement le "
                      f"linéaire — un écart soudain se vérifie)")
    for prop in rows:                                  # métriques dérivées (ex. la distance du slot)
        if prop.key in matrix and any(matrix[prop.key][c].extra for c in cols):
            print(f"\n  détail « {prop.label} » (sonde linéaire) :")
            keys = list(next(matrix[prop.key][c].extra for c in cols if matrix[prop.key][c].extra))
            for k in keys:
                print(f"    {k:<20} " + " ".join(f"{matrix[prop.key][c].extra.get(k, float('nan')):+13.3f}"
                                                 for c in cols))

    print(f"\n  LECTURE : une case ≈ baseline -> l'information n'est PLUS là. Une CHUTE entre deux "
          f"colonnes\n            désigne le module qui la détruit — c'est la valeur de l'outil.")
    print(f"  ({time.time() - t0:.0f} s, WM gelé, aucun entraînement du substrat)")

    if args.json:
        out = {"corpus": args.corpus, "wm": args.wm, "depths": args.depths, "stride": args.stride,
               "palette": pname, "split": args.split, "target": args.target,
               "n_states": len(starts), "columns": cols,
               "edges": stages.edges,
               "rows": {k: {c: {"lin": v[c].lin, "mlp": v[c].mlp, "baseline": v[c].baseline,
                                "n_test": v[c].n_test, **{f"extra.{ek}": ev for ek, ev in v[c].extra.items()}}
                            for c in cols} for k, v in matrix.items()}}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"  matrice écrite dans {args.json} (comparer après retrain = non-régression)")


if __name__ == "__main__":
    main()
