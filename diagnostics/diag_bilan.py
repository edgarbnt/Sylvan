"""BILAN — une commande qui dit où en est le système, sans rien supposer.

POURQUOI CET OUTIL EXISTE. Le 2026-08-02, un audit a montré que trois sources d'état
(CLAUDE.md, la carte d'archi, les docstrings) étaient chacune en retard d'une génération sur
le code, et que 40+ diagnostics ne pouvaient même pas CHARGER le WM servi. Résultat concret :
une session entière passée à mesurer le mauvais checkpoint sur 3,5 % du corpus disponible,
sans voir que l'entité mourait à 12 % de son budget après un seul repas.

Cet outil ne remplace aucun gate. Il répond à « où suis-je ? » en quatre questions, en LISANT
le disque, jamais la doc :

  1. SUBSTRAT   — quel WM, quelle architecture, se charge-t-il, qu'annonce sa méta ?
  2. PERCEPTION — la position lue est-elle JUSTE ? et le latent la porte-t-il ?
  3. VIE        — que fait réellement l'entité (survie, repas, causes de mort) ?
  4. PURETÉ     — quels échafaudages et quelles clés-apparence sont actifs ?

⚠️ SUR LA VÉRITÉ-TERRAIN. La section PERCEPTION compare la position PERÇUE à `food_rel0`,
l'état du monde écrit par Godot dans le corpus. C'est un ORACLE DE MESURE, jamais un oracle
de boucle : il sert à SAVOIR si la perception est juste, exactement comme `diag_slot_localise`.
Le distinguo qui compte (§3 CLAUDE.md) est *à l'inférence*, pas *à la mesure* — une perception
qu'on ne mesure jamais contre le monde est une perception qu'on croit sur parole.

⚠️ CE QUE CET OUTIL NE DIT PAS. Il ne rejoue rien : il lit un corpus DÉJÀ collecté. Un bilan
brillant sur un vieux corpus ne dit rien du comportement d'aujourd'hui — la date du corpus est
affichée pour cette raison.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_bilan.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_bilan.py \
        --wm data/checkpoints/wm_foret_v2_slot/wm_best.pt \
        --perception data/replay_buffer/gate_foret_cl \
        --vie data/replay_buffer/foret_v1_planner data/replay_buffer/foret_v1b_planner
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import statistics as st
import sys
import warnings

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sylvan.models.command_wm import CommandWorldModel  # noqa: E402

# Budget de vie du monde servi — sert à dire « 12 % du budget », pas seulement « 380 ticks ».
DEFAULT_EPISODE_STEPS = 3000
# Barre de perception : au-delà, le planner vise à côté de la bouche (eat_radius = 1,0 m).
PERCEPTION_BAR_M = 1.0


# --------------------------------------------------------------------------------------------- #
# 1. SUBSTRAT
# --------------------------------------------------------------------------------------------- #

def section_substrat(wm_path: str) -> tuple[CommandWorldModel | None, dict]:
    print("\n" + "=" * 78)
    print("1. SUBSTRAT — le world-model")
    print("=" * 78)
    if not os.path.exists(wm_path):
        print(f"  ❌ INTROUVABLE : {wm_path}")
        return None, {}

    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(wm_path)).strftime("%Y-%m-%d %H:%M")
    payload = torch.load(wm_path, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    print(f"  checkpoint : {wm_path}")
    print(f"  entraîné   : {mtime}")

    enc = "attention par rayon" if meta.get("retina_attention") else "MLP dense"
    slot = f"oui ({meta.get('slot_resources')} ressources)" if meta.get("with_slot") else "NON"
    print(f"  encodeur   : {enc}")
    print(f"  canal-slot : {slot}")
    print(f"  dims       : obs={meta.get('obs_dim')} proprio={meta.get('proprio_dim')}")

    # Ce que le checkpoint AVOUE de lui-même — la méta est la seule source non-narrative.
    if meta.get("gates_failed"):
        print(f"  🚨 GATES ÉCHOUÉS déclarés dans la méta : {meta['gates_failed']}")
    if meta.get("slot_note"):
        print(f"  ⚠️  note slot : {meta['slot_note']}")
    if meta.get("queries_cos_to_hand"):
        cos = meta["queries_cos_to_hand"]
        print(f"  ⚠️  requêtes « apprises » vs codées-main : cos={cos}")
        if min(cos) > 0.99:
            print("      → elles ont CONVERGÉ sur les primaires codées-main : l'apprentissage n'a rien acheté.")
    if meta.get("with_position_head"):
        print("  🚨 position_head PRÉSENTE — tête L2-supervisée sur food_rel0 (checkpoint CONTAMINÉ).")

    try:
        wm = CommandWorldModel.from_checkpoint(payload)
        wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)
        print("  ✅ se charge")
    except Exception as exc:                                     # noqa: BLE001 — on veut le message
        print(f"  ❌ NE SE CHARGE PAS : {type(exc).__name__}: {str(exc)[:150]}")
        return None, meta

    # Les deux calages du slot (seuils par-requête + angles du cône) sont faits par
    # `from_checkpoint._calibrate_slot`. On les AFFICHE ici pour que le lecteur voie sur quoi
    # la mesure repose — un calage invisible est un calage qu'on oubliera de refaire ailleurs.
    if getattr(wm, "with_slot", False):
        se = wm.slot_encoder
        if getattr(se, "query_thr", None) is not None:
            print(f"  ↳ seuils par-requête servis : {[round(float(v), 3) for v in se.query_thr]}")
        fov = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360"))
        print(f"  ↳ angles de rayon calés sur un champ de {fov:.0f}°")
    return wm, meta


# --------------------------------------------------------------------------------------------- #
# 2. PERCEPTION
# --------------------------------------------------------------------------------------------- #

def load_perception_pairs(corpus: str, half_fov: float, limit: int = 0):
    """(obs, rétine, position VRAIE, id d'épisode) sur les ticks où la nourriture est en vue.

    L'ID D'ÉPISODE N'EST PAS DÉCORATIF : sans lui on ne peut pas découper honnêtement. À
    0,05 m/tick, deux ticks voisins sont quasi identiques ; un split ALÉATOIRE met donc le
    jumeau de chaque tick de test dans le train, et la sonde MÉMORISE au lieu de généraliser.
    Mesuré le 2026-08-02 sur ce corpus : sonde latent 0,42 m en split aléatoire contre 2,58 m
    en split par épisode — un facteur 6, et un verdict inversé (le latent semblait battre le
    slot, il est en réalité moins bon)."""
    obs_l, ret_l, tgt_l, epi_l = [], [], [], []
    eid = 0
    for f in sorted(glob.glob(os.path.join(corpus, "episode_*.jsonl"))):
        got = False
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("wm", {}).get("food_rel0")
            if not v or len(v) < 3 or v[2] <= 0.5:
                continue
            if abs(math.degrees(math.atan2(v[0], v[1]))) > half_fov:
                continue
            o = r["obs"]
            ret = r.get("wm", {}).get("retina0") or o.get("retina")
            if not ret or len(ret) != 144:
                continue
            obs_l.append(o["proprio"] + ret + [o["energy"] / 100.0])
            ret_l.append(ret)
            tgt_l.append([v[0], v[1]])
            epi_l.append(eid)
            got = True
            if limit and len(obs_l) >= limit:
                break
        if got:
            eid += 1
        if limit and len(obs_l) >= limit:
            break
    if not obs_l:
        return None, None, None, None
    return (torch.tensor(obs_l, dtype=torch.float32),
            torch.tensor(ret_l, dtype=torch.float32),
            torch.tensor(tgt_l, dtype=torch.float32),
            torch.tensor(epi_l))


def _bands(err: torch.Tensor, dist: torch.Tensor) -> None:
    for lo, hi, lab in [(0, 2, "<2m"), (2, 5, "2-5m"), (5, 99, ">5m")]:
        m = (dist >= lo) & (dist < hi)
        if int(m.sum()) < 5:
            continue
        e = err[m]
        print(f"      {lab:6} n={int(m.sum()):4d}  méd={e.median():.2f} m  "
              f"<1m={(e < 1.0).float().mean():.0%}")


@torch.no_grad()
def section_perception(wm: CommandWorldModel, corpus: str, half_fov: float,
                       probe: bool, limit: int) -> dict:
    print("\n" + "=" * 78)
    print("2. PERCEPTION — la position lue est-elle JUSTE ?")
    print("=" * 78)
    print(f"  corpus : {corpus}   (vérité = food_rel0, oracle de MESURE)")

    obs, retina, truth, epi = load_perception_pairs(corpus, half_fov, limit)
    if obs is None:
        print("  ⚠️  aucun tick avec nourriture visible dans le champ — rien à mesurer.")
        return {}
    dist = truth.norm(dim=1)
    print(f"  ticks  : {obs.shape[0]} · {int(epi.max()) + 1} épisodes   "
          f"distance vraie médiane {dist.median():.2f} m")

    out: dict = {}

    # (a) Ce que le SLOT rend — c'est ce que le planner consomme aujourd'hui.
    if getattr(wm, "with_slot", False):
        pos = wm.slot_encoder.positions(retina)[:, 0, :]
        err = (pos - truth).norm(dim=1)
        out["slot"] = float(err.median())
        verdict = "✅" if err.median() < PERCEPTION_BAR_M else "❌"
        print(f"\n  {verdict} SLOT (ce que le planner lit)   : méd={err.median():.2f} m  "
              f"<1m={(err < 1.0).float().mean():.0%}")
        _bands(err, dist)
    else:
        print("\n  — pas de canal-slot sur ce WM.")

    # (b) Ce que le LATENT PORTE — sépare « l'info est absente » de « l'info est mal lue ».
    #     ⚠️ SPLIT PAR ÉPISODE, jamais aléatoire (cf. `load_perception_pairs`).
    if probe:
        n_ep = int(epi.max()) + 1
        if n_ep < 4:
            print("\n  ⚠️  moins de 4 épisodes : split honnête impossible, sondes latent sautées.")
            return out
        lat = torch.cat([wm.encoder(obs[i:i + 2048]) for i in range(0, len(obs), 2048)])
        cut = int(0.8 * n_ep)
        tr, te = epi < cut, epi >= cut
        mu, sd = lat[tr].mean(0, keepdim=True), lat[tr].std(0, keepdim=True).clamp(min=1e-2)
        x = (lat - mu) / sd

        # Ridge fermée : déterministe, pas d'optimiseur, pas de graine à discuter.
        xb = torch.cat([x[tr], torch.ones(int(tr.sum()), 1)], dim=1)
        w = torch.linalg.lstsq(xb.T @ xb + 1e-2 * torch.eye(xb.shape[1]),
                               xb.T @ truth[tr]).solution
        pred_lin = torch.cat([x[te], torch.ones(int(te.sum()), 1)], dim=1) @ w
        err_lin = (pred_lin - truth[te]).norm(dim=1)
        out["latent_lineaire"] = float(err_lin.median())

        with torch.enable_grad():
            torch.manual_seed(0)
            mlp = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.SiLU(),
                                      torch.nn.Linear(64, 32), torch.nn.SiLU(),
                                      torch.nn.Linear(32, 2))
            opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)
            for _ in range(600):
                opt.zero_grad()
                ((mlp(x[tr]) - truth[tr]) ** 2).mean().backward()
                opt.step()
        err_mlp = (mlp(x[te]) - truth[te]).norm(dim=1)
        out["latent_mlp"] = float(err_mlp.median())

        print(f"\n  LATENT → position (held-out PAR ÉPISODE — {cut} train / {n_ep - cut} test) :")
        print(f"      sonde LINÉAIRE : méd={err_lin.median():.2f} m")
        print(f"      sonde MLP      : méd={err_mlp.median():.2f} m")
        if out["latent_mlp"] > out.get("slot", 0.0) > 0:
            print(f"      → le latent est MOINS bon que le slot ({out['slot']:.2f} m) : lire la position")
            print("        depuis le latent n'est PAS la voie (mesuré, split honnête).")

    # (c) GISEMENT vs DISTANCE — c'est le gisement qui décide, pas la distance.
    #     Le coût du planner est −min_dist sur le slot TRANSPORTÉ : une erreur de distance
    #     décale le score de TOUS les candidats pareil (argmax inchangé) ; une erreur de
    #     gisement change QUEL candidat se rapproche.
    if getattr(wm, "with_slot", False):
        pos = wm.slot_encoder.positions(retina)[:, 0, :]
        db = torch.atan2(pos[:, 0], pos[:, 1]) - torch.atan2(truth[:, 0], truth[:, 1])
        db = torch.atan2(torch.sin(db), torch.cos(db)).abs() * 180.0 / math.pi
        dd = (pos.norm(dim=1) - dist).abs()
        out["bearing_deg"] = float(db.median())
        out["dist_err"] = float(dd.median())
        print(f"\n  DÉCOMPOSITION du slot — c'est le gisement qui décide :")
        print(f"      |gisement| : méd={db.median():.1f}°   (q75={db.quantile(0.75):.1f}°)")
        print(f"      |distance| : méd={dd.median():.2f} m")
        lat_err = float(dist.median()) * math.sin(math.radians(float(db.median())))
        print(f"      → à {dist.median():.1f} m, {db.median():.0f}° font {lat_err:.2f} m de côté "
              f"(bouche : {PERCEPTION_BAR_M:.1f} m)")
    return out


# --------------------------------------------------------------------------------------------- #
# 3. VIE
# --------------------------------------------------------------------------------------------- #

def section_vie(corpora: list[str], budget: int) -> dict:
    print("\n" + "=" * 78)
    print("3. VIE — que fait réellement l'entité ?")
    print("=" * 78)
    surv, meals, drinks, causes = [], [], [], []
    dates = []
    for corpus in corpora:
        files = sorted(glob.glob(os.path.join(corpus, "episode_*.jsonl")))
        if not files:
            print(f"  ⚠️  {corpus} : aucun épisode")
            continue
        dates.append(datetime.datetime.fromtimestamp(
            os.path.getmtime(files[0])).strftime("%Y-%m-%d"))
        for f in files:
            rows = [json.loads(l) for l in open(f) if l.strip()]
            if not rows:
                continue
            surv.append(len(rows))
            meals.append(sum(1 for r in rows if r.get("wm", {}).get("ate", 0) > 0.5))
            d = sum(1 for i in range(1, len(rows))
                    if rows[i]["obs"].get("thirst", 0) - rows[i - 1]["obs"].get("thirst", 0) > 20)
            drinks.append(d)
            last = rows[-1]["obs"]
            e, t, h = last.get("energy", 0), last.get("thirst", 0), last.get("health", 100)
            if len(rows) >= budget - 10:
                causes.append("PLEIN")
            elif min(e, t, h) <= 10:
                causes.append(min((e, "faim"), (t, "soif"), (h, "sante"))[1])
            else:
                causes.append("autre")
    if not surv:
        print("  ⚠️  aucun corpus de vie exploitable.")
        return {}

    from collections import Counter
    pct = 100 * st.median(surv) / budget
    print(f"  corpus : {', '.join(os.path.basename(c) for c in corpora)}  "
          f"({len(surv)} vies, collecté {'/'.join(sorted(set(dates)))})")
    print(f"\n  survie médiane : {st.median(surv):.0f} / {budget} ticks  =  {pct:.0f} % du budget")
    print(f"  repas / vie    : méd={st.median(meals):.1f}  moy={st.mean(meals):.2f}")
    print(f"  boissons / vie : méd={st.median(drinks):.1f}  moy={st.mean(drinks):.2f}")
    print(f"  causes de mort : {dict(Counter(causes))}")
    if pct < 50:
        print(f"\n  ❌ l'entité meurt à {pct:.0f} % de son budget — le forageur ne tient pas la vie.")
    return {"survie_med": st.median(surv), "repas_med": st.median(meals), "pct_budget": pct}


# --------------------------------------------------------------------------------------------- #
# 4. PURETÉ
# --------------------------------------------------------------------------------------------- #

def section_purete(wm: CommandWorldModel | None) -> None:
    print("\n" + "=" * 78)
    print("4. PURETÉ — qu'est-ce qui est encore codé-main dans la boucle ?")
    print("=" * 78)
    try:
        from guards import scaffold_banner
        print(scaffold_banner())
    except Exception as exc:                                     # noqa: BLE001
        print(f"  (bannière d'échafaudages indisponible : {type(exc).__name__})")

    if wm is not None and getattr(wm, "with_slot", False):
        se = wm.slot_encoder
        if getattr(se, "color_queries", None) is not None:
            q = se.color_queries
            thr = getattr(se, "query_thr", None)
            print("\n  🔴 CLÉ-APPARENCE ACTIVE — requêtes-couleur du slot (slot_head.py) :")
            for k in range(q.shape[0]):
                t = f"{float(thr[k]):.3f}" if thr is not None else "?"
                print(f"      slot {k} : requête=({q[k, 0]:.2f}, {q[k, 1]:.2f}, {q[k, 2]:.2f})  seuil={t}")
            print("      → « rouge=bouffe, bleu=eau, vert=danger » est CÂBLÉ : ajouter une ressource")
            print("        d'une nouvelle couleur demande de toucher au code, pas de vivre.")


# --------------------------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2_slot/wm_best.pt")
    ap.add_argument("--perception", default="data/replay_buffer/gate_foret_cl",
                    help="corpus pour mesurer la justesse de la position")
    ap.add_argument("--vie", nargs="+",
                    default=["data/replay_buffer/foret_v1_planner",
                             "data/replay_buffer/foret_v1b_planner",
                             "data/replay_buffer/foret_v1c_planner"],
                    help="corpus de comportement")
    ap.add_argument("--fov", type=float,
                    default=float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")))
    ap.add_argument("--budget", type=int, default=DEFAULT_EPISODE_STEPS)
    ap.add_argument("--limit", type=int, default=6000, help="plafond de ticks pour la perception")
    ap.add_argument("--no-probe", action="store_true", help="sauter les sondes latent (plus rapide)")
    a = ap.parse_args()

    torch.set_num_threads(4)
    print("=" * 78)
    print(f"BILAN SYLVAN — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}   FOV={a.fov:.0f}°")
    print("=" * 78)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm, meta = section_substrat(a.wm)

    perc = {}
    if wm is not None:
        perc = section_perception(wm, a.perception, a.fov / 2.0, not a.no_probe, a.limit)
    vie = section_vie(a.vie, a.budget)
    section_purete(wm)

    # --- Verdict : nommer le goulot, une seule phrase ---
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    slot_err = perc.get("slot")
    lat_mlp = perc.get("latent_mlp")
    brg = perc.get("bearing_deg")
    de = perc.get("dist_err")
    if slot_err is not None and slot_err > PERCEPTION_BAR_M:
        print(f"  Le GOULOT est la PERCEPTION : le slot rend {slot_err:.2f} m "
              f"(barre {PERCEPTION_BAR_M:.1f} m = le rayon de la bouche).")
        if brg is not None and de is not None:
            print(f"  Et c'est le GISEMENT, pas la distance : {brg:.0f}° contre {de:.2f} m "
                  f"d'erreur de distance.")
            print("  → viser mieux ne demande pas de mieux estimer la distance, mais de mieux")
            print("    SÉLECTIONNER les rayons qui portent l'objet.")
        if lat_mlp is not None and lat_mlp > slot_err:
            print(f"  Le latent ne peut pas remplacer le slot : {lat_mlp:.2f} m contre "
                  f"{slot_err:.2f} m (split honnête).")
    elif slot_err is not None:
        print(f"  La perception tient ({slot_err:.2f} m). Le goulot est ailleurs.")
    if vie.get("pct_budget", 100) < 50:
        print(f"  Et le comportement le confirme : {vie['pct_budget']:.0f} % du budget de vie, "
              f"{vie['repas_med']:.0f} repas médians.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
