"""GARDE-FOUS DE MESURE — nés des 8 auto-corrections du 2026-07-21 (owner-demandés).

Le projet a déjà d'excellentes règles ÉCRITES (CLAUDE.md §1/§2) et elles n'ont PAS empêché ces
erreurs, parce qu'elles étaient toutes dans la PLOMBERIE DE MESURE, pas dans le raisonnement.
D'où ce module : des vérifications AUTOMATIQUES, à appeler avant tout verdict.

Les trois gardes, et l'erreur RÉELLE que chacune aurait attrapée :

1. `check_constants` — MESURE les constantes physiques sur le corpus et les compare aux valeurs
   DÉCLARÉES dans le code. Aurait attrapé : `SPEED_M_PER_TICK = 0.02` alors que le corps fait
   0.0100 m/tick (constante fausse d'un facteur 2, qui a fabriqué un faux « 1,88× d'inefficacité
   de trajet » ET rendu la portée métabolique 2× optimiste → des morts étiquetées « arbitrage » à
   tort) ; et le restore NOMINAL 60 alors que l'ABSORBÉ réel est 42 (plafond à 100).

2. `scaffold_banner` — liste les flags SYLVAN_* actifs et SIGNALE ceux déclarés « échafaudage ».
   Aurait attrapé : `SYLVAN_PLANNER_FAR_ALIGN=1`, déclaré RETIRABLE dans le code depuis
   2026-07-06, allumé par défaut dans TOUS les harnais, jamais re-testé après le pivot du corps —
   il handicapait l'entité dans toutes les mesures de la semaine (arène ouverte).

3. `sanity` — refuse de laisser rendre un verdict sur un corpus DÉGÉNÉRÉ (entité immobile, zéro
   consommation, corpus trop court). Aurait attrapé : le run FA=0 en monde-mur où l'entité est
   restée IMMOBILE 79 % des ticks — j'ai failli rapporter « la mémoire s'effondre » alors que le
   corps ne bougeait pas.

Usage dans un diag :
    from guards import check_constants, sanity
    assert not sanity(run)["anomalies"], sanity(run)["anomalies"]
    check_constants(run, {"speed": SPEED_M_PER_TICK, "drain_e": DRAIN_PER_TICK}, strict=True)

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/guards.py <corpus> [--selfcheck]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import statistics as st

CONSUME_JUMP = 5.0
TELEPORT_M = 1.0          # au-delà = respawn, pas du mouvement
IMMOBILE_MAX = 0.60       # > 60 % de ticks immobiles = corpus dégénéré (l'entité ne joue pas)
MIN_TICKS = 1000
TOL = 0.15                # 15 % d'écart toléré entre constante déclarée et mesurée

# Flags DÉCLARÉS échafaudage (carte d'archi + commentaires « RETIRABLE » dans le code).
# Un échafaudage actif n'est pas une faute — l'ignorer silencieusement en est une.
SCAFFOLDS = {
    "SYLVAN_PLANNER_FAR_ALIGN": "échafaudage far-target (RETIRABLE, code 2026-07-06) — MESURÉ "
                                "2026-07-21 : handicape en arène OUVERTE, mais PORTEUR en monde-mur",
    "SYLVAN_PLANNER_ALIGN_GAIN": "gain de l'échafaudage far-target",
    "SYLVAN_PLANNER_HEADING_W": "hint de cap A→B (échafaudage porteur daté)",
    "SYLVAN_WP_SALIENCY": "lunette apprise (pure) mais étage waypoint = échafaudage déclaré",
    "SYLVAN_WP_OBSTACLE": "lentille obstacle sur l'étage waypoint (échafaudage déclaré)",
}


def _open(p: str):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def _ticks(run: str) -> list[dict]:
    out = []
    for ep in sorted(glob.glob(os.path.join(run, "ep_*.jsonl*"))):
        with _open(ep) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def _reset_ticks(E: list[float], TH: list[float]) -> set[int]:
    """Indices des ticks qui sont un RESPAWN, pas une consommation.

    ⚠️ C'EST LA GARDE LA PLUS IMPORTANTE DU MODULE : compter les respawns comme des repas est
    l'erreur qui a fait enterrer la mémoire à tort le 2026-07-21 (25 des 39 « repas » d'un bras
    étaient des resets ; comme les deux bras en avaient autant, ce faux constant ÉCRASAIT l'écart
    relatif et fabriquait un faux négatif).
    Deux signatures, pour couvrir les deux mondes :
      - les DEUX jauges sautent au même tick (monde multi-drive : le respawn remet tout à l'init) ;
      - une jauge saute en venant d'un niveau MORTEL (<15) : en food-only seule l'énergie bouge,
        et un respawn vient TOUJOURS de la mort (un vrai repas in extremis est rare — biais
        conservateur assumé : on préfère rater un repas que compter un respawn).
    """
    resets = set()
    for i in range(1, len(E)):
        de, dt = E[i] - E[i - 1], TH[i] - TH[i - 1]
        if de > CONSUME_JUMP and dt > CONSUME_JUMP:
            resets.add(i)
        elif (de > CONSUME_JUMP and E[i - 1] < 15) or (dt > CONSUME_JUMP and TH[i - 1] < 15):
            resets.add(i)
    return resets


def measured_constants(run: str, ticks: list[dict] | None = None) -> dict:
    """Mesure les constantes physiques SUR LE CORPUS (jamais déclarées, toujours mesurées)."""
    T = _ticks(run) if ticks is None else ticks
    if not T:
        raise SystemExit(f"corpus vide : {run}")
    disp = []
    for a, b in zip(T, T[1:]):
        pa, pb = a["wm"].get("torso0"), b["wm"].get("torso0")
        if pa and pb:
            d = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            if d < TELEPORT_M:                       # exclut les respawns
                disp.append(d)
    has_pose = len(disp) > 0                          # sans torso on ne peut RIEN dire du mouvement
    moving = [d for d in disp if d > 1e-6]
    E = [float(t["obs"]["energy"]) for t in T]
    TH = [float(t["obs"]["thirst"]) for t in T]
    resets = _reset_ticks(E, TH)

    def drain(series: list[float]) -> float:
        drops = [a - b for a, b in zip(series, series[1:]) if 0 < a - b < 1.0]
        return st.median(drops) if drops else float("nan")

    def restore(series: list[float]) -> float:
        """Restore ABSORBÉ (pas nominal : le plafond à 100 écrête). Respawns EXCLUS."""
        j = [series[i] - series[i - 1] for i in range(1, len(series))
             if series[i] - series[i - 1] > CONSUME_JUMP and i not in resets]
        return st.median(j) if j else float("nan")

    return {
        "speed": st.median(moving) if moving else float("nan"),   # m/tick quand elle avance
        "drain_e": drain(E), "drain_t": drain(TH),
        "restore_e_absorbed": restore(E), "restore_t_absorbed": restore(TH),
        "immobile_frac": ((len(disp) - len(moving)) / len(disp)) if has_pose else float("nan"),
        "has_pose": has_pose, "resets": len(resets), "ticks": len(T),
    }


def check_constants(run: str, declared: dict, *, strict: bool = False) -> list[str]:
    """Compare les constantes DÉCLARÉES aux constantes MESURÉES. Retourne les écarts (vide = OK)."""
    m = measured_constants(run)
    bad = []
    for k, v in declared.items():
        got = m.get(k)
        if got is None or (isinstance(got, float) and math.isnan(got)):
            bad.append(f"{k} : impossible à mesurer sur ce corpus")
            continue
        if v and abs(got - v) / abs(v) > TOL:
            bad.append(f"{k} : DÉCLARÉ {v:g} vs MESURÉ {got:.4g} "
                       f"(écart {100*abs(got-v)/abs(v):.0f} %) — corriger la constante, pas la mesure")
    if strict and bad:
        raise AssertionError("Constantes fausses :\n  - " + "\n  - ".join(bad))
    return bad


def consumptions(run: str, ticks: list[dict] | None = None) -> int:
    """Consommations RÉELLES (respawns exclus) — l'implémentation canonique, à réutiliser."""
    T = _ticks(run) if ticks is None else ticks
    E = [float(t["obs"]["energy"]) for t in T]
    TH = [float(t["obs"]["thirst"]) for t in T]
    resets = _reset_ticks(E, TH)
    return sum(1 for i in range(1, len(E))
               if i not in resets and (E[i] - E[i - 1] > CONSUME_JUMP or TH[i] - TH[i - 1] > CONSUME_JUMP))


def sanity(run: str) -> dict:
    """Le corpus est-il exploitable ? Un corpus dégénéré ne doit JAMAIS fonder un verdict."""
    T = _ticks(run)
    m = measured_constants(run, ticks=T)              # un seul parse, pas deux
    conso = consumptions(run, ticks=T)
    anomalies = []
    if not m["has_pose"]:
        anomalies.append("aucune pose de corps (torso0) — mouvement NON vérifiable sur ce corpus")
    elif m["immobile_frac"] > IMMOBILE_MAX:
        anomalies.append(f"entité IMMOBILE {100*m['immobile_frac']:.0f} % des ticks "
                         f"(> {100*IMMOBILE_MAX:.0f} %) — le corps ne joue pas, rien à mesurer")
    if conso == 0:
        anomalies.append("ZÉRO consommation réelle — comportement dégénéré")
    if m["ticks"] < MIN_TICKS:
        anomalies.append(f"corpus trop court ({m['ticks']} ticks < {MIN_TICKS})")
    return {"anomalies": anomalies, "consommations": conso, **m}


def scaffold_banner(env: dict | None = None) -> str:
    """Liste les SYLVAN_* actifs et SIGNALE les échafaudages (un échafaudage silencieux = piège)."""
    env = dict(os.environ if env is None else env)
    active = {k: v for k, v in sorted(env.items()) if k.startswith("SYLVAN_")}
    lines = [f"[guards] {len(active)} flags SYLVAN_* actifs"]
    for k, why in SCAFFOLDS.items():
        v = active.get(k)
        if v is not None and v not in ("0", "", "0.0"):
            lines.append(f"  ⚠️  ÉCHAFAUDAGE ACTIF  {k}={v}  — {why}")
    if len(lines) == 1:
        lines.append("  ✅ aucun échafaudage déclaré actif")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        # Le VRAI test : les gardes attrapent-elles les erreurs RÉELLES du 2026-07-21 ?
        good = "data/replay_buffer/arbgrad_graded_s1_r40_fa0"
        broken = "data/replay_buffer/obmemM_off_s1_fa0"
        assert glob.glob(good + "/ep_*.jsonl*"), "corpus de test absent"
        bad = check_constants(good, {"speed": 0.02})          # l'erreur du jour
        assert bad, "garde 1 MUETTE sur la constante de vitesse fausse"
        assert not check_constants(good, {"speed": 0.0100}), "garde 1 crie sur une constante JUSTE"
        if glob.glob(broken + "/ep_*.jsonl*"):
            assert sanity(broken)["anomalies"], "garde 3 MUETTE sur le corpus immobile"
        assert not sanity(good)["anomalies"], "garde 3 crie sur un corpus sain"
        b = scaffold_banner({"SYLVAN_PLANNER_FAR_ALIGN": "1"})
        assert "ÉCHAFAUDAGE ACTIF" in b, "garde 2 MUETTE sur far_align"
        assert "aucun échafaudage" in scaffold_banner({"SYLVAN_PLANNER_FAR_ALIGN": "0"}), \
            "garde 2 crie sur un échafaudage ÉTEINT"
        # Garde anti-respawn (l'erreur la plus coûteuse : 25 des 39 « repas » étaient des resets).
        # Le compteur canonique doit être STRICTEMENT sous le comptage naïf sur un corpus à morts.
        T = _ticks(good)
        E = [float(t["obs"]["energy"]) for t in T]
        TH = [float(t["obs"]["thirst"]) for t in T]
        naif = sum(1 for i in range(1, len(E))
                   if E[i] - E[i - 1] > CONSUME_JUMP or TH[i] - TH[i - 1] > CONSUME_JUMP)
        vrai = consumptions(good, ticks=T)
        assert vrai < naif, "garde anti-respawn INACTIVE (le naïf devrait sur-compter)"
        print(f"[selfcheck] OK — les 3 gardes attrapent les erreurs réelles du 2026-07-21 "
              f"(anti-respawn : naïf {naif} → réel {vrai})")
        return
    print(scaffold_banner())
    if args.corpus:
        s = sanity(args.corpus)
        print(f"\n[guards] {args.corpus}")
        print(f"  constantes MESURÉES : vitesse {s['speed']:.4f} m/tick | drains {s['drain_e']:.4f}/"
              f"{s['drain_t']:.4f} | restore absorbé {s['restore_e_absorbed']:.1f}/"
              f"{s['restore_t_absorbed']:.1f}")
        print(f"  immobile {100*s['immobile_frac']:.0f} % | {s['consommations']} conso | {s['ticks']} ticks")
        print("  " + ("✅ corpus exploitable" if not s["anomalies"]
                      else "🚨 ANOMALIES — ne PAS rendre de verdict :\n    - " + "\n    - ".join(s["anomalies"])))


if __name__ == "__main__":
    main()
