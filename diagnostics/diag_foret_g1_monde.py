"""G1 GRATUIT du monde-forêt — DEUX mesures qu'aucun gate ne couvrait, avant toute collecte.

PÉRIMÈTRE. Cette sonde ne collecte rien et n'entraîne rien. Elle lance Godot en mode babillage
(aucun serveur planner) et LIT ce que le monde rapporte. Coût : quelques minutes.

────────────────────────────────────────────────────────────────────────────────────────────
GATE A — COÛT DE CALCUL (design_foret_complete.md §6quater D)
────────────────────────────────────────────────────────────────────────────────────────────
« Grande forêt + proies + distracteurs + flaques = beaucoup plus de raycasts par tick, sur des
millions de ticks. Mesurer les ticks/seconde sur le monde cible AVANT de s'engager. Si une collecte
de 25 vies dépasse ~45 min, on réduit la densité. Aucun gate ne couvrait ça. »

🚨 PREMIÈRE VERSION DE CE GATE : INVALIDE, ET POURQUOI (mesuré le 2026-07-24).
On mesurait des ticks/seconde EN TEMPS DE MUR. Résultat : 100,6 s en monde plat, 100,6 s en forêt
dense — soit 59,7 ticks/s des deux côtés, pour 6000 ticks. Or 6000 / 60 Hz = 100,0 s exactement :
Godot headless tourne à la cadence physique TEMPS RÉEL (rien ne configure `physics_ticks_per_second`
ni `max_fps` dans project.godot). Le temps de mur ne mesurait donc pas le calcul, il mesurait
l'horloge. Un tel montage rapporte « surcoût 1,00x » quelle que soit la densité, tant qu'on reste
sous le budget temps réel — c'est-à-dire qu'il ne peut RIEN réfuter. On ne garde pas ce PASS.

CE QU'ON MESURE À LA PLACE : le TEMPS CPU consommé par tick (user+sys du processus fils, via
getrusage). C'est le coût de calcul réel, et il reste lisible tant que Godot dort entre deux ticks.
Le budget temps réel est de 1/60 s = 16,67 ms par tick : au-delà, le monde ne tient plus la cadence
et le temps de mur se met à déraper. La grandeur ACTIONNABLE — celle qui dit « réduire la densité » —
est donc la MARGE : combien de fois le coût actuel tient-il dans 16,67 ms.

Conséquence à énoncer clairement : tant qu'on reste sous 16,67 ms/tick, la durée d'une collecte ne
dépend PAS de la densité (25 vies x 3000 ticks = 75 000 ticks = 20,8 min, densité comprise ou non).
Le seul levier sur le temps de mur d'une collecte est le PARALLÉLISME — ce que la spec §2.13 dit déjà.

  PASS ........ CPU/tick(forêt dense) <= 8,33 ms  → au moins 2x de marge avant le mur temps réel
  WARN ........ 8,33 < CPU/tick <= 16,67 ms       → réduire la densité AVANT d'empiler d'autres objets
  KILL ........ CPU/tick > 16,67 ms               → la cadence décroche, la collecte déborde

Contrôles : homéostasie DÉSACTIVÉE, pour que chaque condition exécute EXACTEMENT le même nombre de
ticks (sinon un monde où l'agent meurt plus tôt paraît « moins cher ») ; et une condition de CHARGE
à 120 arbres, pour mesurer comment le coût ÉCHELONNE au lieu de l'extrapoler.

────────────────────────────────────────────────────────────────────────────────────────────
GATE B — ARRANGEMENT ÉCOLOGIQUE (design_foret_complete.md §2.2 et §6)
────────────────────────────────────────────────────────────────────────────────────────────
`forest_solid.gd` implémente un processus de Neyman-Scott/Thomas (peuplements + clairières) et
loggue un indice de Clark-Evans. Le code EXISTE mais n'a JAMAIS été exécuté. Attendu par la spec :
« Clark-Evans < 1 en peuplements, ≈ 1 en uniforme ».

⚠️ POURQUOI ON NE JUGE PAS SUR LE SEUL INDICE. Clark-Evans porte deux biais connus sur ce monde :
  (a) EFFET DE BORD — domaine borné, les arbres du pourtour n'ont pas de voisin au-delà : gonfle R ;
  (b) le tirage uniforme HISTORIQUE tire le rayon en randf_range(rmin, rmax), donc uniformément en
      RAYON et non en AIRE : il laisse une densité en 1/r, donc groupe LÉGÈREMENT par construction.
Le discriminant sans hypothèse est donc la distance au plus proche voisin MESURÉE (mètres), comparée
entre les deux modes à n égal. L'indice est rapporté à côté, comme la spec le demande.

  PASS ................ ppv(peuplements) < ppv(uniforme), IC95 DISJOINTS
                        ET R(peuplements) < 0,90  ET  R(uniforme) dans [0,90 ; 1,10]
  ESTIMATEUR-BIAISÉ ... ppv nettement séparés MAIS R(uniforme) hors bande
                        → l'arrangement marche, c'est la NORMALISATION qui est fausse.
                        Corriger l'estimateur, PAS le monde. (Ne PAS élargir la bande : §2.)
  KILL ................ ppv(peuplements) >= ppv(uniforme) → Neyman-Scott ne groupe pas → réécrire.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g1_monde.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g1_monde.py --selfcheck
"""

from __future__ import annotations

import argparse
import math
import os
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.world import BOSQUETS_V2  # noqa: E402

GODOT = os.path.join(ROOT, "tools", "godot", "godot")

# Densité CIBLE. Bornée par la mesure déjà payée (design_foret_complete.md §3) : 45 arbres = fenêtre
# navigable, 54 => immobile 85 % du temps. On mesure AU plafond documenté, pas au-delà.
TREES = 45
STANDS = 6
CLEARINGS = 3

TREES_STRESS = 120                            # condition de CHARGE : mesurer l'échelonnement, pas l'extrapoler

TICK_BUDGET_MS = 1000.0 / 60.0                # 16,67 ms — le budget d'un tick physique en temps réel
TICK_WARN_MS = TICK_BUDGET_MS / 2.0           # 8,33 ms — moins de 2x de marge = on réduit la densité

# `[forest] structure : 6 peuplements (sigma 3.0 m), 3 clairieres (r 4.0 m) | n=45 ppv_moyen
#  MESURE 0.812 m | aire 359.8 m2 | Clark-Evans MESURE 0.734 (<1 = groupe, 1 = aleatoire)`
RE_STRUCT = re.compile(
    r"\[forest\] structure : (\d+) peuplements .*? n=(\d+) ppv_moyen MESURE ([\d.]+) m"
    r" \| aire ([\d.]+) m2 \| Clark-Evans MESURE ([\d.]+)"
)
RE_PLACED = re.compile(r"\[forest\] episode : (\d+)/(\d+) arbres places")


def _env(trees: int, stands: int, episodes: int, steps: int, run_dir: str, seed: int) -> dict:
    """Le monde GELÉ (BOSQUETS_V2) + le babillage. Aucun serveur : le coût mesuré est celui du MONDE."""
    e = dict(os.environ)
    e.update(BOSQUETS_V2.to_env())
    e.update({
        "SYLVAN_COLLECT": "1",
        "SYLVAN_WM_COLLECT": "1",            # babillage de commandes + snapshot rétine par tick
        "SYLVAN_COLLECTOR_MODE": "babbling",  # => aucun serveur de politique requis
        "SYLVAN_CPG": "1", "SYLVAN_RESIDUAL_GAIN": "0.0", "SYLVAN_TURN_FADE": "0",
        "SYLVAN_WM_VX_MIN": "0.55", "SYLVAN_WM_VX_MAX": "0.75", "SYLVAN_WM_WMAX": "0.6",
        "SYLVAN_POLICY_EXPLORATION_STD_INITIAL": "0", "SYLVAN_POLICY_EXPLORATION_STD_FINAL": "0",
        "SYLVAN_REFLEX_STRENGTH": "0", "SYLVAN_ASSIST_RATIO": "0",
        "SYLVAN_NUM_EPISODES": str(episodes),
        "SYLVAN_MAX_EPISODE_STEPS": str(steps),
        "SYLVAN_SEED": str(seed),
        "SYLVAN_RUN_DIR": run_dir,
        # ticks CONSTANTS entre conditions : sinon un monde plus mortel paraît plus rapide
        "SYLVAN_DISABLE_HOMEOSTASIS": "1",
        "SYLVAN_FOREST_COUNT": str(trees),
        "SYLVAN_FOREST_STANDS": str(stands),
        "SYLVAN_FOREST_CLEARINGS": str(CLEARINGS if stands > 0 else 0),
    })
    return e


def _run(label: str, **kw) -> tuple[float, float, str]:
    """Lance Godot headless. Renvoie (secondes de MUR, secondes de CPU du fils, stdout).

    Le temps CPU vient de getrusage(RUSAGE_CHILDREN) : c'est le seul des deux qui mesure du CALCUL.
    Le temps de mur, lui, est plafonné par la cadence physique temps réel (voir l'en-tête).
    """
    run_dir = f"/tmp/foret_g1_{label}"
    shutil.rmtree(run_dir, ignore_errors=True)
    env = _env(run_dir=run_dir, **kw)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    p = subprocess.run([GODOT, "--path", os.path.join(ROOT, "godot"), "--headless"],
                       env=env, capture_output=True, text=True, timeout=1800)
    dt = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    shutil.rmtree(run_dir, ignore_errors=True)
    out = p.stdout + p.stderr
    if "HOMEOSTASIS DISABLED" not in out:
        # §6bis : ne jamais faire confiance à un réglage DEMANDÉ. On vérifie qu'il a été SERVI.
        raise SystemExit(f"[{label}] le drapeau homéostasie n'a PAS été servi — mesure invalide")
    return dt, cpu, out


def _structs(out: str) -> list[dict]:
    return [{"stands": int(m[0]), "n": int(m[1]), "nn": float(m[2]),
             "area": float(m[3]), "ce": float(m[4])} for m in RE_STRUCT.findall(out)]


def _ci95(xs: list[float]) -> tuple[float, float, float]:
    """(moyenne, demi-largeur IC95, écart-type). IC normal : n >= 20 tirages par condition."""
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, 0.0, 0.0
    sd = statistics.stdev(xs)
    return m, 1.96 * sd / math.sqrt(len(xs)), sd


def gate_a(seed: int, repeats: int) -> dict:
    """COÛT : monde plat vs forêt cible vs forêt de charge, à ticks strictement identiques.

    RÉPÉTÉ. Au premier passage, 120 arbres sont sortis MOINS chers que 45 (1,73 contre 1,86 ms) :
    le coût d'un arbre est donc sous le bruit de mesure. Sans répétition on aurait publié un
    « 4,2 us par arbre » que la condition de charge contredisait dans le même tableau.
    """
    ticks = 2000
    print(f"\n=== GATE A — COÛT DE CALCUL ({ticks} ticks x {repeats} répétitions/condition) ===")
    print(f"    budget d'un tick en temps réel : {TICK_BUDGET_MS:.2f} ms | "
          f"seuil d'alerte {TICK_WARN_MS:.2f} ms")
    res = {}
    conditions = (("plat", 0), (f"foret-{TREES}", TREES), (f"charge-{TREES_STRESS}", TREES_STRESS))
    for label, trees in conditions:
        runs, n = [], 0
        for k in range(repeats):
            wall, cpu, out = _run(f"cost_{label}_{k}", trees=trees, stands=STANDS if trees else 0,
                                  episodes=1, steps=ticks, seed=seed + k)
            runs.append({"ms": cpu / ticks * 1000.0, "rate": ticks / wall})
            placed = RE_PLACED.findall(out)
            n = int(placed[0][0]) if placed else 0
        ms, ms_ci, ms_sd = _ci95([r["ms"] for r in runs])
        res[label] = {"ms": ms, "ms_ci": ms_ci, "ms_sd": ms_sd, "trees": n,
                      "wall_rate": statistics.fmean([r["rate"] for r in runs]),
                      "headroom": TICK_BUDGET_MS / ms}
        print(f"  {label:14s} : {ms:5.2f} ± {ms_ci:.2f} ms/tick  (marge {TICK_BUDGET_MS / ms:4.1f}x)"
              f"   | {n:3d} arbres places | {repeats} runs")

    flat, target, stress = res["plat"], res[f"foret-{TREES}"], res[f"charge-{TREES_STRESS}"]
    delta = stress["ms"] - flat["ms"]
    noise = flat["ms_ci"] + stress["ms_ci"]
    if delta > noise:
        print(f"  coût MARGINAL d'un arbre : {delta / TREES_STRESS * 1000.0:.1f} us/tick "
              f"(écart {delta:+.2f} ms > bruit {noise:.2f} ms sur 0→{TREES_STRESS} arbres)")
    else:
        # §2 : on ne convertit pas du bruit en chiffre. On rapporte une BORNE, pas une valeur.
        print(f"  coût MARGINAL d'un arbre : SOUS LE BRUIT — écart 0→{TREES_STRESS} arbres "
              f"{delta:+.2f} ms <= bruit {noise:.2f} ms  ⇒  borne < {noise / TREES_STRESS * 1000.0:.1f} us/arbre")
    print(f"  25 vies x 3000 ticks : {25 * 3000 / target['wall_rate'] / 60:.1f} min de mur "
          "(plafonné par la cadence temps réel, PAS par la densité)")
    worst = max(target["ms"] + target["ms_ci"], stress["ms"] + stress["ms_ci"])
    if worst > TICK_BUDGET_MS:
        v = "KILL"
    elif worst > TICK_WARN_MS:
        v = "WARN"
    else:
        v = "PASS"
    print(f"  VERDICT A = {v}")
    return {"verdict": v, "delta_ms": delta, "noise_ms": noise, "headroom": target["headroom"],
            "collect_min": 25 * 3000 / target["wall_rate"] / 60, **res}


def gate_b(seed: int, episodes: int) -> dict:
    """ARRANGEMENT : le placement se fait dans begin_episode → des épisodes de 5 ticks suffisent."""
    print(f"\n=== GATE B — ARRANGEMENT ÉCOLOGIQUE ({episodes} épisodes/condition, 5 ticks chacun) ===")
    res = {}
    for label, stands in (("uniforme", 0), ("peuplements", STANDS)):
        _, _, out = _run(f"struct_{label}", trees=TREES, stands=stands,
                         episodes=episodes, steps=5, seed=seed)
        st = _structs(out)
        if len(st) < 3:
            raise SystemExit(f"[{label}] {len(st)} lignes de structure lues — la sonde ne mesure rien")
        nn, nn_ci, _ = _ci95([s["nn"] for s in st])
        ce, ce_ci, _ = _ci95([s["ce"] for s in st])
        n_mean = statistics.fmean([s["n"] for s in st])
        res[label] = {"nn": nn, "nn_ci": nn_ci, "ce": ce, "ce_ci": ce_ci,
                      "n": n_mean, "area": st[0]["area"], "draws": len(st)}
        print(f"  {label:12s} : ppv {nn:.3f} ± {nn_ci:.3f} m | Clark-Evans {ce:.3f} ± {ce_ci:.3f} "
              f"| n={n_mean:.1f}/{TREES} placés | aire {st[0]['area']:.1f} m2 | {len(st)} tirages")

    u, p = res["uniforme"], res["peuplements"]
    disjoint = (p["nn"] + p["nn_ci"]) < (u["nn"] - u["nn_ci"])
    grouped = p["nn"] < u["nn"]
    ce_ref_ok = 0.90 <= u["ce"] <= 1.10
    print(f"  ppv peuplements {p['nn']:.3f} vs uniforme {u['nn']:.3f} → "
          f"{'IC95 DISJOINTS' if disjoint else 'IC95 qui se recouvrent'}")
    if not grouped:
        v = "KILL"
    elif disjoint and p["ce"] < 0.90 and ce_ref_ok:
        v = "PASS"
    elif disjoint:
        v = "ESTIMATEUR-BIAISÉ"
    else:
        v = "KILL"
    print(f"  VERDICT B = {v}  (témoin uniforme R={u['ce']:.3f}, bande attendue [0,90 ; 1,10])")
    return {"verdict": v, **res}


def selfcheck() -> int:
    """Vérifie la sonde elle-même : parseur et statistique, sans lancer Godot."""
    line = ("[forest] structure : 6 peuplements (sigma 3.0 m), 3 clairieres (r 4.0 m) | n=45 "
            "ppv_moyen MESURE 0.812 m | aire 359.8 m2 | Clark-Evans MESURE 0.734 "
            "(<1 = groupe, 1 = aleatoire)")
    got = _structs(line + "\n" + line)
    assert len(got) == 2, got
    assert got[0] == {"stands": 6, "n": 45, "nn": 0.812, "area": 359.8, "ce": 0.734}, got[0]
    print("  [ok] le parseur lit la ligne [forest] structure émise par forest_solid.gd")

    assert _structs("[forest] episode : 45/45 arbres places") == []
    print("  [ok] il ne confond pas la ligne de placement avec la ligne de structure")

    m, ci, sd = _ci95([1.0, 1.0, 1.0, 1.0])
    assert (m, ci, sd) == (1.0, 0.0, 0.0)
    m, ci, _ = _ci95([0.8, 1.0, 1.2])
    assert abs(m - 1.0) < 1e-9 and ci > 0
    print("  [ok] IC95 nul sur une constante, non nul sur une dispersion")

    assert abs(TICK_BUDGET_MS - 16.667) < 0.01, TICK_BUDGET_MS
    print(f"  [ok] budget d'un tick temps réel : {TICK_BUDGET_MS:.2f} ms "
          f"(alerte à {TICK_WARN_MS:.2f} ms)")

    # Le piège qui a invalidé la v1 : le temps de MUR ne peut pas descendre sous ticks/60, quel que
    # soit le calcul. La sonde doit donc juger sur le CPU, jamais sur le mur.
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    assert hasattr(r, "ru_utime") and hasattr(r, "ru_stime")
    print("  [ok] le temps CPU du fils est lisible (getrusage) — c'est lui qui juge, pas le mur")

    assert os.path.exists(GODOT), GODOT
    print("  [ok] binaire Godot présent")
    print("SELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--struct-episodes", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", choices=["a", "b"], default=None)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print(f"MONDE : {BOSQUETS_V2.name} | forêt {TREES} arbres, {STANDS} peuplements, "
          f"{CLEARINGS} clairières | graine {a.seed}")
    va = gate_a(a.seed, a.repeats) if a.only != "b" else None
    vb = gate_b(a.seed, a.struct_episodes) if a.only != "a" else None
    print("\n=== VERDICTS ===")
    if va:
        bound = va["noise_ms"] / TREES_STRESS * 1000.0
        cost = (f"{va['delta_ms'] / TREES_STRESS * 1000.0:.1f} us/arbre"
                if va["delta_ms"] > va["noise_ms"] else f"sous le bruit, borne < {bound:.1f} us/arbre")
        print(f"  A coût de calcul  : {va['verdict']}  (marge {va['headroom']:.1f}x avant le mur "
              f"temps réel ; {cost} ; 25 vies ≈ {va['collect_min']:.1f} min de mur)")
    if vb:
        print(f"  B arrangement     : {vb['verdict']}  "
              f"(ppv {vb['peuplements']['nn']:.3f} vs {vb['uniforme']['nn']:.3f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
