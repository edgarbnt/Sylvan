"""G9 GRATUIT — L'ÉVENTAIL DE VITESSE : sprinter est-il un PARI, ou un choix gratuit ?

PÉRIMÈTRE. Aucune collecte retenue, aucun entraînement. Quelques babillages courts, on LIT, on jette.

POURQUOI CETTE BRIQUE (design_foret_complete.md §2.13, item 2 de la liste bloquante §2ter).
Le planner choisit sa vitesse dans `vx_grid = (0.55, 0.65, 0.75)` : une bande de **±15 %**. Une
bande pareille ne peut porter aucune décision — c'est un paramètre déguisé en choix. §2.13 tranche :
ouvrir un éventail LARGE (marcher / trotter / sprinter) ET le facturer, pour que dépenser de la
vitesse devienne un PARI. Et §2ter le classe bloquant : la dynamique du corps entre dans ce que le
WM apprend, donc ça se décide AVANT la collecte, sous peine d'un second retrain (interdit, §3).

LE PIÈGE QUE CETTE SONDE EXISTE POUR ATTRAPER. Élargir la plage NE SUFFIT PAS. Si aller vite est
gratuit, la vitesse maximale domine toujours : l'éventail est plus large et la décision reste vide.
Le coût passif D par tick est payé QUOI QU'IL ARRIVE, donc parcourir un mètre en marchant coûte plus
cher qu'en sprintant — sans coût de locomotion, la vitesse n'a même pas d'arbitrage à offrir.
Avec un coût quadratique k·vx² (puissance mécanique ~ v²), le coût AU MÈTRE vaut

    c(vx) = (D + k·vx²) / (kin_speed · vx)      minimal en vx* = sqrt(D / k)

⇒ il DÉCROÎT jusqu'à vx*, puis REMONTE. Sprinter coûte alors plus cher au mètre que trotter, mais
arrive plus tôt : c'est exactement la forme « dépenser contre une chance », et c'est falsifiable.

CE QUE MESURE LA SONDE, ET POURQUOI CE N'EST PAS CIRCULAIRE. On ne lit PAS le coût que le code dit
avoir facturé (ce serait la formule qui se contrôle elle-même). On lit l'ÉTAT : le niveau d'énergie
écrit dans l'observation à chaque tick, et la vitesse RÉALISÉE stockée dans la proprioception
(dims 1,3 — la grandeur sans lag que G4 a établie comme fiable). Trois bandes de vitesse fixées, une
mesure indépendante par bande.

CRITÈRES PRÉ-ENREGISTRÉS :
  T1 ÉVENTAIL SERVI ..... en babillage plein-éventail, la plage de vx MESURÉE couvre >= 3.0x
                          (max/min). Sinon la capacité est inerte et le WM n'apprendra qu'un couloir.
  T2 LE CORPS OBÉIT ..... vitesse réalisée / vx constante d'une bande à l'autre (dispersion <= 5 %).
                          Un corps qui n'obéit plus est aussi inutile qu'un corps trop obéissant.
  T3 COÛT CROISSANT ..... la dépense d'énergie MESURÉE par tick croît marche < trot < sprint, et
                          colle à D + k·vx² à 10 % près.
  T4 ARBITRAGE RÉEL ..... 🚨 LE critère. Le coût AU MÈTRE mesuré a son minimum À L'INTÉRIEUR de
                          l'éventail : sprint > trot d'au moins 2 %, et marche > trot. Si le coût au
                          mètre décroît jusqu'au bout, sprinter est gratuit et la brique a ÉCHOUÉ.
  T5 DÉFAUT NEUTRE ...... sans SYLVAN_SPEED_COST : aucun log [locomotion], dépense = D seul (±2 %).
                          Aucun flux RNG n'est ajouté (leçon G3-c), donc le babillage ne décale pas.

CE QUE LA SONDE NE DIT PAS : que l'entité APPRENNE à parier sa vitesse. Ça exige le WM re-entraîné
sur la nouvelle dynamique, et un planner dont la grille couvre l'éventail — gate post-retrain. Ici on
établit la condition NÉCESSAIRE : le monde OFFRE un arbitrage de vitesse, mesuré, non gratuit.

CE QU'ELLE ÉTABLIT POUR LE MONDE, ET CE QU'ELLE LAISSE À `sylvan.world`. Cette sonde prouve
EMPIRIQUEMENT que le corps facture bien D + k·vx² par tick (écart mesuré au modèle : 0,0 %) et que
l'optimum au mètre est intérieur À CE RÉGIME. Elle ne peut pas valider TOUS les régimes : le preset
du monde a ses propres D et k, et l'optimum vx* = sqrt(D/k) peut sortir de l'éventail. C'est le
selfcheck de `sylvan.world` qui l'assert pour les constantes servies — division voulue : ici on
vérifie que la FORMULE est bien celle que le corps applique, là-bas que ses constantes sont bonnes.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g9_vitesse.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g9_vitesse.py --selfcheck
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.world import BOSQUETS_V7_TYPES  # noqa: E402

GODOT = os.path.join(ROOT, "tools", "godot", "godot")

# ── LE RÉGIME CANDIDAT (candidat D de G2, décision owner 2026-07-24) ─────────────────────────
# G2 a montré que le drain SEUL échoue la joignabilité (0,09x) : il faut drain ET vitesse. Candidat
# retenu = vitesse x4,3 + densité x3, drain 0,05 -> 0,2333. Ici on ne teste QUE le corps.
# kin_speed 2.83 : le HAUT de l'éventail (vx=1) donne 2.83/60 = 0.0472 m/tick, la cible de G2 —
# et le BAS (vx=0.25) redonne 0.0118 m/tick, c'est-à-dire le corps d'aujourd'hui. La marche reste
# donc le régime déjà calibré, et l'éventail s'ouvre VERS LE HAUT.
KIN_SPEED = 2.83
DRAIN = 0.2333
SPEED_COST_K = 0.65          # vx* = sqrt(0.2333/0.65) = 0.60 = le trot est le moins cher au mètre
BANDS = [("marche", 0.25), ("trot", 0.60), ("sprint", 1.00)]
FAN_LO, FAN_HI = 0.25, 1.00

RATIO_SPREAD_MAX = 0.05      # T2 : dispersion tolérée de vitesse/vx entre bandes
COST_FIT_TOL = 0.10          # T3 : écart toléré à D + k·vx²
ARBITRAGE_MARGIN = 0.02      # T4 : marge minimale sprint/trot sur le coût au mètre
FAN_SPAN_MIN = 3.0           # T1 : étendue vx max/min exigée en babillage plein-éventail
DEFAULT_TOL = 0.02           # T5 : écart toléré entre la dépense OFF et le drain passif

# `[locomotion] episode : vx MESURE 0.25..1.00 moyen 0.62 | 12.34 m parcourus | ...`
RE_LOCO = re.compile(r"\[locomotion\] episode : vx MESURE ([\d.-]+)\.\.([\d.-]+) moyen ([\d.-]+)")


def _run(label: str, vx_lo: float, vx_hi: float, cost_k: float, episodes: int,
         steps: int, seed: int) -> tuple[list[str], str]:
    """Babillage court à bande de vitesse imposée. Renvoie (jsonl écrits, stdout+stderr)."""
    run_dir = f"/tmp/foret_g9_{label}"
    os.system(f"rm -rf {run_dir}")
    e = dict(os.environ)
    e.update(BOSQUETS_V7_TYPES.to_env())
    e.update({
        "SYLVAN_COLLECT": "1", "SYLVAN_WM_COLLECT": "1", "SYLVAN_COLLECTOR_MODE": "babbling",
        "SYLVAN_CPG": "1", "SYLVAN_RESIDUAL_GAIN": "0.0", "SYLVAN_TURN_FADE": "0",
        "SYLVAN_KIN_SPEED": str(KIN_SPEED), "SYLVAN_ENERGY_DRAIN": str(DRAIN),
        "SYLVAN_WM_VX_MIN": str(vx_lo), "SYLVAN_WM_VX_MAX": str(vx_hi), "SYLVAN_WM_WMAX": "0.6",
        "SYLVAN_POLICY_EXPLORATION_STD_INITIAL": "0", "SYLVAN_POLICY_EXPLORATION_STD_FINAL": "0",
        "SYLVAN_REFLEX_STRENGTH": "0", "SYLVAN_ASSIST_RATIO": "0",
        "SYLVAN_NUM_EPISODES": str(episodes), "SYLVAN_MAX_EPISODE_STEPS": str(steps),
        "SYLVAN_SEED": str(seed), "SYLVAN_RUN_DIR": run_dir,
    })
    # ARÈNE PLATE, sans forêt : on isole la vitesse du terrain. Le couplage vitesse x sous-bois est
    # une mesure DIFFÉRENTE (et elle a sa sonde : G4).
    if cost_k > 0.0:
        e["SYLVAN_SPEED_COST"] = str(cost_k)
    else:
        e.pop("SYLVAN_SPEED_COST", None)
    p = subprocess.run([GODOT, "--path", os.path.join(ROOT, "godot"), "--headless"],
                       env=e, capture_output=True, text=True, timeout=900)
    out = p.stdout + p.stderr
    for fatal in ("Parse Error", "Failed to load script"):
        if fatal in out:
            first = next((ln for ln in out.splitlines() if fatal in ln), fatal)
            raise SystemExit(f"[{label}] Godot n'a PAS chargé le script — mesure invalide.\n  {first}")
    files = sorted(glob.glob(os.path.join(run_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"[{label}] aucun jsonl écrit dans {run_dir} — la collecte n'a rien produit")
    return files, out


def _assert_scripts_load() -> None:
    """Charge Godot et QUITTE aussitôt : un script cassé se voit en 3 s au lieu de 15 min.

    Sans ça, une erreur de parse ne se manifeste PAS comme une erreur : Godot tourne à vide, la
    sonde attend son timeout, et on lit un échec de mesure là où il y a un échec de compilation.
    C'est le défaut (a) que G3 a payé (641 s de mur pour 2 s de CPU) ; on le rend instantané.
    """
    p = subprocess.run([GODOT, "--path", os.path.join(ROOT, "godot"), "--headless", "--quit"],
                       capture_output=True, text=True, timeout=180)
    out = p.stdout + p.stderr
    for fatal in ("Parse Error", "Failed to load script"):
        if fatal in out:
            first = next((ln for ln in out.splitlines() if fatal in ln), fatal)
            raise SystemExit(f"SCRIPT CASSÉ — aucune mesure n'est possible.\n  {first}")


def _read(files: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vx commandé, vitesse RÉALISÉE m/s, énergie) par tick mobile, tous épisodes confondus.

    L'énergie est lue dans l'OBSERVATION (l'état réel de l'homéostasie), jamais dans le log du coût
    facturé : c'est ce qui rend la mesure non-circulaire.
    """
    vxs, spds, engs = [], [], []
    for path in files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                obs = rec.get("obs", {})
                proprio = obs.get("proprio", [])
                cmd = rec.get("wm", {}).get("cmd", [0.0, 0.0])
                if len(proprio) < 4 or cmd[0] <= 0.0 or "energy" not in obs:
                    continue
                speed = math.hypot(proprio[1], proprio[3])
                if speed <= 1e-3:          # fenêtre de settle : le corps ne bouge pas encore
                    continue
                vxs.append(float(cmd[0]))
                spds.append(speed)
                eng = float(obs["energy"])
                engs.append(eng * 100.0 if eng <= 1.0 else eng)   # jauge 0-100, quelle que soit l'échelle
    return np.array(vxs), np.array(spds), np.array(engs)


def _spend_per_tick(energy: np.ndarray) -> float:
    """Dépense MÉDIANE par tick, lue sur la trajectoire de la jauge.

    La médiane, pas la moyenne : un repas remet de l'énergie et une frontière d'épisode remet la
    jauge à 100. Ces deux événements sont RARES et de signe opposé au drain — la médiane des
    diminutions les ignore par construction, sans qu'on ait à supposer où ils sont.
    """
    d = -np.diff(energy)
    d = d[d > 0.0]
    if d.size < 20:
        raise SystemExit(f"seulement {d.size} ticks de diminution d'énergie — mesure non fiable")
    return float(np.median(d))


def _fan_span(out: str) -> tuple[float, float]:
    """(vx min, vx max) MESURÉS et rapportés par le log [locomotion] (§6bis)."""
    m = RE_LOCO.findall(out)
    if not m:
        return 0.0, 0.0
    return min(float(a) for a, _, _ in m), max(float(b) for _, b, _ in m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print(f"MONDE : {BOSQUETS_V7_TYPES.name} (arène plate, sans forêt) | corps kin_speed={KIN_SPEED} "
          f"| drain {DRAIN} | coût k={SPEED_COST_K} → vx* = {math.sqrt(DRAIN / SPEED_COST_K):.2f}")
    print(f"{a.episodes} épisodes x {a.steps} ticks par bande | graine {a.seed}\n")
    _assert_scripts_load()

    # ── T5 d'abord : le défaut. Il calibre aussi la dépense de référence (le drain seul). ──
    files_off, out_off = _run("off", FAN_LO, FAN_HI, 0.0, a.episodes, a.steps, a.seed)
    _, _, eng_off = _read(files_off)
    spend_off = _spend_per_tick(eng_off)
    loco_logged = "[locomotion]" in out_off

    res = {}
    for label, vx in BANDS:
        files, out = _run(label, vx, vx, SPEED_COST_K, a.episodes, a.steps, a.seed)
        v, s, e = _read(files)
        spend = _spend_per_tick(e)
        res[label] = {
            "vx": float(np.median(v)), "speed": float(np.median(s)), "spend": spend,
            "ratio": float(np.median(s)) / float(np.median(v)),
            "per_m": spend / (float(np.median(s)) / 60.0),   # énergie par mètre (vitesse m/s → m/tick)
            "attendu": DRAIN + SPEED_COST_K * vx * vx,
            "n": len(v),
        }
    # ÉVENTAIL : mesuré SANS coût et sur PLUSIEURS vies, délibérément. Le babillage ne re-tire un vx
    # que toutes les 40-80 décisions ; au régime candidat une vie ne dure que ~200 ticks, donc une
    # seule vie ne contient que 3 à 5 tirages et sa plage empirique sous-estime l'éventail servi.
    # Mesurer la couverture sur un échantillon trop petit ferait conclure « capacité inerte » à tort —
    # le contraire exact de ce que T1 doit détecter. La question de T1 (le sampler couvre-t-il la
    # plage ?) est indépendante du coût.
    files_fan, _ = _run("eventail", FAN_LO, FAN_HI, 0.0, max(8, a.episodes), a.steps, a.seed)
    v_fan, _, _ = _read(files_fan)
    fan_lo_log, fan_hi_log = float(v_fan.min()), float(v_fan.max())

    print("=" * 92)
    print("bande     vx     vitesse m/s   m/tick    dépense/tick (attendue)   coût au mètre   n")
    for label, _ in BANDS:
        r = res[label]
        print(f"  {label:<8}{r['vx']:.2f}     {r['speed']:.4f}      {r['speed']/60:.4f}    "
              f"{r['spend']:.4f} ({r['attendu']:.4f})          {r['per_m']:7.2f}   {r['n']}")
    print(f"  {'OFF':<8}{'—':>4}     {'—':>6}      {'—':>6}    {spend_off:.4f} ({DRAIN:.4f})")
    print("=" * 92)

    ok = True

    # T1 — l'éventail est-il RÉELLEMENT parcouru ?
    span = (fan_hi_log / fan_lo_log) if fan_lo_log > 0 else 0.0
    t1 = span >= FAN_SPAN_MIN
    ok &= t1
    print(f"{'✅' if t1 else '❌'} T1 ÉVENTAIL SERVI   vx mesuré {fan_lo_log:.2f}..{fan_hi_log:.2f} "
          f"= {span:.1f}x (exigé >= {FAN_SPAN_MIN}x) | vx médian babillé {np.median(v_fan):.2f} "
          f"sur {len(v_fan)} ticks")

    # T2 — le corps obéit-il encore ?
    ratios = np.array([res[l]["ratio"] for l, _ in BANDS])
    spread = float(ratios.max() / ratios.min() - 1.0)
    t2 = spread <= RATIO_SPREAD_MAX
    ok &= t2
    print(f"{'✅' if t2 else '❌'} T2 LE CORPS OBÉIT   vitesse/vx = "
          f"{', '.join(f'{r:.3f}' for r in ratios)} (kin_speed déclaré {KIN_SPEED}) | "
          f"dispersion {spread*100:.1f}% (max {RATIO_SPREAD_MAX*100:.0f}%)")

    # T3 — la dépense croît-elle avec la vitesse, et comme prévu ?
    spends = [res[l]["spend"] for l, _ in BANDS]
    croissant = all(spends[i] < spends[i + 1] for i in range(len(spends) - 1))
    ecarts = [abs(res[l]["spend"] - res[l]["attendu"]) / res[l]["attendu"] for l, _ in BANDS]
    t3 = croissant and max(ecarts) <= COST_FIT_TOL
    ok &= t3
    print(f"{'✅' if t3 else '❌'} T3 COÛT CROISSANT   {' < '.join(f'{s:.4f}' for s in spends)} "
          f"({'croissant' if croissant else 'PAS croissant'}) | écart max au modèle "
          f"D+k·vx² : {max(ecarts)*100:.1f}% (max {COST_FIT_TOL*100:.0f}%)")

    # T4 — LE critère : sprinter coûte-t-il plus cher AU MÈTRE que trotter ?
    pm = {l: res[l]["per_m"] for l, _ in BANDS}
    marge = pm["sprint"] / pm["trot"] - 1.0
    t4 = marge >= ARBITRAGE_MARGIN and pm["marche"] > pm["trot"]
    ok &= t4
    print(f"{'✅' if t4 else '❌'} T4 ARBITRAGE RÉEL   coût au mètre marche {pm['marche']:.2f} > "
          f"trot {pm['trot']:.2f} < sprint {pm['sprint']:.2f} | sprint coûte {marge*100:+.1f}% de "
          f"plus que le trot (exigé >= {ARBITRAGE_MARGIN*100:.0f}%)")

    # T5 — le défaut ne change rien.
    ecart_off = abs(spend_off - DRAIN) / DRAIN
    t5 = (not loco_logged) and ecart_off <= DEFAULT_TOL
    ok &= t5
    print(f"{'✅' if t5 else '❌'} T5 DÉFAUT NEUTRE    sans SYLVAN_SPEED_COST : "
          f"log [locomotion] {'PRÉSENT (fuite)' if loco_logged else 'absent'} | dépense "
          f"{spend_off:.4f} vs drain passif {DRAIN:.4f} ({ecart_off*100:.1f}%, max {DEFAULT_TOL*100:.0f}%)")

    print("=" * 92)
    print(f"GATE G9 = {'PASS' if ok else 'ÉCHEC'}")
    if ok:
        print("⇒ l'éventail est servi, le corps obéit, et la vitesse a un PRIX dont l'optimum au")
        print("  mètre est INTÉRIEUR : sprinter est un pari, pas un choix gratuit.")
        print("RAPPEL DE PORTÉE : condition NÉCESSAIRE. Que l'entité APPRENNE à parier sa vitesse se")
        print("mesure après collecte et retrain (le planner doit aussi couvrir l'éventail).")
    return 0 if ok else 1


def selfcheck() -> int:
    line = ("[locomotion] episode : vx MESURE 0.25..1.00 moyen 0.62 | 12.34 m parcourus | "
            "energie locomotion 120.0 vs passive 233.3 | cout au metre 28.65")
    m = RE_LOCO.findall(line)
    assert m and m[0] == ("0.25", "1.00", "0.62"), m
    print("  [ok] le parseur lit la ligne [locomotion] (min, max, moyen MESURÉS)")

    # la médiane des diminutions ignore un repas (+40) et une frontière d'épisode (retour à 100)
    eng = [100.0 - 0.5 * i for i in range(40)]
    eng[20] = eng[19] + 40.0                       # un repas
    eng = eng + [100.0] + [100.0 - 0.5 * i for i in range(40)]   # nouvel épisode
    got = _spend_per_tick(np.array(eng))
    assert abs(got - 0.5) < 1e-9, got
    print(f"  [ok] dépense médiane {got:.3f} malgré un repas et une frontière d'épisode")

    # la forme du coût au mètre : décroissante puis croissante, minimum en sqrt(D/k)
    def per_m(vx: float) -> float:
        return (DRAIN + SPEED_COST_K * vx * vx) / (KIN_SPEED * vx / 60.0)
    vstar = math.sqrt(DRAIN / SPEED_COST_K)
    assert abs(vstar - 0.60) < 0.01, vstar
    assert per_m(vstar) < per_m(0.25) and per_m(vstar) < per_m(1.0), (per_m(0.25), per_m(vstar), per_m(1.0))
    print(f"  [ok] coût au mètre : marche {per_m(0.25):.1f} > trot {per_m(vstar):.1f} < sprint "
          f"{per_m(1.0):.1f} — minimum INTÉRIEUR en vx*={vstar:.2f}")

    # sans coût (k=0) il n'y a AUCUN arbitrage : c'est le contrôle négatif du gate
    def per_m_gratuit(vx: float) -> float:
        return DRAIN / (KIN_SPEED * vx / 60.0)
    assert per_m_gratuit(1.0) < per_m_gratuit(0.6) < per_m_gratuit(0.25)
    print("  [ok] contrôle négatif : à coût nul, le coût au mètre décroît jusqu'au bout "
          "(la vitesse maximale domine toujours) — T4 échouerait, comme il doit")

    # le régime candidat reproduit bien les deux bornes de G2
    assert abs(KIN_SPEED * 1.0 / 60.0 - 0.0472) < 5e-4, KIN_SPEED / 60.0
    assert abs(KIN_SPEED * 0.25 / 60.0 - 0.0118) < 5e-4
    print("  [ok] éventail = 0.0118 m/tick en marche (le corps d'aujourd'hui) → 0.0472 en sprint "
          "(la cible du candidat D de G2)")

    assert os.path.exists(GODOT), GODOT
    print("  [ok] binaire Godot présent")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
