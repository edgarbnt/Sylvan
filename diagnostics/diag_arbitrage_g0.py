"""G0 of the target-arbitration critic chantier (docs/design_critique_arbitrage.md) — FREE (0 run/train).

Falsifiable question: over ALREADY-lived multi-drive lives, how many deaths are due to TARGET
ARBITRATION (the resource of the dying drive was SEEN and metabolically REACHABLE, but another
target was chosen at the last useful replan), as opposed to METABOLIC deaths (nothing reachable
— known ceiling, out of scope), PERCEPTUAL (never seen) and DANGER (health) deaths?

Pre-registered verdict (the design doc): place (recoverable arbitration deaths, normalized per
24 lives) > noise ±5 meal-equivalents -> chantier licensed, and the LOCALIZATION (never-switched /
late-switch / camping) picks the voie A/B/C. Place <= noise -> STOP, committed negative.

Reads the BC_LOG stream (ep_*.jsonl[.gz] — drives per tick, plan.target + ego positions per
replan). Cross-check --selfcheck: life count vs parse_lives(godot.log) (diag_hazard_gate).

Run (repo root):
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_arbitrage_g0.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys

# MEASURED body/world constants (declared, not tuned — §2):
DRAIN_PER_TICK = 0.05      # real energy/thirst drain per step (measured, design_critique_sprint Phase 0)
SPEED_M_PER_TICK = 0.01    # MESURÉ (2026-07-21) : déplacement par tick = 0.0100 m, constant
                           # (p50=p90=p99). L ancienne valeur 0.02 était 2x TROP GRANDE et rendait la
                           # portée métabolique 2x optimiste -> morts étiquetées "arbitrage" à tort.
DEATH_THR = 15.0           # same convention as parse_lives (drive <= 15 at the end = cause)
CONSUME_JUMP = 5.0         # drive rise > threshold = consumption event
START_DRIVE = 70.0         # measured reset signature: lives start at (e=70, t=70, h=100)
START_TOL = 0.3            # tolerance around the reset signature
FRESH_H = 99.5             # health at reset
CAMP_WINDOW = 600          # steps — "camping" window before death
LOW_DRIVE = 50.0           # dying drive already below this while consuming the other = camping

DEFAULT_RUNS = [
    "data/replay_buffer/critic_kin_g24as1", "data/replay_buffer/critic_kin_g24as2",
    "data/replay_buffer/critic_kin_g24bs1", "data/replay_buffer/critic_kin_g24bs2",
    "data/replay_buffer/critic_kin_spx3", "data/replay_buffer/critic_kin_spx4",
    "data/replay_buffer/critic_kin_judge1", "data/replay_buffer/critic_kin_judge2",
    "data/replay_buffer/critic_kin_pure1", "data/replay_buffer/critic_kin_pure2",
]

NEEDED_RES = {"energy": "food", "thirst": "water"}

CLASSES = ["arbitrage_jamais", "arbitrage_tardif", "poursuite_echouee",
           "metabolique_vue", "jamais_vue", "danger", "tronquee"]


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def iter_ticks(run: str):
    for ep in sorted(glob.glob(os.path.join(run, "ep_*.jsonl*"))):
        with _open(ep) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def split_lives(run: str) -> list[dict]:
    """Split the BC stream into lives.

    A new life = the MEASURED reset signature (e=70, t=70, h=100) reached by a JUMP from the
    previous tick (drives only drift by 0.05/step mid-life; consumptions move ONE drive by +40).
    A plain "both drives jump" rule misses the interesting case (dying of one drive while the
    OTHER is full) — caught by --selfcheck vs godot.log.
    """
    lives: list[dict] = []
    cur: dict | None = None
    prev_e = prev_t = None
    prev_h = 100.0
    for rec in iter_ticks(run):
        o = rec["obs"]
        e, t, h = float(o["energy"]), float(o["thirst"]), float(o.get("health", 100.0))
        at_start = (abs(e - START_DRIVE) < START_TOL and abs(t - START_DRIVE) < START_TOL
                    and h >= FRESH_H)
        jumped = prev_e is not None and (abs(e - prev_e) > 1.0 or abs(t - prev_t) > 1.0
                                         or h - prev_h > 1.0)
        is_reset = cur is not None and at_start and jumped
        if cur is None or is_reset:
            if cur is not None:
                lives.append(cur)
            cur = {"drives": [], "plans": [], "meals": 0, "drinks": 0,
                   "consume_ticks": []}  # consume_ticks: (local tick idx, "food"|"water")
        else:
            if e - prev_e > CONSUME_JUMP:
                cur["meals"] += 1
                cur["consume_ticks"].append((len(cur["drives"]), "food"))
            if t - prev_t > CONSUME_JUMP:
                cur["drinks"] += 1
                cur["consume_ticks"].append((len(cur["drives"]), "water"))
        cur["drives"].append((e, t, h))
        plan = rec.get("plan")
        if plan is not None:
            entry = {"i": len(cur["drives"]) - 1, "target": plan.get("target", "none")}
            for res in ("food", "water"):
                pos = plan.get(res)
                if pos is not None:
                    entry[res] = math.hypot(float(pos[0]), float(pos[1]))
            cur["plans"].append(entry)
        prev_e, prev_t, prev_h = e, t, h
    if cur is not None and cur["drives"]:
        lives.append(cur)
    return lives


def classify_life(life: dict, reach_factor: float,
                  drains: tuple[float, float] = (DRAIN_PER_TICK, DRAIN_PER_TICK)) -> dict:
    """Classify one life's outcome; for a drive death, find the last lived chance.

    The class is judged on the plan target AT the last useful replan (the latest replan where
    the needed resource was visible AND metabolically reachable given the reserve at that tick).

    `drains` = (energy, thirst) per-tick drain. Metabolic reach depends on the drain of the DYING
    gauge, so an asymmetric body needs its own value per gauge (a graded world drains thirst more
    slowly => water is reachable further than the old single-drain formula assumed). Default keeps
    both at DRAIN_PER_TICK => bit-identical to the previous behaviour on a symmetric body.
    """
    e, t, h = life["drives"][-1]
    n = len(life["drives"])
    out = {"len": n, "meals": life["meals"], "drinks": life["drinks"],
           "flips": 0, "camp": False}
    # Flips: food<->water target switches with no consumption in between (flottement).
    consume_is = [i for i, _ in life["consume_ticks"]]
    last_tgt: str | None = None
    last_i = 0
    for p in life["plans"]:
        tgt = p["target"]
        if tgt not in ("food", "water"):
            continue
        if (last_tgt is not None and tgt != last_tgt
                and not any(last_i < ci <= p["i"] for ci in consume_is)):
            out["flips"] += 1
        last_tgt, last_i = tgt, p["i"]
    # Outcome.
    if h <= DEATH_THR:
        out["class"] = "danger"
        return out
    if e > DEATH_THR and t > DEATH_THR:
        out["class"] = "tronquee"
        return out
    dying = "energy" if e <= t else "thirst"
    needed = NEEDED_RES[dying]
    drive_idx = 0 if dying == "energy" else 1
    # Camping: consumed the OTHER type in the final window while the dying drive was already low.
    other = "water" if needed == "food" else "food"
    for ci, ctyp in life["consume_ticks"]:
        if ctyp == other and ci >= n - CAMP_WINDOW and life["drives"][ci][drive_idx] < LOW_DRIVE:
            out["camp"] = True
            break
    # Last chance: latest replan where the needed resource was seen AND reachable.
    last_chance = None
    ever_seen = False
    for p in life["plans"]:
        if needed in p:
            ever_seen = True
            reserve = life["drives"][p["i"]][drive_idx]
            reach_m = reserve * (SPEED_M_PER_TICK / drains[drive_idx]) * reach_factor
            if p[needed] <= reach_m:
                last_chance = p
    if not ever_seen:
        out["class"] = "jamais_vue"
    elif last_chance is None:
        out["class"] = "metabolique_vue"      # seen, but never within metabolic reach
    elif last_chance["target"] == needed:
        out["class"] = "poursuite_echouee"    # right choice at the last useful replan; execution lost
    elif any(p["target"] == needed and p["i"] > last_chance["i"] for p in life["plans"]):
        out["class"] = "arbitrage_tardif"     # switched to it only AFTER the last chance
    else:
        out["class"] = "arbitrage_jamais"     # seen+reachable, never chosen from there on
    # Dilemma control (§2): other drive's level at death — high = a true arbitration miss,
    # low = maybe an unwinnable dilemma (saving one drive kills the other).
    out["other_at_death"] = (t if dying == "energy" else e)
    return out


def selfcheck(runs: list[str]) -> None:
    sys.path.insert(0, "diagnostics")
    from diag_hazard_gate import parse_lives  # cross-check life split against godot.log
    for run in runs[:2]:
        lives = split_lives(run)
        assert lives, f"no lives parsed in {run}"
        for lf in lives:
            for (e, t, h) in lf["drives"]:
                assert -1.0 <= e <= 101.0 and -1.0 <= t <= 101.0, "drive out of [0,100]"
            for p in lf["plans"]:
                assert p["target"] in ("food", "water", "none", "guard"), p["target"]
    # Life-count cross-check on the first run that has a godot.log.
        glog = os.path.join(run, "godot.log")
        n_ref = len(parse_lives(glog)) if os.path.exists(glog) else None
        print(f"[selfcheck] {os.path.basename(run)}: vies BC={len(lives)} vs godot.log={n_ref}")
        if n_ref is not None:
            assert abs(len(lives) - n_ref) <= 2, "life split inconsistent with godot.log"
    print("[selfcheck] OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    runs = [r for r in args.runs if glob.glob(os.path.join(r, "ep_*.jsonl*"))]
    missing = [r for r in args.runs if r not in runs]
    if missing:
        print(f"[warn] runs absents ignorés: {missing}")
    if not runs:
        raise SystemExit("aucun run avec ep_*.jsonl trouvé")
    if args.selfcheck:
        selfcheck(runs)
        return

    for factor, tag in ((1.0, "OPTIMISTE ligne-droite"), (0.5, "CONSERVATEUR (errance x2)")):
        pooled = {c: 0 for c in CLASSES}
        n_lives = n_camp = n_flips = 0
        per_run = []
        arb_other: list[float] = []
        for run in runs:
            lives = split_lives(run)
            counts = {c: 0 for c in CLASSES}
            camp = 0
            for lf in lives:
                cl = classify_life(lf, factor)
                counts[cl["class"]] += 1
                camp += int(cl["camp"])
                n_flips += cl["flips"]
                if cl["class"].startswith("arbitrage"):
                    arb_other.append(cl["other_at_death"])
            per_run.append((os.path.basename(run), len(lives), counts, camp))
            for c in CLASSES:
                pooled[c] += counts[c]
            n_lives += len(lives)
            n_camp += camp
        arb = pooled["arbitrage_jamais"] + pooled["arbitrage_tardif"]
        arb24 = arb * 24.0 / max(n_lives, 1)
        print(f"\n===== Atteignabilité {tag} (facteur {factor}) — {n_lives} vies, {len(runs)} runs =====")
        print(f"{'run':22s} {'vies':>4s} " + " ".join(f"{c[:9]:>9s}" for c in CLASSES) + f" {'camp':>4s}")
        for name, nl, counts, camp in per_run:
            print(f"{name:22s} {nl:4d} " + " ".join(f"{counts[c]:9d}" for c in CLASSES) + f" {camp:4d}")
        print(f"{'POOLÉ':22s} {n_lives:4d} " + " ".join(f"{pooled[c]:9d}" for c in CLASSES) + f" {n_camp:4d}")
        print(f"  bascules sans consommation (flottement, total) : {n_flips}")
        print("  --- BUT vs proxy ---")
        print(f"  BUT   : morts-par-arbitrage /24 vies = {arb24:.1f}  (barre du bruit = 5.0)")
        print(f"          (jamais-basculé {pooled['arbitrage_jamais']}, bascule-tardive {pooled['arbitrage_tardif']}, campements {n_camp})")
        if arb_other:
            srt = sorted(arb_other)
            q = lambda p: srt[int(p * (len(srt) - 1))]
            frac_hi = sum(1 for v in arb_other if v > 40.0) / len(arb_other)
            print(f"          contrôle dilemme : AUTRE drive à la mort q1/méd/q3 = "
                  f"{q(0.25):.0f}/{q(0.5):.0f}/{q(0.75):.0f} ; part > 40 = {frac_hi:.0%} "
                  f"(haut = vrai raté d'arbitrage, bas = dilemme possible)")
        print(f"  proxy : morts totales = {sum(pooled[c] for c in CLASSES if c != 'tronquee')} "
              f"(dominées par métabolique/perceptuel/danger = PAS la place de l'arbitre)")

    print("\nInterprétation (pré-enregistrée, docs/design_critique_arbitrage.md §G0) :")
    print("  place CONSERVATRICE > 5/24 vies -> chantier licencié ; la répartition")
    print("  jamais-basculé/tardif/campement tranche la voie (A liens-cible / C bascule).")
    print("  place OPTIMISTE <= 5/24 vies -> STOP chantier, négatif commité (le goulot")
    print("  du monde courant n'est pas l'arbitrage de cible).")


if __name__ == "__main__":
    main()
