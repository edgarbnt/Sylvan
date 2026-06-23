"""GATE du déclencheur CHERCHER (offline, 0 entraînement).

Le primitive de perception active se déclenche quand « rien d'engageant n'est perçu » = la VALUE latente du
MEILLEUR candidat est plate/basse. Ce gate vérifie que ce signal (engage) SÉPARE proprement les situations
« approcher » (bouffe perçue à l'avant) des situations « chercher » (bouffe derrière / hors-champ), et fixe τ.

engage_max    = max_candidats max_horizon  sigmoid(V)   (pic de valeur n'importe où dans le rêve)
engage_chosen = sigmoid(mean_horizon V) du candidat argmax (ce que le planner COMMIT, agg=mean comme le live)
SUCCÈS : front-proche engage ≫ τ ≫ derrière engage, avec un trou net → τ fiable.

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_search_trigger.py
"""
import json, glob, math, os, statistics as st
import torch
from sylvan.models.command_wm import CommandWorldModel
from sylvan.models.value_head import load_value_head
from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig

WM = "data/checkpoints/wm_rich_fidele_sym/wm_best.pt"
VH = "data/checkpoints/value_head_food_dream/value_best.pt"
H = int(os.environ.get("H", "120")); CAP = int(os.environ.get("CAP", "500"))
pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
val = load_value_head(VH)
planner = CommandPlanner(wm, CommandPlanConfig(horizon=H)); SEQS = planner._cmd_seqs; N = SEQS.shape[0]
files = sorted(glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl"))

BINS = {"front_close": [], "front_mid": [], "front_far": [], "side": [], "behind": []}


def classify(fx, fz):
    d = math.hypot(fx, fz); brg = abs(math.degrees(math.atan2(fx, fz)))
    if brg > 120:
        return "behind"
    if 60 <= brg <= 120:
        return "side"
    if d < 1.5:
        return "front_close"
    if d < 3.0:
        return "front_mid"
    return "front_far"


n = 0
for f in files:
    for line in open(f):
        r = json.loads(line); w = r.get("wm", {})
        ret = w.get("retina0"); fr = w.get("food_rel0")
        if not ret or not fr or fr[2] < 0.5:
            continue
        fx, fz = fr[0], fr[1]
        obs = torch.tensor(r["obs"]["proprio"] + ret + [0.4], dtype=torch.float32)
        with torch.no_grad():
            out = wm.rollout_open_loop(obs.reshape(1, -1).expand(N, -1).contiguous(), SEQS)
            V = torch.sigmoid(val.logit(out["predicted_latents"]))   # [N,H] proba
        emax = float(V.max())
        Lmean = V.mean(1)                                            # [N]  (agg mean = live)
        echosen = float(Lmean.max())
        BINS[classify(fx, fz)].append((emax, echosen))
        n += 1
    if n >= CAP:
        break


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs); k = max(0, min(len(xs) - 1, int(p / 100 * (len(xs) - 1))))
    return xs[k]


print(f"frames={n}  (engage = à quel point le MEILLEUR futur imaginé est 'repas')\n")
print(f"{'situation':>12} | {'n':>4} | {'engage_max p10/med/p90':>26} | {'engage_chosen p10/med/p90':>26}")
for b in ("front_close", "front_mid", "front_far", "side", "behind"):
    xs = BINS[b]
    em = [a for a, _ in xs]; ec = [c for _, c in xs]
    print(f"{b:>12} | {len(xs):>4} | {pct(em,10):>7.3f}/{pct(em,50):.3f}/{pct(em,90):.3f}      | "
          f"{pct(ec,10):>7.3f}/{pct(ec,50):.3f}/{pct(ec,90):.3f}")

beh = [a for a, _ in BINS["behind"]]; frc = [a for a, _ in BINS["front_close"]]
if beh and frc:
    tau = (pct(beh, 90) + pct(frc, 10)) / 2
    gap = pct(frc, 10) - pct(beh, 90)
    print(f"\nengage_max : derrière p90={pct(beh,90):.3f}  <  front-proche p10={pct(frc,10):.3f}  → τ≈{tau:.3f} "
          f"(trou={'NET ✅' if gap > 0 else 'CHEVAUCHE ❌'} {gap:+.3f})")
print("Lecture: si front-proche ≫ derrière avec un trou net → le déclencheur 'value plate→chercher' est fiable.")
