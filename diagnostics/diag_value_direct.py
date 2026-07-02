"""GATE DÉCISIF (🅑-pur) — le coût-VALEUR latent mène-t-il vers la bouffe ? (test direct sur latents RÊVÉS)

L'énergie a échoué ICI : argmax(énergie) ≈ hasard vs bouffe. On refait le MÊME test avec la tête de valeur :
on roule tous les candidats, on score chacun par V(latents rêvés) (max/mean/endpoint sur le rollout), et on
regarde si le candidat argmax-V s'approche de la bouffe (min_dist << médiane, rang<0.35). Régime AFFAMÉ (e0 bas)
= là où 'repas imminent' a un sens. SUCCÈS → brancher le coût-latent dans le planner. KILL → V sur latents rêvés
ne transfère pas (entraîner sur latents rêvés, ou enrichir le WM).
Usage: SYLVAN_WM_USE_RETINA=1 PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_value_direct.py [wm] [value_head]
"""
import sys, json, glob, math, statistics, os
import torch
from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE
from sylvan.models.value_head import load_value_head
from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig

WM = sys.argv[1] if len(sys.argv) > 1 else "data/checkpoints/wm_command_hex_retina_eat_v1/wm_best.pt"
VH = sys.argv[2] if len(sys.argv) > 2 else "data/checkpoints/value_head_food/value_best.pt"
H = 120
E0S = [0.4, 0.7]      # régime affamé (0.4) + plein (0.7) ; la valeur a du sens quand on PEUT manger
N_SAMP = 200
# Bande de distance courante de la bouffe (mètres). DÉFAUT = medium/far (1.5-4.0, l'engagement).
# Pour tester le CLOSE (le bug) : SYLVAN_DIAG_DMIN=0.8 SYLVAN_DIAG_DMAX=1.6 → rang au close.
DMIN = float(os.environ.get("SYLVAN_DIAG_DMIN", "1.5"))
DMAX = float(os.environ.get("SYLVAN_DIAG_DMAX", "4.0"))

pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
head = load_value_head(VH)
planner = CommandPlanner(wm, CommandPlanConfig(horizon=H)); SEQS = planner._cmd_seqs; N = SEQS.shape[0]
print(f"WM={WM} | value={VH} (AUC {torch.load(VH,map_location='cpu',weights_only=False).get('auc_heldout','?')}) | {N} candidats × H={H}")

files = sorted(glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl"))


def gen():
    for f in files:
        for line in open(f):
            r = json.loads(line); w = r.get("wm", {})
            ret = w.get("retina0"); fr = w.get("food_rel0")
            if not ret or not fr or fr[2] < 0.5:
                continue
            d = math.hypot(fr[0], fr[1])
            if DMIN <= d <= DMAX:
                yield r["obs"]["proprio"], ret, fr[0], fr[1]


def integrate_min(disp, fx, fz):
    x = z = yaw = 0.0; md = float("inf"); mins = torch.zeros(N)
    X = torch.zeros(N); Z = torch.zeros(N); Y = torch.zeros(N)
    mind = torch.full((N,), float("inf"))
    for t in range(H):
        d_fwd, d_lat, d_yaw = disp[:, t, 0], disp[:, t, 1], disp[:, t, 2]
        s, c = torch.sin(Y), torch.cos(Y)
        X = X + d_fwd * s + d_lat * c; Z = Z + d_fwd * c - d_lat * s; Y = Y + d_yaw
        mind = torch.minimum(mind, torch.sqrt((X - fx) ** 2 + (Z - fz) ** 2))
    return mind


def directstats(score, mind):
    best = int(torch.argmax(score)); cm = float(mind[best]); md = float(mind.median())
    rk = float((mind < cm).float().mean()); rch = 1.0 if cm < 1.0 else 0.0
    return cm, md, rk, rch


def corr(a, b):
    da, db = a - a.mean(), b - b.mean()
    d = (da.norm() * db.norm()).item()
    return (da @ db).item() / d if d > 1e-8 else float("nan")


for e0 in E0S:
    acc = {k: {"cm": [], "md": [], "rk": [], "rch": [], "corr": []} for k in ("Vmax", "Vmean", "Vend", "energy")}
    corr_tf = []   # corr(V teacher-forced @t0, -min_dist) — le latent RÉEL porte-t-il l'info à travers les candidats ?
    n = 0
    for proprio, ret, fx, fz in gen():
        obs = torch.tensor(proprio + ret + [e0], dtype=torch.float32)
        with torch.no_grad():
            out = wm.rollout_open_loop(obs.reshape(1, -1).expand(N, -1).contiguous(), SEQS)
            V = head.value(out["predicted_latents"])           # [N,H]
        disp = out["predicted_displacement"] / DISPLACEMENT_SCALE
        epred = out["predicted_next_obs"][..., -1]
        mind = integrate_min(disp, fx, fz)
        for key, sc in (("Vmax", V.max(1).values), ("Vmean", V.mean(1)),
                        ("Vend", V[:, -1]), ("energy", epred[:, -1])):
            cm, md, rk, rch = directstats(sc, mind)
            acc[key]["cm"].append(cm); acc[key]["md"].append(md); acc[key]["rk"].append(rk)
            acc[key]["rch"].append(rch); acc[key]["corr"].append(corr(sc, -mind))
        n += 1
        if n >= N_SAMP:
            break
    print(f"\n=== e0={e0} (n={n}) — l'argmax du score va-t-il vers la bouffe ? ===")
    print(f"{'score':>8} | {'min_d choisi':>12} | {'médiane':>9} | {'rang[0=près]':>12} | {'atteint<1m':>11} | {'corr(score,approche)':>20}")
    for key in ("Vmax", "Vmean", "Vend", "energy"):
        cm = statistics.mean(acc[key]["cm"]); md = statistics.mean(acc[key]["md"])
        rk = statistics.mean(acc[key]["rk"]); rch = statistics.mean(acc[key]["rch"])
        cr = statistics.mean([c for c in acc[key]["corr"] if not math.isnan(c)])
        ok = "✅" if (cm < md - 0.3 and rk < 0.35) else ""
        print(f"{key:>8} | {cm:>12.3f} | {md:>9.3f} | {rk:>12.2f} | {rch:>11.0%} | {cr:>+20.3f}  {ok}")

print("\n(✅ = l'argmax s'approche nettement mieux que la médiane ET rang<0.35 → coût-latent exploitable)")
print("Comparer V* à 'energy' : si V passe et energy non → la tête de valeur EST le bon instrument 🅑-pur.")
