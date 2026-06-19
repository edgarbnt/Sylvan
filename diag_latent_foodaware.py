"""Test GRATUIT 🅑 — l'énergie prédite par le WM-rétine est-elle FOOD-AWARE, et le signal
GRANDIT-il avec l'horizon / à basse énergie initiale ?

Prérequis du planning latent : le coût latent (= énergie/inconfort futur prédit) ne marche que si le WM
prédit PLUS d'énergie pour des commandes qui MÈNENT à la bouffe. Pour chaque échantillon (food visible)
on roule TOUS les candidats du planner dans le WM, on intègre le déplacement imaginé → distance MIN
atteinte à la bouffe, et on lit le gain d'énergie prédit ΔE. On classe les candidats en PROCHES (quartile
qui s'approche le + de la bouffe) vs LOIN, et on regarde l'écart ΔE_near − ΔE_far (POSITIF = food-aware) +
la corrélation (min_dist, ΔE) (NÉGATIF = food-aware).

DEUX BALAYAGES (le coeur de l'extension 🅑) :
  • HORIZON {50,80,120,150} — l'écart proche−loin GRANDIT-il quand on imagine plus loin ?
    (hypothèse : à H court l'énergie draine plus vite qu'elle ne remonte → écart faible ; à H long la
     remontée post-repas se voit → écart plus net → un coût PUR-énergie devient discriminant).
  • ÉNERGIE INITIALE e0 {0.3,0.5,0.7,0.9} (override de obs[-1], à H=120) — l'écart est-il PLUS GRAND à
    basse énergie ? (les données ont été collectées à e0=1.0 = plein → manger ne donne aucune marge ;
    à e0 bas, le repas remonte vraiment → c'est là que le pur-latent devrait le mieux marcher).

Usage: SYLVAN_WM_USE_RETINA=1 PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_latent_foodaware.py [wm_ckpt]
"""
import sys, json, glob, math, statistics
import torch
from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE
from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig

WM = sys.argv[1] if len(sys.argv) > 1 else "data/checkpoints/wm_command_hex_retina_jepa_v2/wm_best.pt"
HMAX = 150
HORIZONS = [50, 80, 120, 150]
ENERGIES = [0.3, 0.5, 0.7, 0.9]
E_SWEEP_H = 120
N_HORIZON = 200      # échantillons pour le balayage horizon (1 rollout/échantillon à HMAX)
N_ESWEEP = 120       # échantillons pour le balayage énergie (×4 e0)

pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
planner = CommandPlanner(wm, CommandPlanConfig(horizon=HMAX))
SEQS = planner._cmd_seqs            # [N,HMAX,2]
N = SEQS.shape[0]
print(f"WM={WM} obs_dim={meta['obs_dim']} | {N} candidats × HMAX={HMAX}")

files = sorted(glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl"))


def _samples(limit):
    """Yield (proprio, retina, fx, fz, e0_data) for food clearly visible at 1.5-4 m."""
    n = 0
    for f in files:
        for line in open(f):
            r = json.loads(line); w = r.get("wm", {})
            ret = w.get("retina0"); fr = w.get("food_rel0")
            if not ret or not fr or fr[2] < 0.5:
                continue
            fx, fz = fr[0], fr[1]
            if math.hypot(fx, fz) < 1.5 or math.hypot(fx, fz) > 4.0:
                continue
            yield r["obs"]["proprio"], ret, fx, fz, r["obs"]["energy"] / 100.0
            n += 1
            if n >= limit:
                return


@torch.no_grad()
def rollout(proprio, ret, e0, horizon):
    """Roll all candidates to `horizon`. Return per-step min_dist-to-food and ΔE arrays.
    Trajectory integrated in the body frame exactly as the planner does."""
    obs = torch.tensor(proprio + ret + [e0], dtype=torch.float32)
    seqs = SEQS[:, :horizon, :].contiguous()
    out = wm.rollout_open_loop(obs.reshape(1, -1).expand(N, -1).contiguous(), seqs)
    disp = out["predicted_displacement"] / DISPLACEMENT_SCALE      # [N,H,3]
    epred = out["predicted_next_obs"][..., -1].clamp(0.0, 1.0)     # [N,H]
    return disp, epred


def integrate(disp, fx, fz):
    """[N,H,3] body-frame disp → (running MIN distance, instantaneous distance) per step, each [N,H]."""
    N_, H_ = disp.shape[0], disp.shape[1]
    x = torch.zeros(N_); z = torch.zeros(N_); yaw = torch.zeros(N_)
    min_dist = torch.full((N_,), float("inf"))
    mins = torch.zeros(N_, H_); inst = torch.zeros(N_, H_)
    for t in range(H_):
        d_fwd, d_lat, d_yaw = disp[:, t, 0], disp[:, t, 1], disp[:, t, 2]
        s, c = torch.sin(yaw), torch.cos(yaw)
        x = x + d_fwd * s + d_lat * c
        z = z + d_fwd * c - d_lat * s
        yaw = yaw + d_yaw
        dist = torch.sqrt((x - fx) ** 2 + (z - fz) ** 2)
        min_dist = torch.minimum(min_dist, dist)
        mins[:, t] = min_dist; inst[:, t] = dist
    return mins, inst  # [N,H], [N,H]


def _corr(a, b):
    da, db = a - a.mean(), b - b.mean()
    denom = (da.norm() * db.norm()).item()
    return (da @ db).item() / denom if denom > 1e-8 else float("nan")


def stats(min_dist_vec, dE_vec):
    """corr(min_dist, ΔE) + ΔE_near − ΔE_far (quartile le + proche vs le + loin)."""
    corr = _corr(min_dist_vec, dE_vec)
    k = max(1, N // 4)
    near = torch.topk(-min_dist_vec, k).indices
    far = torch.topk(min_dist_vec, k).indices
    return corr, dE_vec[near].mean().item(), dE_vec[far].mean().item()


def direct_test(dE_vec, min_dist_vec, inst_vec):
    """LE test décisif : le candidat que le coût latent CHOISIRAIT (argmax énergie prédite) va-t-il vers
    la bouffe ? Retourne (min_dist du choisi, meilleur min_dist possible, médiane min_dist, rang∈[0,1]
    du choisi parmi tous par approche, atteint<1m?, corr(dist_finale,ΔE))."""
    best = int(torch.argmax(dE_vec).item())
    chosen_min = float(min_dist_vec[best])
    best_possible = float(min_dist_vec.min())
    median_min = float(min_dist_vec.median())
    rank = float((min_dist_vec < chosen_min).float().mean())     # 0 = le + proche, 1 = le + loin
    reached = 1.0 if chosen_min < 1.0 else 0.0
    corr_final = _corr(inst_vec, dE_vec)                          # métrique ORIGINALE (dist finale)
    return chosen_min, best_possible, median_min, rank, reached, corr_final


# ───────────────────────── BALAYAGE 1 : HORIZON (e0 = donnée) ─────────────────────────
print("\n=== BALAYAGE HORIZON (e0 = valeur de la donnée ; food visible 1.5-4 m) ===")
acc = {h: {"corr": [], "near": [], "far": [], "corr_final": [],
           "chosen": [], "best": [], "median": [], "rank": [], "reached": []} for h in HORIZONS}
n = 0
for proprio, ret, fx, fz, e0 in _samples(N_HORIZON):
    disp, epred = rollout(proprio, ret, e0, HMAX)
    mins, inst = integrate(disp, fx, fz)            # [N,HMAX], [N,HMAX]
    for h in HORIZONS:
        t = h - 1
        dE = epred[:, t] - e0
        corr, dn, df = stats(mins[:, t], dE)
        if not math.isnan(corr):
            acc[h]["corr"].append(corr)
        acc[h]["near"].append(dn); acc[h]["far"].append(df)
        cm, bp, md, rk, rch, cf = direct_test(dE, mins[:, t], inst[:, t])
        acc[h]["chosen"].append(cm); acc[h]["best"].append(bp); acc[h]["median"].append(md)
        acc[h]["rank"].append(rk); acc[h]["reached"].append(rch)
        if not math.isnan(cf):
            acc[h]["corr_final"].append(cf)
    n += 1
print(f"échantillons={n}  (e0 moyen des données ≈ plein)")
print(f"{'H':>5} | {'corr(min_d,ΔE)':>15} | {'corr(distFin,ΔE)':>16} | {'ΔE_near':>9} | {'ΔE_far':>9} | {'écart':>9}")
prev_gap = None
gaps = {}
for h in HORIZONS:
    corr = statistics.mean(acc[h]["corr"]); cf = statistics.mean(acc[h]["corr_final"])
    dn = statistics.mean(acc[h]["near"]); df = statistics.mean(acc[h]["far"])
    gap = dn - df; gaps[h] = gap
    arrow = "" if prev_gap is None else ("↑" if gap > prev_gap + 1e-5 else ("↓" if gap < prev_gap - 1e-5 else "="))
    print(f"{h:>5} | {corr:>+15.3f} | {cf:>+16.3f} | {dn:>+9.4f} | {df:>+9.4f} | {gap:>+9.4f} {arrow}")
    prev_gap = gap
print(f"\n--- TEST DIRECT : le candidat argmax(énergie prédite) va-t-il vers la bouffe ? ---")
print(f"{'H':>5} | {'min_d choisi':>12} | {'meilleur poss.':>14} | {'médiane':>9} | {'rang[0=près]':>12} | {'atteint<1m':>11}")
direct_ok = {}
for h in HORIZONS:
    cm = statistics.mean(acc[h]["chosen"]); bp = statistics.mean(acc[h]["best"])
    md = statistics.mean(acc[h]["median"]); rk = statistics.mean(acc[h]["rank"]); rch = statistics.mean(acc[h]["reached"])
    # le choix-énergie est "food-aware" si le candidat choisi s'approche nettement mieux que la médiane
    direct_ok[h] = (cm < md - 0.3) and (rk < 0.35)
    print(f"{h:>5} | {cm:>12.3f} | {bp:>14.3f} | {md:>9.3f} | {rk:>12.2f} | {rch:>11.0%}")
grows = all(gaps[HORIZONS[i]] >= gaps[HORIZONS[i-1]] - 1e-4 for i in range(1, len(HORIZONS))) and gaps[HORIZONS[-1]] > gaps[HORIZONS[0]] + 1e-4

# ───────────────────────── BALAYAGE 2 : ÉNERGIE INITIALE (H=120) ─────────────────────────
print(f"\n=== BALAYAGE ÉNERGIE INITIALE (override e0, H={E_SWEEP_H} ; food visible 1.5-4 m) ===")
eacc = {e: {"corr": [], "near": [], "far": [], "chosen": [], "median": [], "rank": [], "reached": []} for e in ENERGIES}
n = 0
samples = list(_samples(N_ESWEEP))
for proprio, ret, fx, fz, _e0 in samples:
    for e0 in ENERGIES:
        disp, epred = rollout(proprio, ret, e0, E_SWEEP_H)
        mins, inst = integrate(disp, fx, fz)
        dE = epred[:, -1] - e0
        corr, dn, df = stats(mins[:, -1], dE)
        if not math.isnan(corr):
            eacc[e0]["corr"].append(corr)
        eacc[e0]["near"].append(dn); eacc[e0]["far"].append(df)
        cm, _bp, md, rk, rch, _cf = direct_test(dE, mins[:, -1], inst[:, -1])
        eacc[e0]["chosen"].append(cm); eacc[e0]["median"].append(md)
        eacc[e0]["rank"].append(rk); eacc[e0]["reached"].append(rch)
    n += 1
print(f"échantillons={n}")
print(f"{'e0':>5} | {'corr(min_d,ΔE)':>16} | {'ΔE_near':>9} | {'ΔE_far':>9} | {'écart(near-far)':>16}")
egaps = {}
for e0 in ENERGIES:
    corr = statistics.mean(eacc[e0]["corr"]); dn = statistics.mean(eacc[e0]["near"]); df = statistics.mean(eacc[e0]["far"])
    egaps[e0] = dn - df
    print(f"{e0:>5.1f} | {corr:>+16.3f} | {dn:>+9.4f} | {df:>+9.4f} | {egaps[e0]:>+16.4f}")
low_helps = egaps[0.3] > 2.0 * max(egaps[0.9], 1e-6) or (egaps[0.3] > egaps[0.9] and statistics.mean(eacc[0.3]["near"]) > 0)
print(f"\n--- TEST DIRECT par énergie initiale (H={E_SWEEP_H}) : l'argmax-énergie va-t-il vers la bouffe quand on a FAIM ? ---")
print(f"{'e0':>5} | {'min_d choisi':>12} | {'médiane':>9} | {'rang[0=près]':>12} | {'atteint<1m':>11}")
edirect_ok = {}
for e0 in ENERGIES:
    cm = statistics.mean(eacc[e0]["chosen"]); md = statistics.mean(eacc[e0]["median"])
    rk = statistics.mean(eacc[e0]["rank"]); rch = statistics.mean(eacc[e0]["reached"])
    edirect_ok[e0] = (cm < md - 0.3) and (rk < 0.35)
    print(f"{e0:>5.1f} | {cm:>12.3f} | {md:>9.3f} | {rk:>12.2f} | {rch:>11.0%}  {'✅' if edirect_ok[e0] else ''}")
low_direct_ok = edirect_ok[0.3] or edirect_ok[0.5]

# ───────────────────────── PROBE 3 : FIDÉLITÉ EAT-DYNAMICS (1 pas, teacher-forced) ─────────────────────────
# Substantie la cause-racine : le WM prédit-il la BOSSE d'énergie quand on mange ? Pour chaque transition,
# 1 pas du WM (obs réelle + commande réelle) → ΔE prédit vs ΔE réel, séparé eat vs non-eat.
print("\n=== PROBE FIDÉLITÉ EAT (1 pas teacher-forced : le WM voit-il manger ?) ===")
eat_pred, eat_real, non_pred, non_real = [], [], [], []
ne = 0
for f in files:
    for line in open(f):
        r = json.loads(line); w = r.get("wm", {})
        ret = w.get("retina0"); cmd = w.get("cmd")
        if not ret or not cmd:
            continue
        e0 = r["obs"]["energy"] / 100.0
        e1 = r["next_obs"]["energy"] / 100.0
        obs = torch.tensor(r["obs"]["proprio"] + ret + [e0], dtype=torch.float32)
        cmds = torch.tensor(cmd[:2], dtype=torch.float32).reshape(1, 1, 2)
        with torch.no_grad():
            out = wm.rollout_open_loop(obs.reshape(1, -1), cmds)
        ep = float(out["predicted_next_obs"][0, 0, -1].clamp(0.0, 1.0)) - e0
        if w.get("ate"):
            eat_pred.append(ep); eat_real.append(e1 - e0)
        else:
            non_pred.append(ep); non_real.append(e1 - e0)
        ne += 1
    if len(eat_real) >= 60 and ne >= 4000:
        break
if eat_real:
    print(f"transitions: eat={len(eat_real)} non-eat={len(non_pred)}")
    print(f"EAT     : ΔE RÉEL moyen = {statistics.mean(eat_real):+.4f} | ΔE PRÉDIT moyen = {statistics.mean(eat_pred):+.4f}")
    print(f"NON-EAT : ΔE RÉEL moyen = {statistics.mean(non_real):+.4f} | ΔE PRÉDIT moyen = {statistics.mean(non_pred):+.4f}")
    pred_sep = statistics.mean(eat_pred) - statistics.mean(non_pred)
    real_sep = statistics.mean(eat_real) - statistics.mean(non_real)
    print(f"séparation eat−noneat : RÉELLE = {real_sep:+.4f} | PRÉDITE = {pred_sep:+.4f} "
          f"→ le WM capte {100*pred_sep/real_sep if real_sep>1e-6 else 0:.0f}% de la bosse")
else:
    print("aucune transition 'ate' dans les fichiers (manger non loggué) — probe sautée")

# ───────────────────────── VERDICT ─────────────────────────
print("\n=== VERDICT 🅑 ===")
# Le TEST DIRECT prime : peu importe la corrélation, est-ce que le candidat que le coût latent
# CHOISIT (max énergie prédite) s'approche de la bouffe mieux que le hasard (médiane) ?
direct_any = any(direct_ok[h] for h in HORIZONS)
direct_all = all(direct_ok[h] for h in HORIZONS)
print(f"TEST DIRECT food-aware (choix-énergie s'approche < médiane-0.3 ET rang<0.35) : "
      f"{'tous H' if direct_all else ('certains H' if direct_any else 'AUCUN H')}")
print(f"écart(near-far) > 0 sur tous les horizons : {all(gaps[h] > 0 for h in HORIZONS)}")
print(f"écart CROÎT avec l'horizon : {grows}")
print(f"basse énergie AIDE (écart e0=0.3 ≥ 2× e0=0.9, ou ΔE_near>0 à e0=0.3) : {low_helps}")
if not direct_any:
    print("🔴 KILL pur-latent : le candidat max-énergie NE va PAS vers la bouffe mieux que le hasard")
    print("   → un coût PUR énergie-future ferait errer l'agent. Le WM doit mieux prédire l'eat-dynamics")
    print("   AVANT 🅑. NE PAS gonfler ; coût HYBRIDE à ancre DOMINANTE seulement, ou escalader.")
elif direct_all and (grows or low_helps):
    print("🟢 PUR-🅑 PLAUSIBLE : le choix-énergie va vers la bouffe à tous les horizons + signal se renforce.")
    print("   → hybride puis pousser l'ancre → 0.")
else:
    print("🟡 HYBRIDE : food-aware partiel (certains horizons), pas robuste.")
    print("   → coût hybride avec ancre résiduelle au bon horizon ; ne pas espérer le pur-latent tout de suite.")
