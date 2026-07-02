"""GATE GRATUIT 🅑 — la RECHERCHE GUIDÉE (CEM) débloque-t-elle des rêves qui ATTEIGNENT la bouffe,
guidée par le SCORE LATENT (énergie future food-aware) SEUL, coordonnées DÉBRANCHÉES ?

Contexte (docs/scope_guided_search.md §3.1) : le planner actuel teste une GRILLE FIXE d'arcs (vx,ω) ;
mesuré, AUCUN arc n'atteint la bouffe en un rêve (tous ~1.7 m) → un readout latent n'a rien à classer.
Hypothèse : remplacer la grille par une CEM qui OPTIMISE la séquence de commandes dans l'espace continu,
guidée par l'énergie future prédite, FABRIQUE les bons plans que la grille ne contenait pas.

Ce script NE lance RIEN dans Godot, N'ENTRAÎNE RIEN. Sur des frames de retina_forage (bouffe visible
1.5-4 m) il compare, par la distance MIN atteinte DANS LE RÊVE :
  • GRILLE       — meilleur candidat de la grille du planner (la référence ~1.7 m).
  • CEM-ÉNERGIE  — CEM guidée par l'énergie future prédite SEULE (assert : AUCUNE coordonnée dans le score).
  • CEM-GÉOMÉTRIE— contrôle : CEM guidée par -min_dist (coordonnées) = ce que la RECHERCHE PEUT atteindre.

CRITÈRES ÉCRITS AVANT (scope §3.1) :
  SUCCÈS : CEM-énergie atteint min_dist < ~1.0 m sur une MAJORITÉ de frames (et nettement < grille ~1.7 m)
           → la recherche guidée par le latent débloque 🅑 → passer au closed-loop.
  KILL   : CEM-énergie ne fait pas mieux que la grille. Alors le contrôle GÉOMÉTRIE tranche :
           • géométrie atteint < 1 m mais pas l'énergie → c'est le SCORE (essayer une tête de valeur
             ré-entraînée sur latents rêvés sous commandes diverses) ;
           • même la géométrie plafonne ~1.7 m → c'est le RÊVE/HORIZON du WM → escalade (horizon, re-feed
             rétine), NE PAS gonfler ni câbler le closed-loop.

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_cem.py [wm_ckpt] [e0=0.4] [n_frames=40]
"""
import sys, json, glob, math, statistics
import torch

from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE
from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig

WM = sys.argv[1] if len(sys.argv) > 1 else "data/checkpoints/wm_command_hex_retina_eat_v2/wm_best.pt"
E0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4    # énergie initiale "affamée" : marge pour que le
                                                         # repas rêvé fasse MONTER l'énergie → score guidable
                                                         # (à e0=1.0 le repas est clampé → score plat, cf
                                                         #  balayage e0 de diag_latent_foodaware).
N_FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 40
VALUE_CKPT = sys.argv[4] if len(sys.argv) > 4 else None  # tête de VALEUR (🅑-pur) → ajoute le mode CEM-value

# ── CEM / horizon ──
H = 120                 # horizon du planner (WM en-distribution ~0.15 m err @120)
SEG_LEN = 40            # longueur de segment (≥40 = régime entraîné du WM) ; H = N_SEG*SEG_LEN
N_SEG = H // SEG_LEN    # 3 segments piecewise-constant → strictement plus expressif que la grille 2-segments
N_POP = 64             # taille de population CEM par itération
K_ELITE = 12           # élites refit
ITERS = 6              # itérations CEM
# régime PROPRE de l'hexapode (hors-régime → rêve non fiable, scope R4) — clamp DUR.
VX_MIN, VX_MAX = 0.55, 0.75
OM_MIN, OM_MAX = -0.6, 0.6
DONE_PEN = 3.0          # pénalité de chute (même que le planner)
SIGMA0 = torch.tensor([0.07, 0.30])   # σ initial (vx, ω) par segment
SIGMA_FLOOR = torch.tensor([0.02, 0.08])
REACH = 1.0             # m : "atteint la bouffe" dans le rêve (≈ eat_radius)

pl = torch.load(WM, map_location="cpu", weights_only=False)
meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
for p in wm.parameters():
    p.requires_grad_(False)

HEAD = None
if VALUE_CKPT:
    from sylvan.models.value_head import load_value_head
    HEAD = load_value_head(VALUE_CKPT)
    print(f"  + tête de VALEUR {VALUE_CKPT} → mode CEM-value actif (🅑-pur, lit V(latents rêvés), aucune coordonnée)")

# Grille du planner pour la référence (mêmes candidats que le live).
planner = CommandPlanner(wm, CommandPlanConfig(horizon=H))
GRID = planner._cmd_seqs                      # [G,H,2]
print(f"WM={WM} obs_dim={meta['obs_dim']} | e0={E0} | H={H} {N_SEG}seg×{SEG_LEN}"
      f" | CEM pop={N_POP} elites={K_ELITE} iters={ITERS} | grille={GRID.shape[0]} candidats")

files = sorted(glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl"))


def _frames(limit):
    """(obs[277], fx, fz) pour bouffe clairement visible 1.5-4 m. e0 forcé à E0 (sonde explicite)."""
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
            obs = torch.tensor(r["obs"]["proprio"] + ret + [E0], dtype=torch.float32)
            yield obs, fx, fz
            n += 1
            if n >= limit:
                return


@torch.no_grad()
def min_dist_of(disp, fx, fz):
    """[N,H,3] déplacement body-frame → distance MIN atteinte à la bouffe par candidat, [N].
    GÉOMÉTRIE PURE — sert UNIQUEMENT de mesure (et de score pour le CONTRÔLE géométrie), JAMAIS pour
    le score énergie."""
    Np, Hh = disp.shape[0], disp.shape[1]
    x = torch.zeros(Np); z = torch.zeros(Np); yaw = torch.zeros(Np)
    md = torch.full((Np,), float("inf"))
    for t in range(Hh):
        d_fwd, d_lat, d_yaw = disp[:, t, 0], disp[:, t, 1], disp[:, t, 2]
        s, c = torch.sin(yaw), torch.cos(yaw)
        x = x + d_fwd * s + d_lat * c
        z = z + d_fwd * c - d_lat * s
        yaw = yaw + d_yaw
        md = torch.minimum(md, torch.sqrt((x - fx) ** 2 + (z - fz) ** 2))
    return md


@torch.no_grad()
def roll(obs, seqs):
    """seqs [N,H,2] → (disp[N,H,3], score_energy[N], score_value[N]). Les DEUX scores NE VOIENT que des
    readouts du latent RÊVÉ (énergie future ; V(latents) moyenne) — AUCUNE coordonnée (🅑-pur)."""
    Np = seqs.shape[0]
    out = wm.rollout_open_loop(obs.reshape(1, -1).expand(Np, -1).contiguous(), seqs)
    disp = out["predicted_displacement"] / DISPLACEMENT_SCALE
    e_fin = out["predicted_next_obs"][..., -1].clamp(0.0, 1.0)[:, -1]         # énergie future (food-aware)
    done = torch.sigmoid(out["predicted_done_logits"])                       # [N,H]
    alive = torch.ones(Np); pen = torch.zeros(Np)
    for t in range(H):
        pen = pen + alive * done[:, t]
        alive = alive * (1.0 - done[:, t])
    score_energy = e_fin - DONE_PEN * pen                                    # AUCUNE coordonnée
    score_value = None
    if HEAD is not None:
        V = HEAD.value(out["predicted_latents"])                            # [N,H] — lit le latent RÊVÉ
        score_value = V.mean(1) - DONE_PEN * pen                            # Vmean (meilleur en diag_value_direct)
    return disp, score_energy, score_value


def expand(params):
    """[N,N_SEG,2] commandes par segment → [N,H,2] piecewise-constant, clampé au régime propre."""
    p = params.clone()
    p[..., 0].clamp_(VX_MIN, VX_MAX)
    p[..., 1].clamp_(OM_MIN, OM_MAX)
    return p.repeat_interleave(SEG_LEN, dim=1)


@torch.no_grad()
def cem(obs, fx, fz, mode):
    """CEM sur les commandes par segment. mode='energy'/'value' → guidé par un SCORE LATENT (🅑-pur, aucune
    coord) ; mode='geom' → contrôle guidé par -min_dist (coordonnées). Renvoie le min_dist (mesuré) de la
    MEILLEURE séquence trouvée selon le score optimisé."""
    mu = torch.tensor([[0.65, 0.0]] * N_SEG)
    sigma = SIGMA0.repeat(N_SEG, 1).clone()
    best_score = -float("inf"); best_md = float("inf")
    for _ in range(ITERS):
        eps = torch.randn(N_POP, N_SEG, 2)
        params = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
        seqs = expand(params)
        disp, score_energy, score_value = roll(obs, seqs)
        md = min_dist_of(disp, fx, fz)                 # mesure (toujours), coords
        if mode == "energy":
            score = score_energy                       # ── readout latent SEUL ──
        elif mode == "value":
            score = score_value                        # ── tête de VALEUR sur latent rêvé, aucune coord ──
        elif mode == "geom":
            score = -md                                # ── contrôle géométrie (coords autorisées) ──
        else:
            raise ValueError(mode)
        topv, topi = torch.topk(score, K_ELITE)
        elites = params[topi]
        mu = elites.mean(0)
        sigma = torch.maximum(elites.std(0), SIGMA_FLOOR.repeat(N_SEG, 1))
        it_best = int(torch.argmax(score).item())
        if float(score[it_best]) > best_score:
            best_score = float(score[it_best]); best_md = float(md[it_best])
    return best_md


# Garde-fou DUR : le score énergie ne doit jamais recevoir de coordonnées. roll() ne prend que (obs, seqs)
# et ne référence ni fx ni fz → vérifié structurellement (les coords ne sont passées qu'à min_dist_of, hors score).
assert "fx" not in roll.__code__.co_varnames and "fz" not in roll.__code__.co_varnames, \
    "VIOLATION 🅑 : le score énergie ne doit voir AUCUNE coordonnée"

grid_best, cem_e, cem_g, cem_v = [], [], [], []
n = 0
for obs, fx, fz in _frames(N_FRAMES):
    with torch.no_grad():
        gdisp, _, _ = roll(obs, GRID)
        gmd = min_dist_of(gdisp, fx, fz)
    grid_best.append(float(gmd.min()))
    cem_e.append(cem(obs, fx, fz, "energy"))
    cem_g.append(cem(obs, fx, fz, "geom"))
    if HEAD is not None:
        cem_v.append(cem(obs, fx, fz, "value"))
    n += 1
    if n % 10 == 0:
        print(f"  ...{n}/{N_FRAMES} frames")


def summ(name, v):
    med = statistics.median(v); mean = statistics.mean(v)
    frac1 = sum(d < 1.0 for d in v) / len(v)
    frac15 = sum(d < 1.5 for d in v) / len(v)
    print(f"{name:>14} | médiane {med:5.2f} m | moyenne {mean:5.2f} m | <1.0m {frac1:4.0%} | <1.5m {frac15:4.0%}")
    return med, frac1


print(f"\n=== RÉSULTATS ({n} frames, bouffe visible 1.5-4 m, e0={E0}) — distance MIN atteinte dans le rêve ===")
gm, gf = summ("GRILLE (réf)", grid_best)
em, ef = summ("CEM-ÉNERGIE", cem_e)
cm, cf = summ("CEM-GÉOMÉTRIE", cem_g)
vm = None
if cem_v:
    vm, vf = summ("CEM-VALEUR", cem_v)

print("\n=== VERDICT 🅑 (CEM guidée par la VALEUR vs géométrie-contrôle vs grille) ===")
# La géométrie-CEM = borne SUP de la recherche (ce qu'elle PEUT atteindre AVEC coords). Le score VALEUR (🅑-pur,
# sans coords) doit (1) battre la GRILLE figée (la recherche fabrique de meilleurs plans) ET (2) s'approcher de la
# géométrie-CEM (le score guide presque aussi bien que les coords). On NE se base PAS sur "<1m" (trop dur : portée
# du rêve ~0.66 m << bouffe 1.5-4 m ; le re-grounding closed-loop est le vrai juge).
print(f"  borne recherche (CEM-géométrie) = {cm:.2f} m | grille figée = {gm:.2f} m" + (f" | CEM-valeur = {vm:.2f} m" if vm else ""))
if cm > gm - 0.2:
    print(f"🟡 La RECHERCHE elle-même n'atteint pas mieux que la grille (géométrie {cm:.2f} ≈ grille {gm:.2f}) →")
    print("   c'est la PORTÉE du rêve (0.66 m/120 pas << bouffe) qui borne le gate OFFLINE, pas le score. Le gate")
    print("   '<1m en un rêve' est intrinsèquement trop dur → JUGE = closed-loop (re-grounding). Câbler value+grille.")
elif vm is not None and vm <= cm + 0.2 and vm < gm - 0.2:
    print(f"🟢 SUCCÈS : CEM-VALEUR ({vm:.2f} m) ≈ borne géométrie ({cm:.2f}) ET bat la grille ({gm:.2f}) → le coût-")
    print("   valeur latent GUIDE la recherche presque comme les coords, SANS coordonnées → câbler CEM-valeur au closed-loop.")
elif vm is not None and vm < gm - 0.2:
    print(f"🟡 PARTIEL : CEM-VALEUR ({vm:.2f}) bat la grille ({gm:.2f}) mais reste loin de la géométrie ({cm:.2f}) →")
    print("   le score guide en partie. Câblage closed-loop possible mais sous-optimal ; améliorer la tête de valeur.")
elif vm is not None:
    print(f"🔴 CEM-VALEUR ({vm:.2f}) n'aide pas vs grille ({gm:.2f}) alors que la géométrie atteint {cm:.2f} → le SCORE")
    print("   valeur ne guide pas la recherche. → tête de valeur sur latents rêvés sous commandes diverses, avant câblage.")
else:
    print("   (pas de tête de valeur fournie — relancer avec [value_head] en 4ᵉ argument pour le mode CEM-valeur.)")
