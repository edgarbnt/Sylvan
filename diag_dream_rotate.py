"""T-mag — le rêve open-loop perd-il la cible PENDANT un virage ? (offline, 0 entraînement)

Hypothèse (carte code + obs 979 'magnitude collapse breaks visual dream despite good directional accuracy') :
la magnitude du latent rêvé s'effondre en rollout (WM cosine+VICReg, pas d'ancre de norme) → les têtes
(value/orient) s'éteignent → une cible DERRIÈRE (qui exige une longue rotation AVANT acquisition) n'est jamais
'vue' arriver devant dans le rêve, alors même que le WM PRÉDIT qu'il a tourné (displacement head).

Test d'AUTO-COHÉRENCE (pas d'oracle, pas de Godot) : pour chaque frame replay où la bouffe est DERRIÈRE
(food_rel0.fwd < 0), on prend le candidat de la grille qui, selon la GÉOMÉTRIE prédite par le WM lui-même
(cumul des déplacements d_fwd/d_lat/d_yaw), finit le mieux ORIENTÉ vers la bouffe. Le long de ce candidat :
  A = ahead_geo  : cos(bearing) impliqué par la POSE prédite  <- ce que le WM 'croit' faire de son corps
  C = ahead_lat  : cos(bearing) lu par OrientHead sur le LATENT rêvé  <- ce que la PERCEPTION latente dit
  V = value(latent) (repas imminent)
  |x| = norme de l'entrée normalisée (latent-mu)/sd de la tête  (collapse ?)
Puis on REFAIT C et V avec l'entrée RENORMALISÉE à |x|_t0 (le test EST aussi un fix gratuit potentiel).
Contrôle : mêmes mesures sur cibles DEVANT (censées marcher).

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_dream_rotate.py
Env: WM_CKPT VALUE_CKPT ORIENT_CKPT H NFRAMES
"""
import json, glob, math, os, statistics as st
import torch
from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE
from sylvan.models.value_head import load_value_head, load_orient_head
from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig

WM = os.environ.get("WM_CKPT", "data/checkpoints/wm_rich_fidele_sym/wm_best.pt")
VH = os.environ.get("VALUE_CKPT", "data/checkpoints/value_head_food_dream/value_best.pt")
OH = os.environ.get("ORIENT_CKPT", "data/checkpoints/orient_head_food/orient_best.pt")
H = int(os.environ.get("H", "120"))
NF = int(os.environ.get("NFRAMES", "40"))
DMIN, DMAX = 1.2, 3.5

pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
val = load_value_head(VH); ori = load_orient_head(OH)
planner = CommandPlanner(wm, CommandPlanConfig(horizon=H)); SEQS = planner._cmd_seqs; N = SEQS.shape[0]
print(f"WM={WM}\nvalue={VH}\norient={OH}\n{N} candidats × H={H} | obs_dim(meta)={meta['obs_dim']}")

files = sorted(glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl"))
assert files, "pas de replay retina_forage trouvé"


def frames(behind):
    for f in files:
        for line in open(f):
            r = json.loads(line); w = r.get("wm", {})
            ret = w.get("retina0"); fr = w.get("food_rel0")
            if not ret or not fr or fr[2] < 0.5:
                continue
            fx, fz = fr[0], fr[1]; d = math.hypot(fx, fz)
            if not (DMIN <= d <= DMAX):
                continue
            if behind and fz >= 0:           # DERRIÈRE = fwd négatif
                continue
            if (not behind) and fz <= 0.7 * d:  # DEVANT net
                continue
            yield r["obs"]["proprio"], ret, fx, fz


def pose(disp):  # disp [N,H,3] (mètres/rad, déjà déscalés) -> X,Z,Y intégrés [N,H]
    Xs = torch.zeros(N); Zs = torch.zeros(N); Ys = torch.zeros(N)
    PX = torch.zeros(N, H); PZ = torch.zeros(N, H); PY = torch.zeros(N, H)
    for t in range(H):
        df, dl, dy = disp[:, t, 0], disp[:, t, 1], disp[:, t, 2]
        s, c = torch.sin(Ys), torch.cos(Ys)
        Xs = Xs + df * s + dl * c; Zs = Zs + df * c - dl * s; Ys = Ys + dy
        PX[:, t] = Xs; PZ[:, t] = Zs; PY[:, t] = Ys
    return PX, PZ, PY


def ahead_geo(PX, PZ, PY, fx, fz):
    dx = fx - PX; dz = fz - PZ
    fwd_e = dz * torch.cos(PY) + dx * torch.sin(PY)
    rgt_e = dx * torch.cos(PY) - dz * torch.sin(PY)
    return fwd_e / (torch.sqrt(fwd_e ** 2 + rgt_e ** 2) + 1e-6)   # [N,H]


def head_rr(head, lat, kind):  # readout RAW + RENORM(entrée à |x|_t0) + norme entrée
    x = (lat - head.mu) / head.sd                 # [N,H,L]
    nrm = x.norm(dim=-1)                           # [N,H]
    tgt = nrm[:, 0:1].clamp(min=1e-6)
    xr = x / nrm.unsqueeze(-1).clamp(min=1e-6) * tgt.unsqueeze(-1)
    o, orn = head.net(x), head.net(xr)
    if kind == "value":
        return torch.sigmoid(o.squeeze(-1)), torch.sigmoid(orn.squeeze(-1)), nrm
    return o[..., 0] / (o.norm(dim=-1) + 1e-6), orn[..., 0] / (orn.norm(dim=-1) + 1e-6), nrm


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = (a.norm() * b.norm())
    return (a @ b).item() / d.item() if d > 1e-8 else float("nan")


printed_dims = False
for behind in (True, False):
    lab = "DERRIÈRE" if behind else "DEVANT"
    K = ("normr40", "normr80", "normr119", "maxA", "Aacq", "Vraw_acq", "Craw_acq", "Vrn_acq",
         "Crn_acq", "corrAC_raw", "corrAC_rn", "Vmax_raw", "Vmax_rn", "wrongway")
    agg = {k: [] for k in K}; n = 0
    for proprio, ret, fx, fz in frames(behind):
        if not printed_dims:
            print(f"dims: proprio={len(proprio)} retina={len(ret)} +energie1 = {len(proprio)+len(ret)+1}")
            printed_dims = True
        obs = torch.tensor(proprio + ret + [0.4], dtype=torch.float32)   # affamé
        with torch.no_grad():
            out = wm.rollout_open_loop(obs.reshape(1, -1).expand(N, -1).contiguous(), SEQS)
        lat = out["predicted_latents"]
        disp = out["predicted_displacement"] / DISPLACEMENT_SCALE
        PX, PZ, PY = pose(disp)
        A = ahead_geo(PX, PZ, PY, fx, fz)                # [N,H]
        Vraw, Vrn, Vn = head_rr(val, lat, "value")
        Craw, Crn, _ = head_rr(ori, lat, "orient")
        bestA = int(A[:, -20:].mean(1).argmax())         # candidat 'bon virage' (géométrie)
        chosen = int(Vraw.mean(1).argmax())              # candidat CHOISI par le coût (value mean, = live)
        a, vraw, craw, vrn, crn, vn = A[bestA], Vraw[bestA], Craw[bestA], Vrn[bestA], Crn[bestA], Vn[bestA]
        acqs = (a > 0.3).nonzero()
        tacq = int(acqs[0]) if len(acqs) else H - 1      # 1er pas où la POSE amène la cible devant
        agg["normr40"].append((vn[40] / vn[0]).item()); agg["normr80"].append((vn[80] / vn[0]).item())
        agg["normr119"].append((vn[119] / vn[0]).item()); agg["maxA"].append(a.max().item())
        agg["Aacq"].append(a[tacq].item())
        agg["Vraw_acq"].append(vraw[tacq].item()); agg["Craw_acq"].append(craw[tacq].item())
        agg["Vrn_acq"].append(vrn[tacq].item()); agg["Crn_acq"].append(crn[tacq].item())
        agg["corrAC_raw"].append(corr(a, craw)); agg["corrAC_rn"].append(corr(a, crn))
        agg["Vmax_raw"].append(vraw.max().item()); agg["Vmax_rn"].append(vrn.max().item())
        agg["wrongway"].append(1.0 if A[chosen, -20:].mean() < 0 else 0.0)
        n += 1
        if n >= NF:
            break
    m = lambda k: st.mean([v for v in agg[k] if not math.isnan(v)]) if any(not math.isnan(v) for v in agg[k]) else float("nan")
    print(f"\n===== cible {lab} (n={n}, e0=0.4 affamé) =====")
    print(f"  collapse magnitude |x|_t/|x|_0 : @40={m('normr40'):.2f} @80={m('normr80'):.2f} @119={m('normr119'):.2f}")
    print(f"  candidat 'bon virage' : maxA={m('maxA'):+.2f}  A@acq={m('Aacq'):+.2f}  (A>0.3 = cible amenée devant par la POSE)")
    print(f"  --- à l'acquisition (POSE met la cible devant) : la PERCEPTION latente suit-elle ? ---")
    print(f"    RAW    : value={m('Vraw_acq'):.3f}  ahead_lat={m('Craw_acq'):+.2f}")
    print(f"    RENORM : value={m('Vrn_acq'):.3f}  ahead_lat={m('Crn_acq'):+.2f}")
    print(f"  corr(A, ahead_lat) sur la traj : RAW={m('corrAC_raw'):+.2f}  RENORM={m('corrAC_rn'):+.2f}")
    print(f"  value MAX sur horizon : RAW={m('Vmax_raw'):.3f}  RENORM={m('Vmax_rn'):.3f}")
    if behind:
        print(f"  [fact2] le coût choisit un virage du MAUVAIS côté : {m('wrongway'):.0%} des frames")

print("\nLECTURE :")
print(" - collapse fort (@119 << 1) = magnitude s'effondre en rollout.")
print(" - DERRIÈRE, RAW ahead_lat reste <0 alors que A>0.3 à l'acquisition = le latent ne transporte PAS la")
print("   perception du virage (incohérent avec la pose que le WM prédit lui-même).")
print(" - si RENORM relève ahead_lat/value vers A => MAGNITUDE = cause -> fix gratuit (renorm en inférence).")
print(" - si RENORM ne change rien => la DIRECTION du latent est fausse (transport D ou encodeur) -> escalade.")
