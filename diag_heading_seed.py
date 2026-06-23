"""GATE du fix 'cap dead-reckoné' (offline, 0 entraînement).

Question : en latent-PUR (bearing amorcé par OrientHead sur le latent RÉEL t0, propagé par la displacement
head), peut-on RANGER les candidats pour qu'une cible DERRIÈRE soit ENGAGÉE — sans lire d'oracle au runtime ?

Score de cap dead-reckoné (sans distance) : heading(θ0) = mean_t cos(θ0 − Y(t)), où
  θ0 = bearing lu UNE fois par OrientHead sur predicted_latents[:,0] (latent t0 ≈ état courant),
  Y(t) = cumul des d_yaw prédits par la displacement head (dead-reckoning du virage).
On compare l'argmax de 3 scores sur le vrai 'facing' géométrique A_final (oracle, ÉVAL SEULEMENT) :
  - value.mean  (le coût ACTUEL, qui rate)            -> baseline (≈82% mauvais côté attendu)
  - heading(θ0 ORACLE)                                -> plafond du mécanisme
  - heading(θ0 LATENT, OrientHead t0)                 -> LE FIX (latent-pur)
GATE (pré-enregistré, cibles DERRIÈRE) : heading(latent) mauvais-côté < 25% ET A_final[choisi] > +0.30.

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_heading_seed.py
"""
import json, glob, math, os, statistics as st
import torch
from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE
from sylvan.models.value_head import load_value_head, load_orient_head
from sylvan.control.planning.command_planner import CommandPlanner, CommandPlanConfig

WM = os.environ.get("WM_CKPT", "data/checkpoints/wm_rich_fidele_sym/wm_best.pt")
VH = os.environ.get("VALUE_CKPT", "data/checkpoints/value_head_food_dream/value_best.pt")
OH = os.environ.get("ORIENT_CKPT", "data/checkpoints/orient_head_food/orient_best.pt")
H = int(os.environ.get("H", "120")); NF = int(os.environ.get("NFRAMES", "60"))
DMIN, DMAX = 1.2, 3.5

pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
val = load_value_head(VH); ori = load_orient_head(OH)
planner = CommandPlanner(wm, CommandPlanConfig(horizon=H)); SEQS = planner._cmd_seqs; N = SEQS.shape[0]
print(f"WM={WM} | {N} candidats × H={H}")
files = sorted(glob.glob("data/replay_buffer/retina_forage/episode_*.jsonl") or
               glob.glob("godot/data/replay_buffer/retina_forage/episode_*.jsonl"))
assert files


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
            if behind and fz >= 0:
                continue
            if (not behind) and fz <= 0.7 * d:
                continue
            yield r["obs"]["proprio"], ret, fx, fz


def pose_yaw_and_A(disp, fx, fz):
    Xs = torch.zeros(N); Zs = torch.zeros(N); Ys = torch.zeros(N)
    PY = torch.zeros(N, H); A = torch.zeros(N, H)
    for t in range(H):
        df, dl, dy = disp[:, t, 0], disp[:, t, 1], disp[:, t, 2]
        s, c = torch.sin(Ys), torch.cos(Ys)
        Xs = Xs + df * s + dl * c; Zs = Zs + df * c - dl * s; Ys = Ys + dy
        PY[:, t] = Ys
        dx, dz = fx - Xs, fz - Zs
        fwd_e = dz * torch.cos(Ys) + dx * torch.sin(Ys)
        rgt_e = dx * torch.cos(Ys) - dz * torch.sin(Ys)
        A[:, t] = fwd_e / (torch.sqrt(fwd_e ** 2 + rgt_e ** 2) + 1e-6)
    return PY, A


def pick(score, A_final):
    j = int(torch.argmax(score)); return A_final[j].item(), (1.0 if A_final[j] < 0 else 0.0)


for behind in (True, False):
    lab = "DERRIÈRE" if behind else "DEVANT"
    R = {k: [] for k in ("v_A", "v_wrong", "ho_A", "ho_wrong", "hl_A", "hl_wrong",
                          "ceil", "fb_ok", "side_ok", "dtheta")}
    n = 0
    for proprio, ret, fx, fz in frames(behind):
        obs = torch.tensor(proprio + ret + [0.4], dtype=torch.float32)
        with torch.no_grad():
            out = wm.rollout_open_loop(obs.reshape(1, -1).expand(N, -1).contiguous(), SEQS)
            lat = out["predicted_latents"]
            V = val.value(lat)                              # [N,H]
            cs0 = ori.cos_sin(lat[:, 0, :]).mean(0)         # bearing t0 partagé (latent-pur)
        disp = out["predicted_displacement"] / DISPLACEMENT_SCALE
        PY, A = pose_yaw_and_A(disp, fx, fz)
        A_final = A[:, -20:].mean(1)                        # [N] vrai facing (oracle, éval)
        th_lat = math.atan2(float(cs0[1]), float(cs0[0]))
        th_orc = math.atan2(fx, fz)
        h_lat = torch.cos(th_lat - PY).mean(1)
        h_orc = torch.cos(th_orc - PY).mean(1)
        v_score = V.mean(1)
        a, wr = pick(v_score, A_final); R["v_A"].append(a); R["v_wrong"].append(wr)
        a, wr = pick(h_orc, A_final); R["ho_A"].append(a); R["ho_wrong"].append(wr)
        a, wr = pick(h_lat, A_final); R["hl_A"].append(a); R["hl_wrong"].append(wr)
        R["ceil"].append(A_final.max().item())
        R["fb_ok"].append(1.0 if (math.cos(th_lat) > 0) == (math.cos(th_orc) > 0) else 0.0)
        R["side_ok"].append(1.0 if (math.sin(th_lat) > 0) == (math.sin(th_orc) > 0) else 0.0)
        dd = abs((th_lat - th_orc + math.pi) % (2 * math.pi) - math.pi)
        R["dtheta"].append(math.degrees(dd))
        n += 1
        if n >= NF:
            break
    m = lambda k: st.mean(R[k]) if R[k] else float("nan")
    print(f"\n===== cible {lab} (n={n}) — A_final[choisi] (vrai facing, +1=face) & % mauvais côté =====")
    print(f"  plafond mécanisme : meilleur A_final dispo = {m('ceil'):+.2f}")
    print(f"  coût ACTUEL (value.mean)     : A={m('v_A'):+.2f}  mauvais-côté={m('v_wrong'):.0%}")
    print(f"  heading θ0 ORACLE            : A={m('ho_A'):+.2f}  mauvais-côté={m('ho_wrong'):.0%}")
    print(f"  heading θ0 LATENT (le FIX)   : A={m('hl_A'):+.2f}  mauvais-côté={m('hl_wrong'):.0%}")
    print(f"  qualité OrientHead t0 : devant/derrière OK={m('fb_ok'):.0%}  côté OK={m('side_ok'):.0%}  |Δθ| méd={m('dtheta'):.0f}°")
    if behind:
        ok = (m('hl_wrong') < 0.25 and m('hl_A') > 0.30)
        print(f"  >>> GATE (mauvais-côté<25% ET A>+0.30) : {'PASSE ✅' if ok else 'ÉCHEC ❌'}")
