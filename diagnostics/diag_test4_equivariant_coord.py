"""TEST 4 / F1 (design WM factorisé) — FAISABILITÉ object-centric, GRATUIT, 0 retrain.

Hypothèse 3c : le bearing échoue parce qu'il est une direction latente ENTORTILLÉE. Parade = représenter la position
égocentrique de l'objet comme une COORDONNÉE EXPLICITE (slot) que l'auto-mouvement transforme par une opération RIGIDE
connue. Ici on teste cette hypothèse AU NIVEAU DE LA COORDONNÉE 2D `p = food_rel0 = (fx, fz)` (pas le latent 128-d) :

  p_{t+1} ?= Rot(k_w · omega_t) · p_t  −  k_v · vx_t · ẑ      (ẑ = avant = (0,1) ; fx=latéral, fz=avant)

On FIT (k_w, k_v) sur train (2 scalaires), on ROULE OPEN-LOOP depuis p_0 sur des segments de virage held-out, et on
mesure le transport du bearing (corr poolée cos(brg), par horizon ET bucket front/ARRIÈRE). Comparaisons :
  - ANALYTIQUE (coord explicite + transform rigide)   = ce qu'un slot-WM ferait
  - STATIQUE p_0 (gèle le bearing initial)            = contrôle (le transform fait-il un VRAI travail ?)
  - réf WM-dream = +0.30 (mesuré dans diag_test3, latent monolithique)

SUCCÈS : analytique ≥ +0.8 global ET ≥ +0.7 arrière → slot-WM = bonne archi, refonte GO.
PARTIEL (0.5-0.8 / arrière faible) : la commande seule dérive → ego-motion à APPRENDRE depuis le proprio (reste factorisé).
KILL (≈ +0.30) : la coord 2D ne se transforme pas rigidement → re-concevoir (très improbable, kinématique).

Usage: BUF=retina_eat_a PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_test4_equivariant_coord.py
"""
import json, glob, math, os, statistics as st
import torch

torch.manual_seed(0); torch.set_num_threads(4)
BUF = os.environ.get("BUF", "retina_eat_a")
L = 40; OMG = 0.30
files = sorted(glob.glob(f"godot/data/replay_buffer/{BUF}/episode_*.jsonl") or
               glob.glob(f"data/replay_buffer/{BUF}/episode_*.jsonl"))[:80]
print(f"BUF={BUF}  fichiers={len(files)}")


def load_eps():
    eps = []
    for f in files:
        seq = []
        for line in open(f):
            r = json.loads(line); w = r.get("wm", {})
            fr = w.get("food_rel0"); cmd = w.get("cmd")
            if not fr or not cmd:
                continue
            seq.append((float(fr[0]), float(fr[1]), float(fr[2]), float(cmd[0]), float(cmd[1])))  # fx,fz,vis,vx,om
        if len(seq) > L + 2:
            eps.append(seq)
    return eps


eps = load_eps()
ntr = max(1, int(0.8 * len(eps)))
print(f"épisodes={len(eps)} (train={ntr}, test={len(eps)-ntr})")

# segments de virage, objet visible à t0
segs = []  # dict: P(L,2), VIS(L), VX(L), OM(L), is_train
for ei, seq in enumerate(eps):
    is_tr = ei < ntr
    for s0 in range(0, len(seq) - L, L):
        win = seq[s0:s0 + L]
        if win[0][2] < 0.5:
            continue
        if st.mean(abs(w[4]) for w in win) <= OMG:
            continue
        P = torch.tensor([[w[0], w[1]] for w in win])
        VIS = torch.tensor([w[2] for w in win])
        VX = torch.tensor([w[3] for w in win]); OM = torch.tensor([w[4] for w in win])
        segs.append((P, VIS, VX, OM, is_tr))
print(f"segments de virage = {len(segs)} (train={sum(s[4] for s in segs)})")


def rigid_step(p, vx, om, k_w, k_v):
    # p: (...,2) [fx,fz] ; vx,om: (...) ; Rot(k_w*om) sur (fx,fz) puis translation avant -k_v*vx sur fz
    a = k_w * om
    ca, sa = torch.cos(a), torch.sin(a)
    fx = ca * p[..., 0] - sa * p[..., 1]
    fz = sa * p[..., 0] + ca * p[..., 1] - k_v * vx
    return torch.stack([fx, fz], dim=-1)


# --- Précalcul vectorisé des transitions TRAIN (objet visible aux 2 bouts) ---
Pt, Pn, VXt, OMt = [], [], [], []
for P, VIS, VX, OM, is_tr in segs:
    if not is_tr:
        continue
    for t in range(L - 1):
        if VIS[t] > 0.5 and VIS[t + 1] > 0.5:
            Pt.append(P[t]); Pn.append(P[t + 1]); VXt.append(VX[t]); OMt.append(OM[t])
Pt = torch.stack(Pt); Pn = torch.stack(Pn); VXt = torch.stack(VXt); OMt = torch.stack(OMt)
print(f"transitions train = {len(Pt)}")

# --- Fit (k_w, k_v) vectorisé ---
k_w = torch.tensor(0.1, requires_grad=True)
k_v = torch.tensor(0.02, requires_grad=True)
opt = torch.optim.Adam([k_w, k_v], lr=0.02)
for it in range(3000):
    pred = rigid_step(Pt, VXt, OMt, k_w, k_v)
    loss = ((pred - Pn) ** 2).sum(-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"fit: k_w={k_w.item():.4f} rad/(ω·pas)  k_v={k_v.item():.4f} m/(vx·pas)  (1-step MSE={loss.item():.5f})")
kw, kv = k_w.detach(), k_v.detach()


@torch.no_grad()
def rollout_analytic(P, VX, OM):
    p = P[0].clone(); out = [p]
    for t in range(L - 1):
        p = rigid_step(p, VX[t], OM[t], kw, kv)
        out.append(p)
    return torch.stack(out)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = a.norm() * b.norm()
    return (a @ b).item() / d.item() if d > 1e-6 else float("nan")


# --- MÉTRIQUE DISCRIMINANTE 1 : Δbearing prédit (1-pas analytique) vs Δbearing réel, sur frames ROTANTES TEST ---
# (pas de confound statique : on teste le CHANGEMENT de bearing, pas son niveau.)
dpred, dtrue, dpred_r, dtrue_r, sweeps = [], [], [], [], []
for P, VIS, VX, OM, is_tr in segs:
    if is_tr:
        continue
    for t in range(L - 1):
        if VIS[t] > 0.5 and VIS[t + 1] > 0.5 and abs(float(OM[t])) > OMG:
            bt0 = math.atan2(float(P[t][0]), float(P[t][1]))
            bt1 = math.atan2(float(P[t + 1][0]), float(P[t + 1][1]))
            pstep = rigid_step(P[t], VX[t], OM[t], kw, kv)
            bp1 = math.atan2(float(pstep[0]), float(pstep[1]))
            dt = math.atan2(math.sin(bt1 - bt0), math.cos(bt1 - bt0))    # vrai Δbearing (wrap)
            dp = math.atan2(math.sin(bp1 - bt0), math.cos(bp1 - bt0))    # prédit Δbearing
            dtrue.append(dt); dpred.append(dp); sweeps.append(abs(dt))
            if abs(bt0) > math.pi / 2:
                dtrue_r.append(dt); dpred_r.append(dp)
cD = corr(torch.tensor(dpred), torch.tensor(dtrue))
cDr = corr(torch.tensor(dpred_r), torch.tensor(dtrue_r)) if dtrue_r else float("nan")
# SANITY (anti-bug convention) : le ω BRUT corrèle-t-il avec le Δbearing réel ? (rotation gauche → bearing baisse)
om_raw = []
for P, VIS, VX, OM, is_tr in segs:
    if is_tr:
        continue
    for t in range(L - 1):
        if VIS[t] > 0.5 and VIS[t + 1] > 0.5 and abs(float(OM[t])) > OMG:
            om_raw.append(-float(OM[t]))
cOm = corr(torch.tensor(om_raw), torch.tensor(dtrue))
print(f"[SANITY] corr(−ω brut, Δbearing réel) = {cOm:+.2f}  (si ≈0 → l'ego-motion n'est PAS commande-déterminée, pas un bug)")
print(f"\n[DISCRIMINANT] Δbearing prédit vs réel (frames rotantes test, n={len(dtrue)}): "
      f"corr GLOBAL={cD:+.2f}  ARRIÈRE={cDr:+.2f}  | |Δbrg réel| moy={math.degrees(st.mean(sweeps)):.1f}°/pas")

# --- MÉTRIQUE DISCRIMINANTE 2 : sur segments à VRAI balayage (range |brg| intra-seg > 45°), analyt vs static ---
hp_a, hp_t, hp_s = [], [], []
nseg_sweep = 0
for P, VIS, VX, OM, is_tr in segs:
    if is_tr:
        continue
    brgs = [math.atan2(float(P[t][0]), float(P[t][1])) for t in range(L) if VIS[t] > 0.5]
    if len(brgs) < 5 or (max(abs(b) for b in brgs) - min(abs(b) for b in brgs)) < math.radians(45):
        continue
    nseg_sweep += 1
    roll = rollout_analytic(P, VX, OM); b0 = math.atan2(float(P[0][0]), float(P[0][1]))
    for t in range(L):
        if VIS[t] > 0.5:
            hp_a.append(math.cos(math.atan2(float(roll[t][0]), float(roll[t][1]))))
            hp_t.append(math.cos(math.atan2(float(P[t][0]), float(P[t][1]))))
            hp_s.append(math.cos(b0))
if hp_a:
    print(f"[BALAYAGE FORT] {nseg_sweep} segments (range|brg|>45°) : "
          f"ANALYT={corr(torch.tensor(hp_a), torch.tensor(hp_t)):+.2f}  STATIC={corr(torch.tensor(hp_s), torch.tensor(hp_t)):+.2f}")
else:
    print(f"[BALAYAGE FORT] AUCUN segment à balayage>45° dans {BUF} test → données sans vrai sweep (cf mur 3b).")

# transport du bearing : pooled corr de cos(brg_pred) vs cos(brg_true), par horizon + bucket
print("\ntransport du bearing (corr poolée cos(brg), segments virage TEST) :")
print(f"{'H':>4} | {'ANALYT':>7} | {'STATIC':>7} || {'ANALYT-arr':>10} | {'STATIC-arr':>10} | {'n_arr':>5}")
for H in (5, 10, 20, 40):
    ap, tp, sp = [], [], []          # global
    aap, atp, asp = [], [], []       # arrière (|brg_true|>90)
    for P, VIS, VX, OM, is_tr in segs:
        if is_tr:
            continue
        roll = rollout_analytic(P, VX, OM)
        for t in range(min(H, L)):
            if VIS[t] > 0.5:
                bt = math.atan2(float(P[t][0]), float(P[t][1]))      # true bearing
                ba = math.atan2(float(roll[t][0]), float(roll[t][1]))  # analytic
                b0 = math.atan2(float(P[0][0]), float(P[0][1]))        # static (frozen t0)
                ct, ca, c0 = math.cos(bt), math.cos(ba), math.cos(b0)
                ap.append(ca); tp.append(ct); sp.append(c0)
                if abs(bt) > math.pi / 2:
                    aap.append(ca); atp.append(ct); asp.append(c0)
    g_a = corr(torch.tensor(ap), torch.tensor(tp)); g_s = corr(torch.tensor(sp), torch.tensor(tp))
    if aap:
        r_a = corr(torch.tensor(aap), torch.tensor(atp)); r_s = corr(torch.tensor(asp), torch.tensor(atp))
    else:
        r_a = r_s = float("nan")
    print(f"{H:>4} | {g_a:>+7.2f} | {g_s:>+7.2f} || {r_a:>+10.2f} | {r_s:>+10.2f} | {len(aap):>5}")

# verdict sur H=40 global + arrière
ap, tp, aap, atp = [], [], [], []
for P, VIS, VX, OM, is_tr in segs:
    if is_tr:
        continue
    roll = rollout_analytic(P, VX, OM)
    for t in range(L):
        if VIS[t] > 0.5:
            bt = math.atan2(float(P[t][0]), float(P[t][1])); ba = math.atan2(float(roll[t][0]), float(roll[t][1]))
            ap.append(math.cos(ba)); tp.append(math.cos(bt))
            if abs(bt) > math.pi / 2:
                aap.append(math.cos(ba)); atp.append(math.cos(bt))
G = corr(torch.tensor(ap), torch.tensor(tp)); Rr = corr(torch.tensor(aap), torch.tensor(atp)) if aap else float("nan")
print(f"\nVERDICT (H=40) : analytique global={G:+.2f}  arrière={Rr:+.2f}   (réf WM-dream +0.30 ; seuils +0.8 / +0.7 arr)")
if G >= 0.8 and (math.isnan(Rr) or Rr >= 0.7):
    print(">>> GO : coordonnée explicite + transform rigide transporte le bearing → slot-WM = bonne archi. Build S1.")
elif G >= 0.5:
    print(">>> PARTIEL : la commande seule dérive (esp. arrière) → ego-motion à APPRENDRE depuis proprio (reste factorisé).")
else:
    print(">>> KILL : la coord 2D ne se transforme pas rigidement → re-concevoir (improbable).")
