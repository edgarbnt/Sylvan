"""diag_critic_life.py — GATE GRATUIT du critique appris (boucle jour/nuit), AVANT tout build/entraînement lourd.
Question (CLAUDE.md §1) : le VÉCU multi-pulsions déjà collecté (godot/data/replay_buffer/bmds) contient-il le signal
de FORESIGHT qui manque au planner myope — « quand l'énergie est basse, être près de la bouffe = survivre » — et un
critique simple V(état)=survie-future l'apprend-il sur held-out ?

État (par pas) : energy, thirst, position bouffe via le SLOT promu (encode_slot sur la rétine), position eau via le
radar eau. Cible = pas-restants jusqu'à la fin de l'épisode (Monte-Carlo de survie). Modèle = ridge (déterministe,
robuste sur peu de données). k-fold PAR ÉPISODE.

SUCCÈS : (1) le SIGNAL existe dans le vécu — corr(survie-restante, food_dist | énergie basse) nettement NÉGATIVE
(loin de la bouffe quand on a faim → on meurt) ; (2) un critique l'APPREND — corr held-out V vs survie-restante > 0.5
ET reproduit le signe « faim → préférer bouffe proche ». → la boucle jour/nuit est justifiée.
KILL : signal absent (|corr| < 0.2) OU V non appris (corr held-out < 0.3) → la foresight n'est pas dans cet état/vécu
→ repenser (état, cible, plus de données) AVANT d'entraîner. Pas d'enchaînement à l'aveugle.

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_critic_life.py
"""
import os, glob, json, math
import numpy as np
import torch
from sylvan.models.command_wm import CommandWorldModel

BUF = os.environ.get("BUF", "godot/data/replay_buffer/bmds")
WM_CKPT = os.environ.get("WM_CKPT", "data/checkpoints/wm_objcentric_s1/wm_best.pt")
RETINA_DIM = 144


def load_wm():
    ck = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    m = ck["meta"]
    wm = CommandWorldModel(obs_dim=m["obs_dim"], proprio_dim=m["proprio_dim"],
                           predictor_arch=m.get("predictor_arch", "shallow"),
                           with_slot=True, slot_resources=m.get("slot_resources", 1))
    wm.load_state_dict(ck["model"]); wm.eval()
    return wm


def radar_xz(radar):
    """12-secteur radar → (dist, cos_brg, sin_brg) de la cible la plus proche, ou None."""
    if not radar:
        return None
    best, bi = 1e9, -1
    for i, v in enumerate(radar):
        if 0.0 < v < best:
            best, bi = v, i
    if bi < 0:
        return None
    ang = bi * (2 * math.pi / len(radar))
    d = best * 12.0  # échelle approx (radar normalisé) — n'importe : on veut un proxy distance monotone
    return d, math.cos(ang), math.sin(ang)


@torch.no_grad()
def build_dataset(wm):
    eps = []
    for f in sorted(glob.glob(f"{BUF}/*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        T = len(rows)
        feats, ys = [], []
        for t, r in enumerate(rows):
            o = r["obs"]
            e = float(o.get("energy", 100.0)) / 100.0
            th = float(o.get("thirst", 100.0)) / 100.0
            ret = o.get("retina")
            if ret and len(ret) == RETINA_DIM:
                p = wm.slot_encoder.positions(torch.tensor(ret, dtype=torch.float32))[0, :].tolist()
                fdist = math.hypot(p[0], p[1]); fcos = p[1] / (fdist + 1e-6); fsin = p[0] / (fdist + 1e-6)
            else:
                fdist, fcos, fsin = 6.0, 0.0, 0.0
            w = radar_xz(o.get("vision_water") or o.get("vision") or [])
            wdist, wcos, wsin = w if w else (6.0, 0.0, 0.0)
            feats.append([1.0, e, th, fdist, fcos, fsin, wdist, wcos, wsin, e * fdist, th * wdist])
            ys.append(float(T - 1 - t))  # pas-restants = survie future (Monte-Carlo)
        eps.append((np.array(feats), np.array(ys)))
    return eps


def ridge_fit(X, y, lam=10.0):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-9 else float("nan")


def main():
    wm = load_wm()
    eps = build_dataset(wm)
    print(f"épisodes={len(eps)} ; pas total={sum(len(y) for _, y in eps)}")

    # (1) LE SIGNAL EXISTE-T-IL DANS LE VÉCU (brut, sans modèle) ? corr(survie-restante, food_dist) en énergie basse
    allX = np.vstack([X for X, _ in eps]); ally = np.concatenate([y for _, y in eps])
    e_col, fdist_col = allX[:, 1], allX[:, 3]
    low = e_col < 0.3
    print(f"\n[1] SIGNAL BRUT (énergie<0.3, n={int(low.sum())}) :")
    print(f"    corr(survie-restante, food_dist) = {corr(ally[low], fdist_col[low]):+.2f}  (NÉGATIF attendu : loin+faim → meurt)")
    print(f"    corr(survie-restante, energy)     = {corr(ally, e_col):+.2f}  (POSITIF attendu)")

    # (2) UN CRITIQUE L'APPREND-IL ? k-fold PAR ÉPISODE
    preds_all, ys_all, low_pred, low_fdist = [], [], [], []
    for i in range(len(eps)):
        Xtr = np.vstack([eps[j][0] for j in range(len(eps)) if j != i])
        ytr = np.concatenate([eps[j][1] for j in range(len(eps)) if j != i])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; mu[0], sd[0] = 0.0, 1.0
        w = ridge_fit((Xtr - mu) / sd, ytr)
        Xte, yte = eps[i]
        pred = ((Xte - mu) / sd) @ w
        preds_all.append(pred); ys_all.append(yte)
        lo = Xte[:, 1] < 0.3
        low_pred.append(pred[lo]); low_fdist.append(Xte[lo, 3])
    P, Y = np.concatenate(preds_all), np.concatenate(ys_all)
    LP, LF = np.concatenate(low_pred), np.concatenate(low_fdist)
    r = corr(P, Y)
    foresight = corr(LP, LF)  # V vs food_dist en énergie basse — NÉGATIF = V "veut" la bouffe quand on a faim
    print(f"\n[2] CRITIQUE APPRIS (held-out, k-fold par épisode) :")
    print(f"    corr(V, survie-restante)        = {r:+.2f}  (>0.5 = prédit la survie)")
    print(f"    corr(V, food_dist | énergie<0.3) = {foresight:+.2f}  (NÉGATIF = V valorise aller manger quand on a faim)")

    sig_ok = corr(ally[low], fdist_col[low]) < -0.2
    learn_ok = r > 0.5 and foresight < -0.1
    print(f"\n>>> VERDICT : signal-présent={sig_ok}  critique-apprend={learn_ok} → "
          f"{'PASS — boucle jour/nuit justifiée' if (sig_ok and learn_ok) else 'FAIL/KILL — repenser état/cible/données AVANT build (§1)'}")


if __name__ == "__main__":
    main()
