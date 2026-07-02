"""diag_eat_dynamics.py — GATE GRATUIT eat-dynamics du WM (préalable au critique appris MODEL-BASED).
Un critique model-based apprend V sur l'IMAGINATION du WM ; il ne peut valoriser « aller manger » que si le WM
PRÉDIT la remontée d'énergie au contact de la bouffe. On teste la condition nécessaire (teacher-forced 1-pas) :
à chaque VRAI repas du vécu (saut d'énergie réel), le WM prédit-il une bosse positive — et discrimine-t-il repas
vs non-repas ?

Données : retina_wm_a/b (cmd + retina0 + energy + wm.ate, 74 repas). État WM = proprio ++ retina0 ++ energy/100.
À une transition de repas (i-1 → i, energy[i]-energy[i-1] > SEUIL) : pred = wm.forward(obs_{i-1}, cmd_{i-1}) →
énergie prédite à i ; on compare ΔE prédit vs ΔE réel. Baseline = transitions SANS repas (drain lisse).

SUCCÈS : le WM prédit une bosse nette aux repas (capte > ~30% du saut réel ET ΔE_repas >> ΔE_non-repas) → signal
eat présent dans l'imagination → critique model-based FAISABLE.
KILL : aux repas le WM prédit ~le drain (capte < ~10%, pas de discrimination) → le WM ne modélise PAS le repas →
il faut d'abord réparer l'eat-dynamics du WM (retrain énergie food-aware) AVANT le critique. Pas de build à l'aveugle.

Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_eat_dynamics.py
"""
import os, glob, json
import torch
from sylvan.models.command_wm import CommandWorldModel

BUFS = os.environ.get("BUFS", "retina_wm_a retina_wm_b").split()
WM_CKPT = os.environ.get("WM_CKPT", "data/checkpoints/wm_objcentric_s1/wm_best.pt")
JUMP = float(os.environ.get("JUMP", "5.0"))
RETINA_DIM = 144


def load_wm():
    ck = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    m = ck["meta"]
    wm = CommandWorldModel(obs_dim=m["obs_dim"], proprio_dim=m["proprio_dim"],
                           predictor_arch=m.get("predictor_arch", "shallow"),
                           with_slot=True, slot_resources=m.get("slot_resources", 1))
    wm.load_state_dict(ck["model"]); wm.eval()
    return wm


@torch.no_grad()
def pred_next_energy(wm, proprio, retina, e_norm, cmd):
    obs = torch.tensor(proprio + retina + [e_norm], dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
    c = torch.tensor([[list(cmd[:2])]], dtype=torch.float32)  # [1,1,2]
    out = wm.rollout_open_loop(obs, c)  # prédiction 1-pas depuis l'obs réelle
    return float(out["predicted_next_obs"][0, 0, -1]) * 100.0  # dé-normalise


def main():
    wm = load_wm()
    eat_real, eat_pred, non_real, non_pred = [], [], [], []
    nonsample = 0
    for buf in BUFS:
        for f in sorted(glob.glob(f"data/replay_buffer/{buf}/*.jsonl")):
            rows = [json.loads(l) for l in open(f)]
            for i in range(1, len(rows)):
                o0, o1 = rows[i - 1]["obs"], rows[i]["obs"]
                w0 = rows[i - 1].get("wm", {})
                if not w0.get("retina0") or not w0.get("cmd"):
                    continue
                e0, e1 = float(o0.get("energy", 0.0)), float(o1.get("energy", 0.0))
                d_real = e1 - e0
                pe = pred_next_energy(wm, o0["proprio"], w0["retina0"], e0 / 100.0, w0["cmd"])
                d_pred = pe - e0
                if d_real > JUMP:
                    eat_real.append(d_real); eat_pred.append(d_pred)
                else:
                    nonsample += 1
                    if nonsample % 200 == 0:  # échantillon des non-repas (drain lisse)
                        non_real.append(d_real); non_pred.append(d_pred)

    def mean(x):
        return sum(x) / len(x) if x else float("nan")
    er, ep = mean(eat_real), mean(eat_pred)
    nr, npd = mean(non_real), mean(non_pred)
    capture = ep / er if er else float("nan")
    print(f"REPAS (n={len(eat_real)})      : ΔE réel moy = {er:+.1f}   | ΔE prédit moy = {ep:+.2f}   | capté = {100*capture:.0f}% du saut")
    print(f"NON-REPAS (n={len(non_real)})  : ΔE réel moy = {nr:+.2f}  | ΔE prédit moy = {npd:+.2f}")
    discrim = ep - npd
    print(f"DISCRIMINATION ΔE_pred(repas) − ΔE_pred(non-repas) = {discrim:+.2f}  (positif = le WM 'sait' que le contact nourrit)")
    ok = (capture > 0.30) and (discrim > 2.0)
    print(f"\n>>> VERDICT : capté={100*capture:.0f}%(>30%?)  discrim={discrim:+.2f}(>2?) → "
          f"{'PASS — eat-dynamics présent → critique model-based faisable' if ok else 'FAIL/KILL — le WM ne modélise pas le repas → réparer l eat-dynamics AVANT le critique (§1)'}")


if __name__ == "__main__":
    main()
