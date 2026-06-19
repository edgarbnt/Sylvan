"""GATE GRATUIT (JEPA-pur) — le latent ACTUEL porte-t-il un signal de VALEUR exploitable ?

🅑-pur = le planner note des états LATENTS via une tête APPRISE (pas de coordonnées). Avant d'enrichir le WM,
on teste si le latent GELÉ encode déjà « vais-je manger dans K pas ? » (cible SCALAIRE nette, ≠ position qui
butait sur l'argmin-sur-rayons). Tête = petit MLP latent→logit, label = un repas survient dans les K prochains
pas du MÊME épisode. Split held-out PAR ÉPISODE (pas de fuite). Données = eat-riche (323 repas).
Contrôles : (oracle) distance-bouffe→manger-bientôt = plafond/label apprenable ; (énergie) readout seul.
SUCCÈS AUC≥~0.70 → le latent porte le signal → brancher un coût-valeur latent. KILL AUC≈0.5 → latent l'a jeté →
il FAUT une tête auxiliaire pendant l'entraînement du WM (enrichir le latent). Le contrôle oracle DOIT être haut
(sinon label/K mal posé).
Usage: SYLVAN_WM_USE_RETINA=1 PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_eat_value_probe.py [wm_ckpt]
"""
import sys, json, glob, math
import torch
from sylvan.models.command_wm import CommandWorldModel

WM = sys.argv[1] if len(sys.argv) > 1 else "data/checkpoints/wm_command_hex_retina_eat_v1/wm_best.pt"
K = 20          # "manger dans les K prochains pas"
DIRS = ["godot/data/replay_buffer/retina_eat_a", "godot/data/replay_buffer/retina_eat_b"]

pl = torch.load(WM, map_location="cpu", weights_only=False); meta = pl["meta"]
wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                       predictor_arch=meta.get("predictor_arch", "shallow"))
wm.load_state_dict(pl["model"]); wm.eval()
print(f"WM={WM} obs_dim={meta['obs_dim']} | label='repas dans {K} pas' | latent gelé teacher-forced")

# Charge les épisodes : par frame on garde (obs vector, cmd, ate, dist-bouffe, énergie). Label = eat dans K pas.
ep_obs, ep_cmd, ep_lab, ep_dist, ep_en, ep_id = [], [], [], [], [], []
eid = 0
for d in DIRS:
    for f in sorted(glob.glob(f"{d}/episode_*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        ate = [1.0 if r.get("wm", {}).get("ate") else 0.0 for r in rows]
        n = len(rows)
        for i, r in enumerate(rows):
            w = r.get("wm", {}); ret = w.get("retina0"); fr = w.get("food_rel0")
            if not ret:
                continue
            lab = 1.0 if sum(ate[i + 1:i + 1 + K]) > 0 else 0.0
            ep_obs.append(r["obs"]["proprio"] + ret + [r["obs"]["energy"] / 100.0])
            ep_cmd.append((w.get("cmd") or [0.0, 0.0])[:2])
            ep_lab.append(lab)
            ep_dist.append(math.hypot(fr[0], fr[1]) if fr else 10.0)
            ep_en.append(r["obs"]["energy"] / 100.0)
            ep_id.append(eid)
        eid += 1
OBS = torch.tensor(ep_obs, dtype=torch.float32); CMD = torch.tensor(ep_cmd, dtype=torch.float32)
LAB = torch.tensor(ep_lab); DIST = torch.tensor(ep_dist); EN = torch.tensor(ep_en); EID = torch.tensor(ep_id)
print(f"frames={len(LAB)} | épisodes={eid} | positifs (repas<{K}pas)={100*LAB.mean():.1f}%")

# latent teacher-forced (1 pas, commande réelle), batché
lats = []
with torch.no_grad():
    for s in range(0, OBS.shape[0], 4096):
        o = OBS[s:s + 4096]; c = CMD[s:s + 4096].reshape(-1, 1, 2)
        lats.append(wm.rollout_open_loop(o, c)["predicted_latents"][:, 0, :])
LAT = torch.cat(lats)

# split par épisode (70/30)
ne = eid; cut = int(ne * 0.7)
tr = EID < cut; te = ~tr


def auc(score, label):
    s, l = score.flatten(), label.flatten()
    order = torch.argsort(s); ranks = torch.empty_like(s); ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    np_, nn_ = l.sum().item(), (1 - l).sum().item()
    return float("nan") if np_ == 0 or nn_ == 0 else (ranks[l == 1].sum().item() - np_ * (np_ + 1) / 2) / (np_ * nn_)


class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 256), torch.nn.SiLU(), torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def fit_auc(X, y_tr, y_te, epochs=600, lr=2e-3, wd=1e-4):
    mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    net = MLP(X.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    pw = ((1 - y_tr).sum() / (y_tr.sum() + 1e-6)).clamp(1, 50)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(net(Xn[tr]), y_tr); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return auc(net(Xn[te]), y_te)


auc_lat = fit_auc(LAT, LAB[tr], LAB[te])
auc_en = fit_auc(EN.unsqueeze(1), LAB[tr], LAB[te])
auc_oracle = auc(-DIST[te], LAB[te])   # plus proche = plus susceptible de manger bientôt

print(f"\n=== AUC held-out ('repas dans {K} pas') ===")
print(f"  LATENT gelé (MLP)          : {auc_lat:.3f}   ← le chiffre du GATE")
print(f"  énergie-readout seul (MLP) : {auc_en:.3f}   (réf : signal mou qui a fait échouer 🅑)")
print(f"  ORACLE distance-bouffe     : {auc_oracle:.3f}   (contrôle : label apprenable ? doit être haut)")
print(f"\n=== VERDICT ===")
if math.isnan(auc_oracle) or auc_oracle < 0.7:
    print(f"⚠️ label mal posé (oracle {auc_oracle:.2f} bas) — ajuster K avant de conclure.")
elif auc_lat >= 0.70:
    print(f"🟢 le latent PORTE le signal de valeur (AUC {auc_lat:.2f}) → brancher un coût-VALEUR latent (tête apprise)")
    print(f"   suffit, sans ré-entraîner le WM. C'est le bon instrument (≠ énergie {auc_en:.2f}).")
elif auc_lat >= 0.60:
    print(f"🟡 signal FAIBLE dans le latent (AUC {auc_lat:.2f}) → une tête apprise aide un peu ; probablement")
    print(f"   à coupler avec l'enrichissement du WM (tête auxiliaire) pour être robuste.")
else:
    print(f"🔴 le latent a JETÉ le signal (AUC {auc_lat:.2f} ≈ oracle {auc_oracle:.2f} inatteignable) → il FAUT une")
    print(f"   TÊTE AUXILIAIRE pendant l'entraînement du WM pour forcer le latent food-aware (puis coût-valeur latent).")
