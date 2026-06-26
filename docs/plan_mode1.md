# Mode-1 — Plan d'implémentation (phasé par gates)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> Spec de référence : `docs/design_mode1.md`. Gate-0 ✅ déjà PASS (la rétine porte le signal courte-portée).

**Goal:** Une politique APPRISE drive-symétrique qui décide `(vx,ω)` pour survivre à plusieurs pulsions, à la
place du coût designed du planner — model-free PPO, perception rétine, warm-start BC, récompense = douleur
universelle, scalable sans retrain.

**Architecture:** Politique au command-cadence au-dessus du CPG+résidu `hexapod_v2` GELÉS (n'émet que `(vx,ω)`,
même contrat TCP → 0 changement Godot). Observation = proprio + tokens-pulsion drive-symétriques (perception
rétine couleur-gatée + niveau + valence). Warm-start par BC du planner, puis RL-finetune (PPO réutilisé).

**Tech Stack:** Python 3.12 (venv `env_pytorch_3.12`), PyTorch **CPU**, Godot headless (TCP JSON), modules
existants `ppo/update.py` + `ppo/rollout.py` (réutilisés tels quels), `serve_planner_command.py` (forké).

## Global Constraints (copiés verbatim — s'appliquent à CHAQUE task)
- **CPU OBLIGATOIRE** (GPU AMD = HIP crash). venv `env_pytorch_3.12/bin/python`, depuis la racine, `PYTHONPATH=python`, `GODOT_BIN="$(pwd)/tools/godot/godot"`.
- **PPO `--lr 1e-4`** (3e-4 diverge). Symétrie **OFF** (`sym_coef=0` ; `symmetry.py` hardcodée 18-D).
- **Substrat moteur GELÉ** : CPG + résidu `data/checkpoints/hexapod_v2/policy_best.pt`, inference-only. **0 changement Godot.**
- **Régime propre** : `SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5`.
- **Perception = rétine apprise, ZÉRO oracle** : la politique lit `retina` (144), JAMAIS `vision`/`vision_fine`/`vision_water` (radars-oracle). Le lien couleur↔pulsion (rouge=faim, bleu=soif) = définition-corps.
- **Récompense = douleur universelle** `−Σ_d (1−niveau_d)²` (zéro how-to). **Action bornée** `vx∈[0.55,0.75]`, `ω∈[-0.6,0.6]`.
- **Tuer un train** : `pkill -9 -f serve_mode1_collect ; pkill -9 -f train_mode1_ppo ; pkill -9 -f 'godot --path godot'` PUIS vérifier 0 restant. **Lancer en background = la commande python SEULE.**
- **BUT mesuré** : médiane survie multi-pulsions via le harness `baseline_multidrive_slot`. **À BATTRE : ~2300.**
- **Gater le cher derrière le pas-cher** : ne PAS construire la Phase 2 (RL) avant que Gate-1 (Phase 1, BC) passe. Un run raté = négatif informatif → STOP + escalade.
- **Tenir `architecture.json` à jour** (nœud `mode_1`) dans le commit qui change la conception/preuves.

---

## Structure des fichiers

**Phase 1 (Gate-1 — BC) :**
- Créer `python/sylvan/control/mode1/__init__.py` — package.
- Créer `python/sylvan/control/mode1/policy.py` — `DriveSymmetricPolicy` (le réseau) + bornes action.
- Créer `python/sylvan/control/mode1/obs.py` — `build_tokens(payload)` (assemble proprio + tokens-pulsion depuis la rétine couleur-gatée).
- Créer `python/scripts/serve_mode1.py` — serveur déterministe (fork de `serve_planner_command.py`, `plan()`→`policy`).
- Créer `train_mode1_bc.py` (racine) — entraînement BC (régression `obs→(vx,ω)`).
- Créer `collect_mode1_bc.sh` (racine) — collecte (planner multi-pulsions + rétine → buffer `(proprio,retina,energy,thirst,cmd)`).
- Créer `gate1_mode1_bc.sh` (racine) — éval Gate-1 (serveur Mode-1 BC dans le harness multi-pulsions).

**Phase 2 (Gate-2 — RL, gated derrière Gate-1) :**
- Modifier `godot/scripts/rl/reward_manager.gd` — objectif `survival_multi` (`−Σ(1−niveau)²`).
- Créer `python/scripts/serve_mode1_collect.py` — serveur stochastique + log transitions command-cadence.
- Créer `train_mode1_ppo.py` (racine) — RL (réutilise `ppo/update.py`+`ppo/rollout.py`, `action_dim=2`, sym off).

**Phase 3 (Gate-3 + Gate-S, gated derrière Gate-2) :** scripts de run long multi-seed + gate scalabilité.

---

## PHASE 1 — Gate-1 : le BC sur rétine atteint la baseline

> Objectif de phase : prouver que la politique drive-symétrique, nourrie à la rétine, peut **reproduire la
> navigation du planner** et survivre **≈ baseline (~2300)**. Valide obs + réseau + déploiement + que la rétine
> suffit. AUCUN RL, AUCUNE reward ici. **C'est le gate qui débloque (ou tue) la Phase 2.**

### Task 1 : La politique drive-symétrique + l'assemblage des tokens

**Files:**
- Create: `python/sylvan/control/mode1/__init__.py`
- Create: `python/sylvan/control/mode1/policy.py`
- Create: `python/sylvan/control/mode1/obs.py`
- Test: `python/sylvan/control/mode1/test_policy_sanity.py` (script-assert, idiome repo)

**Interfaces:**
- Produces: `DriveSymmetricPolicy(proprio_dim=132, ray_dim=36, hidden=128, action_dim=2)` ; méthode
  `forward(proprio:[B,132], tokens:[B,D,TOK]) -> mean:[B,2]` (mean non-bornée) ; `act(proprio, tokens) -> (vx,ω)`
  bornées via `map_action`. Attribut `log_std:[2]` (pour le RL plus tard).
- Produces: `TOK = 2 + 36 = 38` (niveau, valence, 36 profondeurs couleur-gatées).
- Produces: `build_tokens(payload:dict) -> (proprio:Tensor[132], tokens:Tensor[D,38], drive_meta:list)` où
  chaque drive actif (faim=rouge, soif=bleu si présent) donne un token.
- Produces: `map_action(mean:[B,2]) -> cmd:[B,2]` : `vx = 0.55 + 0.10*sigmoid`, `ω = 0.6*tanh`.

- [ ] **Step 1 : Écrire `policy.py`** (réseau permutation-invariant)

```python
# python/sylvan/control/mode1/policy.py
import torch
import torch.nn as nn

N_RAYS = 36
TOK = 2 + N_RAYS  # niveau, valence, 36 profondeurs couleur-gatées

def map_action(mean: torch.Tensor) -> torch.Tensor:
    """mean[...,0]→vx∈[0.55,0.75] ; mean[...,1]→ω∈[-0.6,0.6] (régime propre, design §2.5)."""
    vx = 0.55 + 0.10 * torch.sigmoid(mean[..., 0:1])
    om = 0.6 * torch.tanh(mean[..., 1:2])
    return torch.cat([vx, om], dim=-1)

class DriveSymmetricPolicy(nn.Module):
    """proprio + N tokens-pulsion → encodeur PARTAGÉ par token → pooling invariant (mean) → tronc → (vx,ω).
    Aucun slot 'énergie'/'soif' en dur : ajouter une pulsion = un token de plus, MÊMES poids (design §2.3)."""
    def __init__(self, proprio_dim=132, hidden=128, action_dim=2):
        super().__init__()
        self.token_enc = nn.Sequential(nn.Linear(TOK, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.proprio_enc = nn.Sequential(nn.Linear(proprio_dim, hidden), nn.SiLU())
        self.trunk = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))  # utilisé au RL (Phase 2)

    def forward(self, proprio, tokens):
        # proprio:[B,132] ; tokens:[B,D,TOK] (D variable). Pooling mean = invariant par permutation des drives.
        h_tok = self.token_enc(tokens).mean(dim=1)          # [B,hidden]
        h = torch.cat([self.proprio_enc(proprio), h_tok], -1)  # [B,2*hidden]
        return self.trunk(h)                                  # [B,action_dim] (mean non-bornée)

    @torch.no_grad()
    def act(self, proprio, tokens):
        return map_action(self.forward(proprio, tokens))
```

- [ ] **Step 2 : Écrire `obs.py`** (tokens depuis la rétine couleur-gatée — leçon Gate-0)

```python
# python/sylvan/control/mode1/obs.py
import torch

N_RAYS = 36
# couleurs-corps (food_manager.gd : rouge=bouffe, bleu=eau)
RED = "red"; BLUE = "blue"

def _color_gated_depths(retina, color):
    """retina = 144 floats = 36×[depth,R,G,B]. Retourne 36 profondeurs : depth si le rayon matche la couleur, sinon 1.0."""
    out = []
    for r in range(N_RAYS):
        d, R, G, B = retina[4*r], retina[4*r+1], retina[4*r+2], retina[4*r+3]
        if color == RED:
            hit = (R > G) and (R > B) and (R > 0.3) and (d < 0.999)
        else:  # BLUE
            hit = (B > R) and (B > G) and (B > 0.3) and (d < 0.999)
        out.append(d if hit else 1.0)
    return out

def build_tokens(payload: dict):
    """Construit (proprio[132], tokens[D,38], drive_meta) depuis le payload TCP du serveur.
    Drives actifs : faim (toujours, rouge) ; soif (si 'thirst' présent, bleu). Valence +1 (approcher-consommer)."""
    proprio = torch.tensor(payload["proprio"], dtype=torch.float32)
    retina = payload["retina"]
    assert len(retina) == 4 * N_RAYS, f"retina attendue {4*N_RAYS}, reçue {len(retina)} (SYLVAN_RETINA_PLANNER=1 ?)"
    toks, meta = [], []
    # token FAIM (rouge)
    e = float(payload.get("energy", 0.0)) / 100.0
    toks.append([e, 1.0] + _color_gated_depths(retina, RED)); meta.append("food")
    # token SOIF (bleu) — seulement si la pulsion existe
    if "thirst" in payload and payload["thirst"] is not None:
        t = float(payload["thirst"]) / 100.0
        toks.append([t, 1.0] + _color_gated_depths(retina, BLUE)); meta.append("water")
    tokens = torch.tensor(toks, dtype=torch.float32)  # [D,38]
    return proprio, tokens, meta
```

- [ ] **Step 3 : Écrire le sanity-check** (dims + invariance par permutation + bornes action)

```python
# python/sylvan/control/mode1/test_policy_sanity.py
# Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python python/sylvan/control/mode1/test_policy_sanity.py
import torch
from sylvan.control.mode1.policy import DriveSymmetricPolicy, map_action, TOK

def main():
    torch.manual_seed(0)
    pol = DriveSymmetricPolicy()
    proprio = torch.randn(4, 132)
    tokens = torch.randn(4, 2, TOK)  # 2 drives
    out = pol(proprio, tokens)
    assert out.shape == (4, 2), out.shape
    # INVARIANCE PAR PERMUTATION : échanger les 2 drives ne change pas la sortie
    out_swap = pol(proprio, tokens.flip(dims=[1]))
    assert torch.allclose(out, out_swap, atol=1e-6), "PAS invariant par permutation des drives !"
    # SCALABILITÉ : 1 drive et 3 drives passent sans erreur de poids (mêmes paramètres)
    assert pol(proprio, torch.randn(4, 1, TOK)).shape == (4, 2)
    assert pol(proprio, torch.randn(4, 3, TOK)).shape == (4, 2)
    # BORNES action
    cmd = map_action(torch.randn(1000, 2) * 5)
    assert (cmd[:, 0] >= 0.55).all() and (cmd[:, 0] <= 0.75).all(), "vx hors bornes"
    assert (cmd[:, 1] >= -0.6).all() and (cmd[:, 1] <= 0.6).all(), "ω hors bornes"
    print("OK: policy sanity (dims, invariance permutation, scalabilité 1/2/3 drives, bornes action)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4 : Lancer le sanity-check** — `PYTHONPATH=python ./env_pytorch_3.12/bin/python python/sylvan/control/mode1/test_policy_sanity.py` → attendu : `OK: policy sanity ...`. (Crée `__init__.py` vide si l'import échoue.)
- [ ] **Step 5 : Commit** — `git add python/sylvan/control/mode1/ && git commit -m "Mode-1 Task1 : politique drive-symetrique + tokens retine couleur-gatee (invariance permutation verifiee)"`

### Task 2 : Collecte des données BC (planner multi-pulsions + rétine) + vérif eau-dans-rétine

**Files:**
- Create: `collect_mode1_bc.sh`
- Confirm/Modify: `godot/scripts/main.gd` (replay-buffer logging — vérifier que `thirst` ET `retina0` ET `cmd` sont loggés par pas en mode planner multi-pulsions ; sinon ajouter `thirst` au log `wm`).

**Interfaces:**
- Produces: buffer `data/replay_buffer/mode1_bc_*/*.jsonl` où chaque ligne a `obs.proprio[132]`, `obs.energy`,
  `obs.thirst`, `wm.retina0[144]`, `wm.cmd[2]` (la commande du planner = LABEL BC).

- [ ] **Step 1 : Écrire `collect_mode1_bc.sh`** — lance le planner multi-pulsions, rétine ON, buffer-write ON, eau ON.

```bash
#!/usr/bin/env bash
# collect_mode1_bc.sh — collecte BC : le planner multi-pulsions (expert) drive l'env, on logge (obs retine, cmd).
set -euo pipefail
export GODOT_BIN="$(pwd)/tools/godot/godot"
export PYTHONPATH=python
PREFIX="${1:-mode1_bc_a}"; EPISODES="${2:-40}"; SEED="${3:-1}"
# régime propre + rétine + eau (multi-pulsions) + écriture buffer
export SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5
export SYLVAN_RETINA_PLANNER=1 SYLVAN_WM_USE_RETINA=1
export SYLVAN_WATER_COUNT=8 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05
export SYLVAN_REPLAY_PREFIX="$PREFIX" SYLVAN_SEED="$SEED"
# NB : réutilise le lanceur du harness multi-pulsions (planner serveur + godot), en mode collecte.
#     CONFIRMER à l'exécution le nom exact du flag d'écriture buffer dans baseline_multidrive_slot.sh / main.gd.
bash baseline_multidrive_slot.sh "$EPISODES" "$SEED"   # adapter : pointer le buffer-write, garder le planner comme driver
```

- [ ] **Step 2 : Lancer une MINI-collecte (2 ép) + vérifier dims, thirst, ET rayons bleus** (runtime water-in-retina) :

```bash
bash collect_mode1_bc.sh mode1_bc_smoke 2 1
PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<'PY'
import glob, json
f = sorted(glob.glob("data/replay_buffer/mode1_bc_smoke/*.jsonl"))[0]
rows = [json.loads(l) for l in open(f)]
r0 = rows[0]
print("proprio", len(r0["obs"]["proprio"]), "| retina0", len(r0["wm"]["retina0"]),
      "| has thirst", "thirst" in r0["obs"], "| has cmd", bool(r0["wm"].get("cmd")))
blue = red = 0
for r in rows:
    ret = r.get("wm", {}).get("retina0", [])
    for i in range(0, len(ret), 4):
        d, R, G, B = ret[i:i+4]
        if d < 0.999 and B > R and B > G and B > 0.3: blue += 1
        if d < 0.999 and R > G and R > B and R > 0.3: red += 1
print("red-ray hits", red, "| blue-ray hits", blue)
assert "thirst" in r0["obs"], "thirst PAS loggé → ajouter au log de main.gd"
assert blue > 0, "EAU PAS rendue dans la rétine → mettre l'eau sur la couche rétine (layer 8)"
print("OK: BC data shape + thirst + EAU-DANS-RÉTINE confirmée")
PY
```

Expected : `OK: BC data shape + thirst + EAU-DANS-RÉTINE confirmée`. **Si `blue-ray hits = 0`** → corriger le rendu rétine de l'eau (couche 8) AVANT de continuer (c'est la vérif différée du Gate-0). **Si `thirst` absent** → l'ajouter au log `wm`/`obs` dans `main.gd`.

- [ ] **Step 3 : Collecte complète** — `bash collect_mode1_bc.sh mode1_bc_a 40 1 && bash collect_mode1_bc.sh mode1_bc_b 40 2` (≥ ~30k transitions, drives variés via la dynamique du planner).
- [ ] **Step 4 : Commit** — `git add collect_mode1_bc.sh godot/scripts/main.gd && git commit -m "Mode-1 Task2 : collecte BC (planner multi-pulsions + retine), eau-dans-retine verifiee runtime"`

### Task 3 : Entraînement BC (régression `obs → (vx,ω)`)

**Files:**
- Create: `train_mode1_bc.py`
- Test: held-out match-rate inclus dans le script (idiome repo).

**Interfaces:**
- Consumes: `DriveSymmetricPolicy`, `build_tokens` (Task 1) ; buffers `mode1_bc_a/b` (Task 2).
- Produces: checkpoint `data/checkpoints/mode1_bc/policy.pt` = `{"model": state_dict, "meta": {...}}`.

- [ ] **Step 1 : Écrire `train_mode1_bc.py`** (charge buffers → tokens → régression MSE sur la commande du planner ; split par épisode 80/20 ; reporte le match-rate held-out).

```python
# train_mode1_bc.py — BC : régresse la politique drive-symétrique sur les commandes du planner (Task 2).
# Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python train_mode1_bc.py
import os, glob, json, torch, torch.nn as nn
from sylvan.control.mode1.policy import DriveSymmetricPolicy, map_action
from sylvan.control.mode1.obs import _color_gated_depths, N_RAYS, RED, BLUE

BUFS = os.environ.get("BUFS", "mode1_bc_a mode1_bc_b").split()
OUT = os.environ.get("OUT", "data/checkpoints/mode1_bc")

def load():
    P, T, Y, ep = [], [], [], []
    e = 0
    for buf in BUFS:
        for f in sorted(glob.glob(f"data/replay_buffer/{buf}/*.jsonl")):
            rows = [json.loads(l) for l in open(f)]; had = False
            for r in rows:
                w = r.get("wm", {})
                if not w.get("retina0") or not w.get("cmd"): continue
                ret = w["retina0"]
                food = [float(r["obs"].get("energy", 0))/100.0, 1.0] + _color_gated_depths(ret, RED)
                water = [float(r["obs"].get("thirst", 0))/100.0, 1.0] + _color_gated_depths(ret, BLUE)
                P.append(r["obs"]["proprio"]); T.append([food, water])
                Y.append([float(w["cmd"][0]), float(w["cmd"][1])]); ep.append(e); had = True
            if had: e += 1
    return (torch.tensor(P), torch.tensor(T), torch.tensor(Y), torch.tensor(ep), e)

def main():
    P, T, Y, ep, ne = load()
    print(f"BC: {len(Y)} transitions sur {ne} épisodes")
    ntr = max(1, int(0.8*ne)); tr = ep < ntr; te = ~tr
    pol = DriveSymmetricPolicy(); opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    for it in range(3000):
        opt.zero_grad()
        cmd = map_action(pol(P[tr], T[tr]))
        loss = ((cmd - Y[tr])**2).mean(); loss.backward(); opt.step()
        if it % 500 == 0: print(f"  it{it} mse={loss.item():.4f}")
    with torch.no_grad():
        pred = map_action(pol(P[te], T[te]))
    # match-rate : |Δvx|<0.05 ET |Δω|<0.1 (commande "proche" du planner)
    match = ((pred[:,0]-Y[te,0]).abs() < 0.05) & ((pred[:,1]-Y[te,1]).abs() < 0.1)
    mr = match.float().mean().item()
    print(f"[held-out] match-rate(|Δvx|<0.05,|Δω|<0.1)={100*mr:.0f}%  | ω-MAE={ (pred[:,1]-Y[te,1]).abs().mean():.3f}")
    os.makedirs(OUT, exist_ok=True)
    torch.save({"model": pol.state_dict(), "meta": {"proprio_dim": 132, "tok": T.shape[-1]}}, f"{OUT}/policy.pt")
    print(f">>> sauvé {OUT}/policy.pt | SUCCÈS Task3 si match-rate ≥ ~60% (BC a appris la nav du planner)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2 : Lancer le BC** — `PYTHONPATH=python ./env_pytorch_3.12/bin/python train_mode1_bc.py`. Attendu : mse qui descend, **match-rate held-out ≥ ~60 %**. (Si < ~40 % → l'obs ou le réseau n'apprend pas la nav → STOP, diagnostiquer avant Gate-1.)
- [ ] **Step 3 : Commit** — `git add train_mode1_bc.py && git commit -m "Mode-1 Task3 : entrainement BC (regression sur commande planner), match-rate held-out reporte"`

### Task 4 : Serveur Mode-1 déterministe (fork de `serve_planner_command.py`)

**Files:**
- Create: `python/scripts/serve_mode1.py`
- Reference: `python/scripts/serve_planner_command.py` (squelette TCP + chaînage résidu + reset à copier).

**Interfaces:**
- Consumes: `DriveSymmetricPolicy.act`, `build_tokens` (Task 1) ; checkpoint `mode1_bc/policy.pt` (Task 3).
- Produces: serveur TCP qui, sur le MÊME payload que `serve_planner_command`, répond `{"action":[18], "command":[vx,ω]}`.

- [ ] **Step 1 : Écrire `serve_mode1.py`** — repartir de `serve_planner_command.py`, GARDER : la boucle TCP, le chargement du résidu gelé, la construction de l'obs-résidu (commande-dans-vision, cf `serve_planner_command.py:~322`), le `reset`. REMPLACER : l'appel `planner.plan(...)` par :

```python
# au lieu de planner.plan(...) :
from sylvan.control.mode1.policy import DriveSymmetricPolicy, map_action
from sylvan.control.mode1.obs import build_tokens
# (chargement, une fois)
_ck = torch.load("data/checkpoints/mode1_bc/policy.pt", map_location="cpu", weights_only=False)
_pol = DriveSymmetricPolicy(proprio_dim=_ck["meta"]["proprio_dim"]); _pol.load_state_dict(_ck["model"]); _pol.eval()
# (par requête, à la place du plan())
proprio, tokens, _meta = build_tokens(payload)
cmd = _pol.act(proprio.unsqueeze(0), tokens.unsqueeze(0))[0]   # (vx,ω) bornées
vx, om = float(cmd[0]), float(cmd[1])
# PUIS : réutiliser TEL QUEL le chaînage résidu existant (obs-résidu avec (vx,ω) injectés) → action[18]
```

NB : Mode-1 **n'utilise PAS le WM** (perception = rétine directe). `serve_mode1.py` charge seulement le résidu + la politique BC. Garder le même `replan_every` que le planner (commande tenue sur la fenêtre).

- [ ] **Step 2 : Smoke test** — lancer `serve_mode1.py` sur un port, lui envoyer un payload factice (proprio 132 + retina 144 + energy + thirst), vérifier la réponse `{"action":[18], "command":[vx∈[0.55,0.75], ω∈[-0.6,0.6]]}`. (Petit client python inline.)
- [ ] **Step 3 : Commit** — `git add python/scripts/serve_mode1.py && git commit -m "Mode-1 Task4 : serveur deterministe (politique BC -> commande -> residu gele), contrat TCP inchange"`

### Task 5 : ⛓️ GATE-1 — le BC atteint la baseline (le verrou de la Phase 2)

**Files:**
- Create: `gate1_mode1_bc.sh` (fork de `baseline_multidrive_slot.sh` pointant `serve_mode1.py` au lieu de `serve_planner_command.py`).

- [ ] **Step 1 : Écrire `gate1_mode1_bc.sh`** — identique à `baseline_multidrive_slot.sh` (mêmes env : drain 0.05, eau, rétine, régime propre, ≥12 ép) mais lance `serve_mode1.py` comme serveur de décision. Parser de survie identique (`[Godot] Episode|Step|Energy|Thirst`).
- [ ] **Step 2 : Lancer Gate-1** — `bash gate1_mode1_bc.sh 12 1` → médiane de survie + `diag_death_multidrive.py` (taux de close `food_d<1m`).
- [ ] **Step 3 : VERDICT falsifiable**
  - **SUCCÈS** : médiane survie **≥ ~2000** (dans le bruit du planner ~2300) **ET** close ≈ planner. → la rétine suffit, le déploiement marche, le warm-start est viable → **débloque la Phase 2 (RL)**.
  - **KILL** : médiane **< ~1500** ou ne close jamais → obs/perception insuffisante (ou bug de déploiement) → **STOP**, diagnostiquer (rétine vs radar ? bug de chaînage résidu ? tokens ?) AVANT tout RL. Ne PAS enchaîner de tweaks à l'aveugle.
- [ ] **Step 4 : Commit + MAJ carte** — `git add gate1_mode1_bc.sh tools/archi_hud/architecture.json` ; mettre le résultat Gate-1 dans le nœud `mode_1` (`etat_detail`/`preuves`) ; valider la carte (`validate_architecture.py`) ; commit `"Mode-1 Gate-1 : BC vs baseline = <médiane> (PASS/KILL)"`.

---

## PHASE 2 — Gate-2 : un RL court montre un gradient de survie  *(GATED derrière Gate-1)*

> **Ne PAS construire avant que Gate-1 passe.** Détail step-by-step à écrire APRÈS Gate-1 (son code exact dépend
> du résultat). Design + critères falsifiables figés ici :

- **Task 6 — `survival_multi` reward** (`reward_manager.gd`) : `reward_t = −Σ_d (1−niveau_d)²` (faim+soif), terminal mort. Vérifier `max_episode_steps` ≥ 3000 en mode survie. **Aucun bonus manger/boire, aucun gradient d'approche** (douleur pure, design §2.4).
- **Task 7 — `serve_mode1_collect.py`** : fork de `serve_mode1.py` rendu STOCHASTIQUE (échantillonne `Normal(mean, exp(log_std))` avant `map_action`), + **log des transitions au command-cadence** : `(proprio, tokens, command, reward-fenêtre, done)` en JSONL pour le buffer PPO.
- **Task 8 — `train_mode1_ppo.py`** : réutilise `ppo/update.py` + `ppo/rollout.py` (`action_dim=2`, `sym_coef=0`, `--lr 1e-4`, GAE γ=0.99 λ=0.95) ; **warm-start depuis `mode1_bc/policy.pt`** ; exploration = **resets à drives randomisés** (randomiser `SYLVAN_INIT_ENERGY/THIRST` par épisode) + entropie.
- **Task 9 — ⛓️ GATE-2** : run RL COURT (budget borné) depuis le warm-start BC.
  - **SUCCÈS** : médiane **> BC (+≥200)** OU taux « manger-quand-faim » qui monte, **sans divergence** (KL<0.03, std stable). → débloque Phase 3.
  - **KILL** : survie s'effondre sous BC et y reste, ou std/KL divergent → **STOP + escalade** (négatif informatif, pas d'enchaînement de tweaks). Tuer tôt si divergence claire.

---

## PHASE 3 — Gate-3 (run long) + Gate-S (scalabilité)  *(GATED derrière Gate-2)*

- **Task 10 — ⛓️ GATE-3 (run long, le BUT)** : RL long, **multi-seed ≥ 3**. **SUCCÈS** : médiane multi-pulsions **> 2300**, closed-loop → promotion live (serveur Mode-1 remplace le planner ; planner gardé en fallback). **KILL** : ne dépasse pas 2300 malgré convergence → négatif informatif, STOP + escalade.
- **Task 11 — ⛓️ GATE-S (scalabilité no-retrain, falsifie §3/§4)** : politique **GELÉE** + une **3ᵉ pulsion jamais vue** au test (un 3ᵉ token, p.ex. fatigue+repos OU une 2ᵉ ressource consommable). **SUCCÈS** : survie maintenue avec 3 pulsions **sans retrain**. (Le gate EST le test.)

---

## Self-review (couverture du spec)

- **Action (vx,ω) sur substrat gelé** → Task 1 (map_action, bornes) + Task 4 (chaînage résidu, 0 changement Godot). ✅
- **Perception rétine apprise, zéro oracle** → Task 1 `obs.py` (couleur-gaté, jamais radar) + Task 2 (rétine ON). ✅
- **Politique drive-symétrique (scalable)** → Task 1 (pooling invariant, vérifié) + Task 11 (Gate-S). ✅
- **Warm-start BC** → Tasks 2-3 (collecte+train) ; **RL model-free** → Tasks 7-8. ✅
- **Récompense douleur universelle** → Task 6. **Exploration (off-policy)** → Task 8 (resets randomisés+entropie). ✅
- **Critères falsifiables + gates** → Gate-1 (Task 5), Gate-2 (Task 9), Gate-3 (Task 10), Gate-S (Task 11). ✅
- **BUT mesuré ~2300** → Gate-1/3 (harness multi-pulsions). ✅
- **Caveats** : eau-dans-rétine vérifiée runtime (Task 2 Step 2) ; signatures exactes (résidu-obs, flag buffer-write) **à confirmer à l'exécution** contre `serve_planner_command.py`/`main.gd` (notées dans les tasks, pas des placeholders : l'action de repli est spécifiée).

> **Note d'honnêteté (§2)** : ce plan est détaillé et exécutable pour la **Phase 1 (Gate-1)** ; les Phases 2-3
> sont volontairement au niveau design+critères (gated). On n'écrit le code exact du RL qu'APRÈS que le BC ait
> prouvé le chemin — sinon on construit le cher avant que le pas-cher décide (anti-§1).
