# Mémoire spatiale — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que l'entité se souvienne de la position ego d'un objet quand elle ne le voit plus (dead-reckoning + re-grounding), validé gratuitement AVANT tout build, puis bâti comme belief-slot côté serveur.

**Architecture:** Le belief (coordonnée ego de l'objet) vit dans le SERVEUR entre replans ; il est dead-reckoné par l'ego-motion RÉELLE (convention géométrique, proven test5 +0.98) et re-groundé quand l'objet est re-perçu (gate de saillance du slot_head). L'opérateur de transport géométrique est celui de test5/test6 (translate −déplacement, rotate −Δyaw). Le belief sert de slot t0 au planner WM-slot quand l'objet n'est pas vu.

**Tech Stack:** Python 3.12 (`env_pytorch_3.12/bin/python`, CPU obligatoire), PyTorch, scripts diag au format `diag_*` à la racine (convention projet : self-check par `assert`, PAS pytest), Godot headless pour les gates closed-loop (NATIF).

## Global Constraints

- venv : `env_pytorch_3.12/bin/python`, depuis la RACINE, `PYTHONPATH=python`. CPU OBLIGATOIRE (GPU AMD = HIP crash).
- Dims hexapode FIGÉES : proprio=132, action=18, obs(WM)=145 ou 277 (rétine). Ne pas les toucher.
- Calib transport du WM vivant = `slot_calib=(1,-1,-1)` (kf,kl,ky), fed la displacement RÊVÉE. Le dead-reckoning de la MÉMOIRE utilise l'ego-motion RÉELLE → convention GÉOMÉTRIQUE (test5 `transport`). Ne PAS confondre les deux conventions (cf Notes Task 1).
- Forager vivant à NE PAS casser : `wm_objcentric_s1` / `run_forage_wmslot.sh` (sans occlusion, comportement byte-proche exigé).
- Principe N°1 : le test gratuit Task 1 GATE tout le reste. Un KILL en Task 1 = STOP + escalade, PAS d'enchaînement de tweaks ni de build.
- Principe N°2 : mesurer le BUT honnête (drift en m/° vs vérité-terrain `food_rel0`), jamais un proxy.
- `tools/archi_hud/architecture.json` à jour DANS le commit qui change le module (valider via `tools/archi_hud/validate_architecture.py`).
- Tuer proprement : `pkill -9 -f serve_planner_command ; pkill -9 -f 'godot --path godot'` + vérifier 0 restant.

---

## Task 1 (DÉCISIF, GRATUIT) : diagnostic de dérive `diag_slot_memory_drift.py`

C'est LE gate. Pur offline sur buffers existants. Aucune dépense en aval n'est autorisée tant qu'il n'est pas PASSÉ contre les critères pré-enregistrés.

**Files:**
- Create: `diag_slot_memory_drift.py` (racine, conforme à `diag_test5_proprio_egomotion.py` / `diag_test6_slot_transport.py`)

**Interfaces (Produces — réutilisés par le build) :**
- `wrap(a: float) -> float` — wrap angle dans (−π, π].
- `egomotion_from_torso(t0: list[3], t1: list[3]) -> tuple[float,float,float]` — torso=(x,z,yaw) → `(dyaw, dfwd, dlat)`, convention géométrique de test5.
- `transport_geom(p: list[2], dyaw: float, dfwd: float, dlat: float) -> list[2]` — `p` ego-coord à t → ego-coord à t+1 d'un point monde-statique (translate −déplacement, rotate −Δyaw). Identique verbatim à test5/test6.

**Notes de convention (à NE PAS rater) :**
La mémoire dead-reckone par l'ego-motion RÉELLE (torso/proprio). La bonne opération est `transport_geom` ci-dessous (proven test5 : transport du bearing par ego-motion vraie). Le WM vivant, lui, utilise `transport_slot(calib=(1,-1,-1))` fed la displacement RÊVÉE (convention displacement-head). Pour relier les deux : `transport_slot(p, disp=(dfwd, dlat, dyaw), calib=(1,-1,-1))` ≡ `transport_geom(p, dyaw=-dyaw, dfwd=dfwd, dlat=-dlat)`. La variante (b) ci-dessous (round-trip sur ego-motion VRAIE) est le juge : elle DOIT être quasi-parfaite à petit N si la convention est correcte ; sinon on a trouvé un bug de convention GRATUITEMENT avant tout build.

- [ ] **Step 1 : Écrire le self-check des fonctions pures (mode `--selfcheck`)**

Crée le squelette du script avec un mode self-check qui teste les invariants (convention projet : asserts, pas pytest) :

```python
"""diag_slot_memory_drift.py — MÉMOIRE SPATIALE, gate gratuit décisif (CLAUDE.md §1).
Mesure la dérive d'un belief-slot dead-reckoné par l'ego-motion RÉELLE (torso) pendant une
occlusion artificielle, vs vérité-terrain food_rel0. Décide si la mémoire est FAISABLE avant tout build.
Usage:
  PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_slot_memory_drift.py --selfcheck
  BUFS="retina_wm_a retina_wm_b" PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_slot_memory_drift.py
"""
import os, sys, json, glob, math
import torch

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

def egomotion_from_torso(t0, t1):
    x0, z0, yaw0 = t0; x1, z1, yaw1 = t1
    dyaw = wrap(yaw1 - yaw0)
    dx, dz = x1 - x0, z1 - z0
    dfwd = dx * math.sin(yaw0) + dz * math.cos(yaw0)
    dlat = dx * math.cos(yaw0) - dz * math.sin(yaw0)
    return dyaw, dfwd, dlat

def transport_geom(p, dyaw, dfwd, dlat):
    px, pz = p[0] - dlat, p[1] - dfwd
    ca, sa = math.cos(-dyaw), math.sin(-dyaw)
    return [ca * px - sa * pz, sa * px + ca * pz]

def selfcheck():
    # (1) agent immobile → belief inchangé
    p = [0.4, 2.1]
    assert max(abs(a - b) for a, b in zip(transport_geom(p, 0.0, 0.0, 0.0), p)) < 1e-9
    # (2) ROUND-TRIP convention : un point monde-statique vu à t0, après un pas réel de l'agent,
    #     transporté par l'ego-motion de CE pas, doit retomber sur sa position ego réelle à t1.
    #     On simule : agent à pose torse t0 puis t1 ; food à position MONDE fixe.
    import random
    random.seed(0)
    for _ in range(2000):
        yaw0 = random.uniform(-math.pi, math.pi)
        t0 = [random.uniform(-3, 3), random.uniform(-3, 3), yaw0]
        t1 = [t0[0] + random.uniform(-0.3, 0.3), t0[1] + random.uniform(-0.3, 0.3), wrap(yaw0 + random.uniform(-0.4, 0.4))]
        fw = [random.uniform(-4, 4), random.uniform(-4, 4)]  # food MONDE
        def to_ego(t, f):
            dx, dz = f[0] - t[0], f[1] - t[1]; y = t[2]
            return [dx * math.cos(y) - dz * math.sin(y), dx * math.sin(y) + dz * math.cos(y)]
        p0 = to_ego(t0, fw); p1 = to_ego(t1, fw)
        dyaw, dfwd, dlat = egomotion_from_torso(t0, t1)
        pred = transport_geom(p0, dyaw, dfwd, dlat)
        err = math.hypot(pred[0] - p1[0], pred[1] - p1[1])
        assert err < 1e-6, f"round-trip cassé: err={err} (convention egomotion/transport)"
    print("[selfcheck] OK — convention egomotion↔transport_geom validée (round-trip < 1e-6).")

if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck(); sys.exit(0)
    main()
```

⚠️ La fonction `to_ego` du self-check encode l'hypothèse « torso=(x,z,yaw), food_rel0 = food monde projetée en repère ego par Rot(yaw) ». Si le round-trip échoue malgré une convention transport correcte, c'est que la définition (x,z,yaw) / le signe de food_rel0 du buffer diffère → vérifier contre `egomotion_from_torso` de test5 (identique) et la déf `dfwd=dx·sin+dz·cos` ; ajuster `to_ego` pour matcher LA convention du buffer (ne pas tordre le transport).

- [ ] **Step 2 : Lancer le self-check, vérifier qu'il PASSE**

Run: `PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_slot_memory_drift.py --selfcheck`
Expected: `[selfcheck] OK — convention egomotion↔transport_geom validée (round-trip < 1e-6).`
Si FAIL : la convention est cassée → corriger AVANT toute mesure (sinon les chiffres de drift sont faux). C'est le 1er garde-fou gratuit.

- [ ] **Step 3 : Implémenter le chargement des buffers + l'encodeur de slot**

Ajoute (au-dessus de `main`) le chargement (calqué sur `diag_test6_slot_transport.load_eps`, + `torso0`) et le slot-encodeur (le WM vivant `wm_objcentric_s1`, apples-to-apples) :

```python
DEVICE = "cpu"
BUFS = os.environ.get("BUFS", "retina_wm_a retina_wm_b").split()
WM_CKPT = os.environ.get("WM_CKPT", "data/checkpoints/wm_objcentric_s1/wm_best.pt")

def load_eps():
    eps = []
    for buf in BUFS:
        for f in sorted(glob.glob(f"data/replay_buffer/{buf}/*.jsonl")):
            seq = []
            for line in open(f):
                r = json.loads(line); w = r.get("wm", {})
                ret = w.get("retina0"); fr = w.get("food_rel0"); t0 = w.get("torso0")
                if not ret or not fr or not t0:
                    continue
                seq.append({"retina": ret, "food": [float(fr[0]), float(fr[1])],
                            "vis": float(fr[2]), "torso": [float(t0[0]), float(t0[1]), float(t0[2])]})
            if len(seq) > 50:
                eps.append(seq)
    return eps

def load_encoder():
    from sylvan.models.command_wm import CommandWorldModel
    ck = torch.load(WM_CKPT, map_location=DEVICE, weights_only=False)
    m = ck["meta"]
    wm = CommandWorldModel(obs_dim=m["obs_dim"], proprio_dim=m["proprio_dim"],
                           action_dim=m["action_dim"], with_slot=True)
    wm.load_state_dict(ck["state_dict"]); wm.eval()
    return wm

@torch.no_grad()
def encode_slot_from_retina(wm, retina):
    # même chemin que command_wm.encode_slot, mais on a la rétine isolée → slot_encoder.positions
    r = torch.tensor(retina, dtype=torch.float32).unsqueeze(0)
    return wm.slot_encoder.positions(r)[0, 0, :].tolist()  # [x_right, z_fwd]
```

- [ ] **Step 4 : Implémenter la mesure de dérive (cœur du diagnostic)**

```python
NS = [5, 10, 20, 40]           # pas-depuis-vu mesurés
STRIDE_K = 15                  # échantillonnage des points "dernier-vu" k le long de chaque épisode

def drift_runs(eps, wm):
    # structures: results[variant][baseline][N] = list d'erreurs (bearing°, pos m)
    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    maxN = max(NS)
    for seq in eps:
        for k in range(0, len(seq) - maxN - 1, STRIDE_K):
            if seq[k]["vis"] < 0.5:
                continue  # k doit être un pas où l'objet EST vu (dernier-vu honnête)
            # belief(k) selon variante
            bk_a = encode_slot_from_retina(wm, seq[k]["retina"])   # (a) perception réaliste
            bk_b = list(seq[k]["food"])                            # (b) vérité-terrain (borne haute)
            beliefs = {"a_encode": list(bk_a), "b_truth": list(bk_b)}
            frozen  = {"a_encode": list(bk_a), "b_truth": list(bk_b)}  # baseline gelé (jamais transporté)
            for n in range(1, maxN + 1):
                dyaw, dfwd, dlat = egomotion_from_torso(seq[k + n - 1]["torso"], seq[k + n]["torso"])
                for v in beliefs:
                    beliefs[v] = transport_geom(beliefs[v], dyaw, dfwd, dlat)
                if n in NS:
                    truth = seq[k + n]["food"]
                    bt = math.atan2(truth[0], truth[1])
                    for v in beliefs:
                        # dead-reckoné
                        bp = math.atan2(beliefs[v][0], beliefs[v][1])
                        agg[v]["deadreckon"][n].append((abs(math.degrees(wrap(bp - bt))),
                                                        math.hypot(beliefs[v][0]-truth[0], beliefs[v][1]-truth[1])))
                        # gelé (mémoire statique)
                        fp = math.atan2(frozen[v][0], frozen[v][1])
                        agg[v]["frozen"][n].append((abs(math.degrees(wrap(fp - bt))),
                                                    math.hypot(frozen[v][0]-truth[0], frozen[v][1]-truth[1])))
    return agg

def median(xs):
    xs = sorted(xs); n = len(xs)
    return float("nan") if n == 0 else (xs[n//2] if n % 2 else 0.5*(xs[n//2-1]+xs[n//2]))
```

(Baseline « aveugle » (ii) = ce que le live renvoie sans objet visible : non-mesurable offline proprement car la perception sur une rétine vide n'a pas de vérité ; on le DOCUMENTE comme « belief inutilisable » — le contraste pertinent quantitatif est dead-reckoné vs gelé.)

- [ ] **Step 5 : Implémenter `main()` — agrégation + verdict pré-enregistré**

```python
def main():
    eps = load_eps()
    print(f"épisodes={len(eps)} ; frames={sum(len(e) for e in eps)} ; bufs={BUFS}")
    wm = load_encoder()
    agg = drift_runs(eps, wm)
    print("\nMÉDIANES (bearing MAE °, position MAE m) par variante / N :")
    PASS_BRG, PASS_POS, PASS_N = 20.0, 0.5, 30
    verdicts = []
    for v in ("b_truth", "a_encode"):
        print(f"\n  variante {v}:")
        for n in NS:
            dr = agg[v]["deadreckon"][n]; fr = agg[v]["frozen"][n]
            drb, drp = median([e[0] for e in dr]), median([e[1] for e in dr])
            frb, frp = median([e[0] for e in fr]), median([e[1] for e in fr])
            print(f"    N={n:>2}: dead-reckon brg={drb:5.1f}° pos={drp:4.2f}m | gelé brg={frb:5.1f}° pos={frp:4.2f}m | n={len(dr)}")
        # verdict sur la variante réaliste (a) au seuil N
        if v == "a_encode":
            dr = agg[v]["deadreckon"]; fr = agg[v]["frozen"]
            # interpolation simple : on prend le plus grand N <= PASS_N mesuré
            ncheck = max([n for n in NS if n <= PASS_N])
            drb = median([e[0] for e in dr[ncheck]]); drp = median([e[1] for e in dr[ncheck]])
            frp = median([e[1] for e in fr[ncheck]])
            beats_frozen = drp < 0.8 * frp
            ok = (drb < PASS_BRG) and (drp < PASS_POS) and beats_frozen
            print(f"\n  >>> VERDICT (variante réaliste a_encode, N={ncheck}): "
                  f"brg={drb:.1f}°(<{PASS_BRG}) pos={drp:.2f}m(<{PASS_POS}) bat_gelé={beats_frozen} → "
                  f"{'PASS — build autorisé' if ok else 'FAIL/KILL — STOP + escalade (CLAUDE.md §1)'}")
```

- [ ] **Step 6 : Lancer sur les buffers réels, capturer les chiffres**

Run: `BUFS="retina_wm_a retina_wm_b" PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_slot_memory_drift.py`
Expected: tableau de médianes par variante/N + une ligne VERDICT explicite.
Lecture diagnostique : si (b_truth) dérive AUSSI à petit N → bug convention/opérateur (revenir Step 1-2). Si seul (a_encode) dérive → erreur perception t0 (bord de champ), pas le transport.

- [ ] **Step 7 : Statuer contre les critères pré-enregistrés (HARD GATE)**

- **SUCCÈS** : `a_encode` dead-reckoné garde **bearing MAE < 20° ET position MAE < 0.5 m à N≈30**, ET bat le gelé (pos < 0.8× gelé). → build autorisé (Tasks 2+).
- **KILL** : position MAE > ~1 m en < 15 pas aveugles, OU dead-reckon ≈ gelé. → **STOP. Écrire le négatif informatif** (la cause = ego-motion/calib, pas la mémoire) dans la mémoire auto + escalader à l'owner. NE PAS lancer Tasks 2+. NE PAS enchaîner des tweaks.
- Reporter le verdict + le tableau à l'owner avant de continuer.

- [ ] **Step 8 : Commit**

```bash
git add diag_slot_memory_drift.py docs/superpowers/plans/2026-06-25-memoire-spatiale.md
git commit -m "Mémoire spatiale Task 1 : gate gratuit diag_slot_memory_drift (dérive belief dead-reckoné vs vérité-terrain)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

> **GATE DUR** — Les Tasks 2 à 5 ne s'exécutent QUE si Task 1 = PASS. Leur forme exacte (notamment la source d'ego-motion live et la politique de re-grounding) est informée par le profil de dérive mesuré en Task 1. Si Task 1 = KILL, ce plan s'arrête ici et on escalade.

## Task 2 (GATED) : source d'ego-motion live `proprio→egomotion`

Au déploiement il n'y a pas la pose torse en oracle ; l'ego-motion vient du proprio (F2 : proprio→ego-motion R²≈+0.98). Choix tranché ICI sur preuve, sans run long.

**Files:**
- Create: `python/sylvan/models/egomotion_head.py` (petite tête `EgomotionHead` : proprio[132] → (dyaw,dfwd,dlat)) OU module analytique si le proprio expose déjà les vitesses.
- Create: `train_egomotion_head.py` (racine) — régression sur `retina_wm_a/b` (cibles = `egomotion_from_torso` entre frames consécutives), cross-val par épisode.
- Test: self-check `--selfcheck` dans `train_egomotion_head.py` (R² test ≥ 0.9 par composante, sinon KILL).

**Interfaces:**
- Consumes: `egomotion_from_torso` (Task 1) pour fabriquer les cibles.
- Produces: `EgomotionHead.predict(proprio: list[132]) -> (dyaw, dfwd, dlat)` ; checkpoint `data/checkpoints/egomotion_head/best.pt`.

- [ ] **Step 1** : Décider analytique-vs-tête en inspectant le proprio. Test gratuit `diag_egomotion_source.py` : corr(proprio-dérivé, ego-motion vraie). Si une combinaison analytique (vitesses linéaire/angulaire × dt) donne déjà corr ≥ 0.95 → ANALYTIQUE (pas de tête, pas de train). Sinon → tête.
- [ ] **Step 2** : Implémenter le module retenu + son `--selfcheck` (R²/corr test ≥ 0.9 par composante).
- [ ] **Step 3** : Lancer le self-check, vérifier le seuil. KILL si < 0.9 (sinon le dead-reckoning live dérivera) → escalade.
- [ ] **Step 4** : Commit (+ `architecture.json` si un module naît).

## Task 3 (GATED) : module `SlotMemory` côté serveur

**Files:**
- Modify: `python/scripts/serve_planner_command.py` (état + boucle belief, ~3 endroits : init, per-tick update dans `predict_full`, passage au planner).
- Modify: `python/sylvan/control/planning/command_planner.py:198-229` (branche `plan_wm_slot` : accepter un `slot_belief` override du slot t0).

**Interfaces:**
- Consumes: `transport_geom` (Task 1), `EgomotionHead.predict` ou analytique (Task 2), `wm.slot_encoder` + gate de saillance, `planner.plan(..., slot_belief=...)`.
- Produces: comportement serveur stateful : belief persistant fourni au planner.

- [ ] **Step 1** : Écrire un self-check serveur hors-godot (`--selfcheck` dans serve) : séquence synthétique d'obs (objet visible 10 pas, occulté 20 pas) → asserts : (i) sans occlusion belief ≈ perception (re-ground) ; (ii) sous occlusion belief = dead-reckon (continue de bouger cohéremment) ; (iii) re-apparition → re-ground propre (saut borné).
- [ ] **Step 2** : Run self-check, vérifier qu'il échoue (SlotMemory absent).
- [ ] **Step 3** : Implémenter `SlotMemory` (dead-reckon par ego-motion live → re-ground si saillant, sinon garde) + brancher dans `predict_full` + l'override `slot_belief` dans `command_planner.plan_wm_slot`.
- [ ] **Step 4** : Run self-check, vérifier qu'il passe.
- [ ] **Step 5** : NON-RÉGRESSION GRATUITE : avec re-grounding chaque tick (pas d'occlusion), le belief = perception → vérifier que la branche `plan_wm_slot` renvoie la MÊME commande qu'avant (diff numérique nul/epsilon sur un batch d'obs enregistrées). Sinon le filet 360° est cassé.
- [ ] **Step 6** : Commit (+ `architecture.json` : module `memoire_spatiale` état partiel/échafaudage).

## Task 4 (GATED, NATIF) : masque d'occlusion + gates closed-loop

**Files:**
- Modify: serveur ou `perception.gd` — masque rétine paramétrable `SYLVAN_OCCLUDE_*` (zéro les rayons d'un objet une fois vu, ou dans une zone d'angle).
- Create: `diag_nav_ab_memory.sh`, `run_forage_memory.sh` (variantes occlusion de `diag_nav_ab_wmslot.sh` / `run_forage_wmslot.sh`).

- [ ] **Step 1** : Implémenter le masque d'occlusion + un flag mémoire ON/OFF.
- [ ] **Step 2** : (NATIF, owner) `bash diag_nav_ab_memory.sh` AVEC occlusion, mémoire ON vs OFF → engagement/approche. Critère : mémoire ON ≥ OFF (atteindre un objet vu-puis-occulté).
- [ ] **Step 3** : (NATIF, owner) `bash run_forage_memory.sh` AVEC occlusion, ON vs OFF → survie/repas. Critère : ON ≥ OFF.
- [ ] **Step 4** : SANS occlusion : non-régression vs forager vivant (engagement 15/16, foraging méd ~860+).
- [ ] **Step 5** : Commit des scripts + résultats.

## Task 5 (GATED) : promotion + carte

- [ ] **Step 1** : Si (et seulement si) PUR ET ≥ baseline aux gates Task 4 → promouvoir (mémoire ON par défaut dans `run_forage_wmslot.sh`).
- [ ] **Step 2** : Mettre `architecture.json` à jour (module `memoire_spatiale` : état, role/apporte, preuves [chiffres Task 1 + Task 4], limites [drift au-delà de N, bord de champ], code, focus → suivant = cône de vision). Valider `validate_architecture.py`.
- [ ] **Step 3** : Écrire la mémoire auto (`sylvan-objectcentric-pur.md` : mémoire spatiale faite, chiffres, leçon). Commit.

---

## Self-Review (auteur)

- **Couverture spec** : §1 où-vit-la-mémoire → Task 3. §2 test gratuit → Task 1 (intégral). §3 SlotMemory → Tasks 2+3. §4 gate occlusion → Task 4. Promotion+carte → Task 5. Hors-scope (cône, internalisation WM, multi-type) → laissés hors plan (différés). ✅
- **Placeholders** : Task 1 entièrement codée (le seul travail certain). Tasks 2-5 GATÉES derrière Task 1 et volontairement non sur-spécifiées en code car leur détail dépend du verdict/profil de dérive de Task 1 — c'est de la DISCIPLINE de gating (CLAUDE.md §1), pas de la paresse ; chaque step y est une action concrète testable.
- **Cohérence des types** : `egomotion_from_torso → (dyaw,dfwd,dlat)` et `transport_geom(p,dyaw,dfwd,dlat)` cohérents Task 1↔3 ; `slot_belief` override nommé identiquement Task 3 (serve↔planner). ✅
