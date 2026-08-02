---
title: "Diagnostic perception→conséquence — session 2026-07-30"
date: 2026-07-30
status: PAUSÉ — à reprendre
next: Option A (deep supervision position, --w-position dans le retrain WM)
---

# Diagnostic perception→conséquence en forêt — 2026-07-30

## Le problème initial

Le slot cosinus localise la nourriture à **2,18 m** d'erreur dans le monde forêt
(contre 0,24 m dans le monde typé). La question : pourquoi et comment corriger ?

## Ce qu'on a essayé et appris

### 1. La cause-racine n'est PAS le buisson-marqueur
Le `SYLVAN_FOOD_BUSH` est DÉSACTIVÉ dans FORET_V1. La nourriture apparaît bien
avec ses 4 teintes sur la rétine. Le problème est que les TRONCS D'ARBRES avec
`forest_appearance_var=0.15` produisent des couleurs qui ressemblent à la
nourriture — 39,9 % des rayons d'arbres tombent dans le même espace (depth,R,G,B)
que les rayons de nourriture. Un seuil cosinus ne peut pas les séparer.

### 2. Classification par rayon : échec structurel (39,9 % de chevauchement 4D)

| Approche | Erreur médiane | < 0,5 m | Pourquoi |
|---|---|---|---|
| Cosinus (codé-main) | 2,18 m | 5 % | 1 requête rouge pour 4 teintes + bruit troncs |
| MLP 4→1 par rayon (Option A) | 2,09 m | 15 % | 39,9 % chevauchement 4D → inséparable sans contexte inter-rayons |
| Token-score BCE (supervisé) | 2,10 m | — | Même cause, même avec self-attention |
| Token-score JEPA (transport) | 1,50 m | 18 % | Apprend un peu, plafonne sans gate sémantique |

Leçon : la classification par rayon NE PEUT PAS marcher sans contexte inter-rayons.
L'encodeur d'attention (99,7 % précision type) le fait parce qu'il voit les 36
rayons ENSEMBLE — mais cette information est dans le LATENT, pas dans le slot.

### 3. Le latent PORTE la position

| Sonde | Erreur médiane | < 0,5 m | Gap linéaire/MLP |
|---|---|---|---|
| Linéaire (128→2) | 2,05 m | 5 % | — |
| MLP (128→64→32→2) | **0,55 m** | 45 % | **3,7×** |

- La sonde MLP donne 0,55 m : preuve que l'information spatiale EST dans le latent
- Le gap 3,7× entre linéaire et MLP est le symptôme clé (voir §littérature)

### 4. position_head : 0,55 m mais CONTAMINÉE

Entraînée sur `L2(latent, food_rel0)`, elle code implicitement "nourriture = rouge/rose".
Change les couleurs → elle casse. Violation du §3 CLAUDE.md. **Ne pas promouvoir.**

Checkpoint contaminé : `data/checkpoints/wm_foret_attn_hue_pos/wm_best.pt`
Checkpoint PUR correspondant : `data/checkpoints/wm_foret_attn_hue/wm_best.pt`

### 5. Bootstrap position_head → value_head : ÉCHEC de généralisation

La position_head a généré 63 repas en 6 épisodes (pattern "approche→mange" prouvé).
La value_head entraînée sur `ate` atteint AUC 0,78 (teacher-forced) et AUC 0,78 (dream).
Mais `plan_latent` + value_head → **0 repas**. 

Cause : distributional shift. La value_head est entraînée sur les commandes EXÉCUTÉES
pendant la collecte, mais doit scorer les 117 commandes CANDIDATES du planner — des
latents rêvés sous des commandes jamais vues pendant l'entraînement.

### 6. Ce que dit la littérature (juillet 2026)

**LeCun (LeWM, 2026)** : la position est **linéairement décodable** du latent JEPA si
et seulement si on utilise SIGReg (forcer des latents Gaussiens). Preuve mathématique
que le latent devient une rotation linéaire de l'état du monde.

**Zhang (ICLR 2026)** : confirme l'hypothèse de représentation linéaire — les world
models encodent la position comme des directions ~linéaires dans le latent. MLP ≈ linéaire.

**Zahorodnii (ICLR 2025)** : la "deep supervision" (ajouter une perte de position au
WM) AMÉLIORE le modèle — les features NON supervisées deviennent aussi plus décodables,
et ça double la capacité effective.

**LeMario** : "An external objective still has to tell [the JEPA] what matters."
Le JEPA apprend ce qui est PRÉDICTIBLE, pas ce qui est PERTINENT.

### 7. Diagnostic du gap linéaire/MLP

Notre WM utilise VICReg (pas SIGReg). Conséquence : le latent est sain (eff_rank 38/128)
mais NON-FACTORISÉ — les dimensions ne correspondent pas à des variables physiques
indépendantes. Le gap 3,7× entre sonde linéaire et MLP est le symptôme direct.

## La prochaine étape (Option A — validée par la littérature)

**Ajouter `--w-position` dans la perte JEPA** (deep supervision, cf Zahorodnii).

Principe : le `food_rel0` du corpus sert de cible pour une sonde LINÉAIRE
`Linear(128, 2)` branchée sur le latent. La perte est `L2(linear_probe(latent), food_rel0)`.

Pourquoi c'est propre :
- La position est une OBSERVATION du corpus (écrite par Godot), pas un oracle externe
- La sonde est LINÉAIRE (pas MLP) — elle force le latent à s'organiser linéairement
  (aligné avec LeCun 2026)
- La littérature montre que ça AMÉLIORE les autres features (pas de contamination)
- Le gradient TRAVERSE l'encodeur → il apprend à encoder la position linéairement
- Après retrain, le slot lit `Linear(128,2)(latent)` → JEPA-pur

Résultat attendu : sonde linéaire < 1 m (contre 2,05 m actuel).

Coût : ~40 min de retrain WM. Commande à préparer :

```bash
PYTHONPATH=python SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python \
  -m scripts.train_wm_command \
  --runs data/replay_buffer/foret_v1_planner ... \
  --out data/checkpoints/wm_foret_attn_hue_poslin \
  --proprio-dim 133 --retina-attention \
  --w-position 5.0 \
  --epochs 20 --seq-len 64 --lr 1e-4 \
  ... (flags JEPA standard)
```

## Fichiers modifiés cette session

### Code (à conserver)
- `python/sylvan/models/slot_head.py` : support `token_features` (score sur tokens 64D), `affinity_net` (MLP local), `_attend()` accepte `tokens` ou rien
- `python/sylvan/models/encoders.py` : `RetinaAttentionEncoder.affinity_head` + `affinity_attn` (self-attention inter-rayons) + `get_affinity()` + `_extract_tokens()`
- `python/sylvan/models/command_wm.py` : `with_position_head` (sonde, PAS à servir), `slot_calib` partagé, `from_checkpoint` tolère `strict=False`

### Scripts
- `scripts/train_position_head.py` : entraîne position_head (WM gelé) — sonde, pas production
- `scripts/train_slot_token.py` : entraîne token_score JEPA (plafond 1,50 m — négatif informatif)
- `scripts/train_slot_affinity.py` : entraîne affinity_head BCE (échec 2,10 m — négatif informatif)
- `scripts/train_value_foret.py` : entraîne value_head sur `ate` ou `ΔE`, supporte `--dream`

### Diagnostics
- `diagnostics/diag_slot_learned_affinity.py` : test Option A (MLP 4→1 par rayon) — négatif informatif
- `diagnostics/diag_latent_carries_position.py` : sonde linéaire + MLP latent→position (0,55 m GO)
- `diagnostics/diag_slot_decision_impact.py` : impact du slot sur la décision planner

### Checkpoints produits
- `data/checkpoints/wm_foret_attn_hue_pos/wm_best.pt` : WM + position_head (0,55 m, **CONTAMINÉ**)
- `data/checkpoints/value_head_foret/value_best.pt` : teacher-forced, AUC 0,78
- `data/checkpoints/value_head_foret_dream/value_best.pt` : dream-trained, AUC 0,78 (ne généralise pas)

### Code à nettoyer plus tard (dette)
- `slot_head.py` : `affinity_net`, branche `if affinity is not None` — Option A et B mortes
- `encoders.py` : `affinity_head`, `affinity_attn`, `get_affinity()` — non utilisées en production
- `command_wm.py` : `slot_learned_affinity`, `slot_token_features` — flags morts
- `command_planner.py` : ligne 573 modifiée pour `with_position_head` (à retirer)
- `serve_planner_command.py` : 3 gardes `hasattr(wm, "slot_encoder")` ajoutés

### Plan mémoire sauvegardé
- `memory/sylvan-position-head-leak.md` : leçon apprise, ne pas utiliser food_rel0 comme cible directe
