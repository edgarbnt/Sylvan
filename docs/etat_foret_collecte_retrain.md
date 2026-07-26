# ÉTAT — monde-forêt : collecte, retrain, et **verrou A1 LEVÉ**

**Date** : 2026-07-25 · **Branche** : `feat/perception-consequence` · Rien n'est allé sur `main`.
Session autonome (owner absent). Tout ce qui suit est **mesuré** ; les hypothèses sont étiquetées.

---

## 1. Ce qui est SOLIDE

| brique | preuve |
|---|---|
| monde forêt complet | contrat de monde **12/12 vert** sur le corpus servi |
| déterminisme | G10 PASS ; re-confirmé par accident (deux collectes → **122 215 ticks** au tick près) |
| calibration ancrée sur le réel | budget de trajet **53,9 m/vie** (terrain 0,635 mesuré), pas les 84,9 m supposés |
| corpus | 450 vies / 122 215 ticks, **0 part dégénérée**, 3 conséquences vécues (313 consommations, 3 970 pts de dégâts sur 110 vies) |
| **WM comme modèle de DYNAMIQUE** | open-loop pos **0,020 m** (h=10) / **0,199 m** (h=50) / 0,322 m (h=80) ; yaw 0,5° ; **jalon @50 ATTEINT** |

Checkpoints : `wm_foret_v1` (base, obs 278 / proprio 133) · `wm_foret_v1_slot` (canal-slot greffé).

## 2. Ce qui ÉCHOUE — gate A1

**Le type ne survit pas à l'encodeur** : **29,1 %** de lecture depuis le latent contre **27,3 %** de
majorité — le hasard. Cible : > 70 %. C'est le gate qui justifiait tout le chantier.

Ce que la mesure a **éliminé** :
- le monde et la rétine : la teinte est **100 % séparable dans la rétine** (cos au prototype 1,0000 au p05) ;
- ❌ **l'instabilité du type** — mon premier diagnostic, **RÉFUTÉ** : la repousse est à 2500 ticks, la
  vie la plus longue fait 827 → *aucune* des 450 vies n'atteint une repousse ; le type était déjà
  stable. La re-collecte après « correctif » a rendu le même total au tick près (no-op parfait).

**Hypothèse restante (non testée à fond) : l'absence de PRESSION.** Rien dans l'objectif n'oblige
l'encodeur à garder l'apparence. Le type n'agit sur le monde qu'au contact, via le multiplicateur
nutritif : **313 événements sur 122 215 ticks = 0,26 %**. Prédire son propre latent sans apparence est
une solution parfaite du JEPA ; VICReg empêche l'effondrement sans exiger *cette* information.
C'est le manque nommé au §6ter : *« un chemin d'apprentissage entre la CONSÉQUENCE et la PERCEPTION »*.
Sonde courte (3 époques avec reconstruction rétine) : 29,1 → 29,6 % — trop court pour conclure.

## 3. Gate A2 — structurellement inapplicable en l'état

`build_typed_slots` exige **K=3** groupes de couleur (une teinte par drive). Or §2.5 impose **4
teintes de nourriture** : le clustering trouve **K=5** (3 rouges + bleu + vert). Et le danger noie la
contingence — **9 372** reliefs de dégâts contre **104** d'énergie, parce qu'un repas est instantané
quand les dégâts durent les ~110 ticks passés dans la zone — si bien que *tous* les groupes se lient
à « damage ». Rien n'a été relâché pour le faire passer.

Deux voies possibles, à trancher par l'owner : autoriser **plusieurs clusters par drive**, ou compter
les **événements** de dégât au lieu des ticks.

## 3bis. Verrou A1 — étapes 1 et 2 du chemin owner, FAITES (2026-07-25)

**Étape 1 — le risque « sonder au mauvais endroit » était RÉEL, deux fois.** La sonde lisait
`predicted_latents[:,0]`, qui n'est pas la sortie de l'encodeur mais `to_latent(GRU(encoder(obs)))`
— déjà passé par le goulot récurrent ; et elle étiquetait avec la palette **v7** au lieu de celle
servie. Corrigé : deux sites rendus, palette choisie sur le corpus (foret_v1, cos 1,0000).

| site | LINÉAIRE | MLP | majorité |
|---|---|---|---|
| **ENCODEUR** | 30,4 % | **33,3 %** | 27,3 % |
| LATENT RSSM | 27,2 % | 29,7 % | 27,3 % |

⇒ C'est la **perception** qui jette l'apparence ; le récurrent finit le travail.

**Étape 2 — l'expérience gratuite répond NON.** Tête auxiliaire décodant la teinte depuis la sortie
de l'encodeur (cible dérivée de la rétine, `--w-hue`, tête non sauvée) :

| pression | encodeur (MLP) |
|---|---|
| w_hue = 0 | 33,3 % |
| w_hue = 5 | 35,7 % |
| w_hue = 50 | **37,3 %** |

Une pression **10× plus forte n'achète que +1,6 pt**. Ça sature très loin des 70 %.

**Trois alternatives écartées par la mesure** (pas par l'argument) :
1. le gradient atteint bien l'encodeur (`total.backward()` inclut le terme, tête dans l'optimiseur) ;
2. ce n'est **pas** la SÉLECTION (mon argmin sur les rayons) : la sonde sans sélection
   « le type k est-il PRÉSENT ? » est aussi au hasard (62-64 % pour des bases 59-73 %) ;
3. ce n'est **pas** un effondrement : rang effectif **38,0/128** (encodeur), **45,3/128** (latent) —
   mieux que le 34/128 cité dans les notes du projet.

⇒ **L'encodeur a la capacité, reçoit le gradient, et n'alloue rien à l'apparence.** Le levier n'est
donc pas une tête de décodage. Restent les pistes nommées par l'owner : pression venant de la
**VALEUR/CONSÉQUENCE** (value-aware model learning, représentations reward-predictive), ou un
encodeur à **ATTENTION** — le slot y arrive, lui, parce qu'il lit la rétine par attention
géométrique explicite au lieu d'un MLP dense.

## 3ter. Piste ATTENTION testée — nécessaire, mais PAS suffisante (2026-07-25 nuit)

**À tâche isolée, l'architecture EST le mur** (aucune perte WM en concurrence ; même corpus,
découpe, largeur, graine, budget) :

| lecture de la rétine | précision type | params |
|---|---|---|
| MLP dense (copie de l'encodeur du WM) | **41,5 %** | 104 320 |
| **attention par rayon** | **99,0 %** | 92 545 |

Moins de paramètres ⇒ ce n'est pas la capacité, c'est la **structure**.

**Mais dans le WM, l'attention seule ne suffit pas.** Ré-entraîné avec `RetinaAttentionEncoder`
sous l'objectif JEPA habituel : **30,4 %** à l'encodeur — pas mieux que le dense (33,3 %).

⇒ **Conclusion corrigée** : l'architecture est **nécessaire** (le dense ne peut pas, même seul) mais
**pas suffisante** (l'attention peut, et ne le fait pas si rien ne le lui demande). Les deux causes
ne s'excluent pas.

**L'expérience manquante est précisément identifiée** — les trois autres cases sont remplies :

| | sans pression | avec pression (w_hue=50) |
|---|---|---|
| encodeur dense | 33,3 % | 37,3 % |
| encodeur attention | 30,4 % | **← JAMAIS FAIT** |

Commande, telle quelle (~35 min, aucune re-collecte) :

```
PYTHONPATH=python SYLVAN_WM_USE_RETINA=1 env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs data/replay_buffer/foret_v1{,b,c}_{planner,babble,explore} \
  --out data/checkpoints/wm_foret_attn_hue --proprio-dim 133 \
  --retina-attention --w-hue 50 --epochs 20 --seq-len 64 --lr 1e-4 \
  --w-latent 1.0 --w-proprio 0.0 --w-radar 0.0 --w-energy 20.0 --w-displacement 10.0 --w-done 1.0 \
  --latent-loss cosine --vicreg-var 1.0 --vicreg-cov 1.0 --vicreg-gamma 1.0 \
  --w-rollout 3.0 --predictor-arch shallow --mirror-augment
```
puis greffer le canal-slot et relancer `diag_latent_carries_type.py`.

## 3quater. 🎉 A1 LEVÉ — il fallait les DEUX leviers (2026-07-26)

La case manquante était la bonne, et c'est une **interaction**, pas une addition :

| encodeur | sans pression | pression w_hue=50 |
|---|---|---|
| dense (MLP) | 33,3 % | 37,3 % |
| **attention par rayon** | 30,4 % | **99,7 %** |

(linéaire 99,7 % / MLP 99,5 % ; majorité 27,3 % ; cible fixée à >70 %.)

Aucun levier seul ne bougeait l'aiguille — tous ≈ hasard. Ensemble ils saturent. Le latent RSSM
suit : **96,8 %** en linéaire, **99,5 %** en MLP (contre 29,7 % avant) — l'information traverse
désormais le goulot récurrent.

**✅ Non-régression : la dynamique ne paie rien, elle s'AMÉLIORE.**

| horizon | WM dense | WM attention+pression |
|---|---|---|
| h=10 | 0,020 m | **0,011 m** |
| h=50 | 0,199 m | **0,132 m** |
| h=80 | 0,322 m | **0,236 m** |
| yaw @50 | 0,5° | **0,2°** |

Jalon @50 **ATTEINT**. Checkpoint : `wm_foret_attn_hue`.

**⚠️ Ce que ça ne dit pas encore.** La pression vient d'une tête de décodage dont la GRANDEUR est
choisie à la main (la teinte). C'est une perception **apprise** — aucun seuil cosinus codé-main,
aucun oracle, la cible sort de la rétine — mais ce n'est pas encore une pression née de la
**conséquence vécue**, qui reste le north-star §6ter. A2 (requêtes-slot apprises) reste ouvert.

## 4. Ce que je recommande

1. **Promouvoir `wm_foret_attn_hue`** comme WM vivant du monde-forêt (A1 passé, dynamique meilleure).
   Reste à faire avant promotion : re-greffer/rebâtir le canal-slot proprement et rejouer les gates
   closed-loop (le foraging n'a pas encore été mesuré sur ce WM).
2. **Remplacer la pression main par une pression de CONSÉQUENCE** (§6ter) : la teinte est aujourd'hui
   une grandeur choisie ; le north-star est que le gradient vienne du vécu (value-aware model
   learning). L'architecture à attention est désormais en place pour l'accueillir.
3. **A2** (requêtes-slot apprises) : `build_typed_slots` exige K=3 alors que le monde sert 4 teintes
   de nourriture → autoriser plusieurs clusters par drive, ou compter les événements de dégât.

## 5. Outils ajoutés cette session

`scripts/pad_wm_gaze.py` (élargit un WM 132→133 sans décaler la rétine) ·
`scripts/graft_slot_channel.py` (greffe le canal-slot — géométrie pure lisant la rétine) ·
`scripts/collect_foret_v1.sh` (collecte MIXTE : planner 50 % / babillage 30 % / exploration 20 %) ·
`scripts/collect_foret_all.sh` · `diagnostics/diag_foret_g11_portee.py` (trajet par repas, avec ancre).
Trois outils lisaient le corpus par-épisode de travers (`critic_corpus`, `guards`, `build_typed_slots`) — corrigés.
