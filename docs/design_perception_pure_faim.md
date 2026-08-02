# Perception PURE de la nourriture — dissoudre « bouffe = rouge »

**Statut** : pré-inscrit le 2026-08-02, jamais lancé. **But** : modularité, pas performance.

---

## 1. Le but

L'entité doit reconnaître **elle-même** ce qui est bon ou mauvais dans son monde. Concrètement :

> Ajouter un fruit bleu ⇒ elle le mange ⇒ elle apprend qu'il nourrit ⇒ **zéro ligne de code touchée.**

Aujourd'hui c'est impossible : `python/sylvan/models/slot_head.py` contient

```python
color_queries = [[1,0,0], [0,0,1], [0,1,0]]   # rouge=bouffe, bleu=eau, vert=danger
aff = (cos(rayon_rgb, requête) - query_thr).clamp(min=0)
```

C'est la dernière clé-apparence structurelle du projet. **C'est elle qu'on retire, et rien d'autre.**

⚠️ **Ce n'est PAS un chantier de performance.** Le critère de promotion est celui de la doctrine
du 2026-07-22 : un composant appris légèrement moins performant que son équivalent designé est
promouvable ; ce qui est refusé, c'est un composant **activement nuisible**.

---

## 2. Ce qu'il ne faut PAS toucher

La perception a deux moitiés et **une seule est impure** :

| moitié | quoi | verdict |
|---|---|---|
| **GÉOMÉTRIE** | angles de rayon connus → soft-argmax → coordonnée | **garder** — c'est l'optique du capteur, pas de la cognition |
| **SÉLECTION** | quels rayons appartiennent à quelle pulsion | **remplacer** — c'est le `cos > seuil` codé-main |

Mesuré le 2026-08-02 : avec une sélection **parfaite**, la géométrie de production rend
**0,24 m et 5,0° de gisement** (2,6° au-delà de 2 m). Elle n'est pas le problème.

---

## 3. La recette : refaire pour la faim ce qui a MARCHÉ pour le danger

`python/scripts/train_danger_saliency.py` est le **seul** module de perception marqué `pur` dans
la carte d'archi, et il a **passé son juge 41/9**. Il a dissous la règle codée-main
« danger = vert ». Sa forme :

```
P(dégâts au tick t | rétine_t) = σ( b + max_{rayons touchants k}  s(rgb_k) · g(d_k) )
```

- **`s(rgb)`** — MLP 3→16→1, **la couleur SEULE, jamais la distance**. C'est elle qui apprend
  « à quoi ressemble ce qui compte ». Parce qu'elle ne voit que l'apparence, ce qu'elle apprend
  de près **vaut à toute distance** — c'est la propriété clé, et c'est ce qui manquait à toutes
  les approches réfutées au §5.
- **`g(d)`** — portée apprise, σ((ρ−d)/τ).
- **max-pooling** — la conséquence a UNE source. ⚠️ La forme SOMME a été testée et a ÉCHOUÉ
  (crédit partiel à la mauvaise couleur, ρ̂ figé) : négatif banké, ne pas y revenir.
- **prior de parcimonie** λ·mean(s(touchants)) — « rien ne compte sans preuve vécue » : les
  apparences jamais contraintes retombent à zéro.

C'est exactement le cadre **Multiple Instance Learning** (sac = les 36 rayons, étiquette au
niveau du sac, attribution par l'attention) — [Ilse et al., ICML 2018](https://arxiv.org/abs/1802.04712).

### Transposition à la faim

```
P(je consomme bientôt | rétine_t) = σ( b + max_k  s_faim(rgb_k) · g_faim(d_k) )
```

- **étiquette** : la consommation VÉCUE (`wm.ate`, ou le saut d'énergie). Aucune couleur.
- **déploiement** : le slot sélectionne sur `s_faim(rgb) > 0.5` au lieu de `cos > query_thr`.
  La géométrie du soft-argmax ne change pas d'une ligne.
- **modularité** : une tête par pulsion. Soif = même code, étiquette = saut de soif.
  Danger = la tête qui existe déjà.

---

## 4. L'état mesuré (2026-08-02) — le point de départ

| | valeur |
|---|---|
| slot servi — position / gisement / distance | 1,42 m / **23,1°** / **0,06 m** |
| plafond (sélection parfaite) | 0,24 m / **5,0°** |
| comportement : survie médiane | **370** ticks sur 3000 |
| comportement : repas par vie | **1,3** (le monde en exige ~8) |

**Le slot est excellent en distance et mauvais en gisement.** L'entité rate parce qu'elle vise
à côté (23° à 3,2 m = 1,24 m, pour une bouche de 1,0 m), pas parce qu'elle juge mal la distance.

### Le fait qui borne tout

| | part | gisement |
|---|---|---|
| un rayon touche vraiment la cible | **39 %** | **10,5°** |
| aucun rayon ne la touche | 61 % | 33,0° |

Taux de contact par distance : **70 % (<2 m) → 38 % (2-4 m) → 16 % (>4 m)**.

⇒ Quand l'entité voit vraiment, elle localise bien. Le reste du temps elle invente — **et elle ne
sait pas qu'elle invente** : la gate de visibilité servie est à AUC **0,559** (hasard = 0,50).

Une meilleure sélection ne peut donc pas tout régler. Deux leviers complémentaires, mesurés,
à traiter APRÈS ce chantier :
- **mémoire** : parmi les 61 % de ticks aveugles, **52,7 %** avaient une observation dans les
  40 derniers ticks. `python/sylvan/control/slot_memory.py` existe, opt-in, jamais promu.
  Ferait passer la part de ticks avec cible valide de 39 % à ~71 %.
- **savoir qu'on ne voit pas** : une tête apprise atteint AUC 0,670 contre 0,559.

---

## 5. Ce qui est RÉFUTÉ — ne pas recommencer

Tous mesurés sur `gate_foret_cl`, split par épisode, contre `food_rel0` en oracle de MESURE.

| tentative | résultat | pourquoi |
|---|---|---|
| classifieur par rayon (MLP sur depth+RGB) | 2,09 m | **39,9 %** des rayons d'arbres partagent le volume (depth,R,G,B) des rayons de nourriture |
| idem sur les tokens de l'encodeur | 2,10 m | idem |
| idem + self-attention inter-rayons | 2,10 m | idem |
| lire la position depuis le latent | plafond **1,68 m** | même avec étiquettes parfaites ET couverture complète |
| 4 requêtes serrées (une par teinte) | 23,6° vs 23,1° | la forme de la requête n'est pas le facteur |
| requêtes ré-apprises de la conséquence | `cos_to_hand` = **0,998–1,0** | elles convergent sur les primaires codées-main |
| consistance-de-transport seule | verrouille sur les **troncs** | résidu = `prey_speed × gap` exactement ; les arbres sont statiques, donc meilleurs |
| rétro-propagation des repas | 3,47 m | n'étiquette que < 2 m (15 ticks = 0,75 m de trajet) |

🚨 **Deux pièges de mesure qui ont produit de faux verdicts ce jour-là** :
1. **Split aléatoire = fuite.** À 0,05 m/tick les ticks voisins sont quasi identiques.
   Latent : 0,42 m en split aléatoire contre **2,58 m** par épisode. Facteur 6, verdict inversé.
   ⇒ **toujours découper par ÉPISODE.**
2. **Recoder le transport = juger sa propre erreur.** Balayage des 8 conventions de signe sur
   21 869 transitions : celle de production (`slot_calib`) est 1re à 0,046 m, celle que j'avais
   recodée 5e à 0,241 m. ⇒ **appeler `wm.transport_slot`, ne jamais le réécrire.**

---

## 6. Gates PRÉ-ENREGISTRÉS (posés avant de lancer)

Sur le modèle de ceux du danger (§P5 de `design_purete_hjepa.md`), **CV par épisode**.

| gate | critère | pourquoi ce seuil |
|---|---|---|
| **G-cons** | AUC(P̂(consommation), tick de consommation) > **0,75** | la tête danger visait 0,90 avec 17× plus de signal ; 0,75 est le seuil où l'attribution reste exploitable |
| **G-loc** | rappel des rayons de nourriture ≥ **0,90** et faux-flags sur les touchants non-nourriture ≤ **5 %** | la règle-couleur sert d'**oracle d'ÉVAL seulement** |
| **G-gis** | gisement médian ≤ **23,1°** (le cosinus servi) | doctrine : « ne s'effondre pas », pas « bat la baseline » |
| **G-mod** | `s_faim` re-entraînée sur un corpus où la nourriture a changé de teinte retrouve un G-loc ≥ 0,90 | **c'est LE gate du chantier** : il teste la modularité, pas la performance |
| 🛑 **KILL** | gisement > **30°** ou AUC < 0,65 | activement nuisible |

**Risque principal, à dire d'avance** : le danger disposait de **9 372** ticks de dégâts ; la faim
n'a que **534** repas sur 271 731 ticks (0,20 %). C'est **17× moins de signal**. Si G-cons échoue,
la conclusion n'est pas « l'appris ne marche pas » mais « ce monde ne rend pas assez de
conséquence pour qu'on apprenne à voir » — et c'est le MONDE qu'il faudra changer.

---

## 7. Matériel

**Corpus** (457 vies, 271 731 ticks, 534 repas, 670 boissons, **0 dégât**) :
`data/replay_buffer/foret_v1{,b,c}_{planner,babble,explore}` + `bootstrap_poshead_multi`.

⚠️ Ne PAS n'utiliser que `gate_foret_cl` (9 282 ticks, 12 vies) — c'est 3,5 % du corpus, et
c'est l'erreur de la session du 2026-07-30.

**WM** : `data/checkpoints/wm_foret_v2_slot/wm_best.pt` (gelé — on n'entraîne QUE la tête).

**Monde** : `foret_v1`, cône 120°, `eat_radius=1.0`, 4 teintes de nourriture.

**Fichiers** :
- modèle à écrire : `python/sylvan/models/drive_saliency.py` (jumeau de `DangerSaliency`)
- trainer : `python/scripts/train_drive_saliency.py` (jumeau de `train_danger_saliency.py`)
- branchement : `slot_head.py` — remplacer la branche `color_queries` par `s_drive(rgb) > 0.5`
- bilan : `diagnostics/diag_bilan.py` (déjà réparé — split par épisode, gisement/distance séparés)
