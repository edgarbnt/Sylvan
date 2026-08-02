# ÉTAT DES LIEUX — 2026-08-02

**Méthode** : tout ce qui suit est LU sur disque ou MESURÉ sur corpus. Rien n'est repris de la doc
sans vérification — parce que la doc s'est révélée en retard d'une génération sur le code.

---

## 1. CE QUI EST VRAI AUJOURD'HUI (mesuré)

### 1.1 Le WM forêt est BON — et c'est un acquis solide

| mesure | valeur | verdict |
|---|---|---|
| A1 — le type survit-il à l'encodeur ? | **99,9 %** (hasard 44 %) | ✅ verrou levé |
| open-loop position @ h=50 | **0,051 m** | ✅ excellent |
| architecture | attention par rayon (`RetinaAttentionEncoder`) | ✅ |

C'est l'aboutissement du diagnostic du 2026-07-24 : l'encodeur DENSE détruisait l'apparence
(82,9 % dans la rétine → 29,5 % à la sortie) parce que la couleur de la nourriture était CONSTANTE
pendant l'entraînement. Le monde forêt sert 4 teintes variables + l'encodeur d'attention + la
pression `--w-hue` → **l'information d'apparence traverse maintenant la perception**.

**Lignée** : `wm_foret_v1` (dense) → `wm_foret_attn` → `wm_foret_attn_hue` (A1 levé) →
**`wm_foret_v2`** (retrain, courant) → `wm_foret_v2_slot` / `_typed` (canaux-slot greffés).

### 1.2 Le forageur NE MARCHE PAS — et c'est le fait central

Mesuré sur les 225 vies planner du corpus forêt (`foret_v1{,b,c}_planner`) :

| | valeur | budget |
|---|---|---|
| survie médiane | **370–387 ticks** | 3000 |
| repas par vie | **1,3–1,9** | ~8 nécessaires |
| cause de mort | **85 % faim ou soif** | — |
| vies pleines | 6–10 sur 75 | — |

L'entité meurt à **12 % de son budget**, de faim ou de soif, après un seul repas.

### 1.3 La cause est la PERCEPTION — et c'est prouvé par un contrôle

Même monde, même corps, même planner. **Seule la perception change** :

| perception | erreur position | survie méd | repas/vie |
|---|---|---|---|
| slot cosinus (servi) | **2,17 m** | **380** | **1,3** |
| position_head (sonde) | **0,55 m** | **1833** | **10,5** |

**×4,8 sur la survie, ×8 sur les repas, en ne changeant que la lecture de position.**
C'est la mesure la plus importante du projet en ce moment : elle localise le goulot sans ambiguïté.

### 1.4 Pourquoi le slot échoue en forêt

Le slot lit la **rétine brute** avec des requêtes-couleur **codées-main**
(`slot_head.py:90` → `[[1,0,0],[0,0,1],[0,1,0]]`, seuil `query_thr`).

Trois faits qui se cumulent :
1. **Les requêtes sont GREFFÉES d'un autre monde.** Tous les checkpoints `_slot` portent
   `slot_note: "canal greffé depuis src ; requêtes NON re-dérivées"` — la source est
   `wm_objcentric_kin_typed`, monde différent, corps différent.
2. **Les re-dériver ne change rien.** `wm_foret_v2_typed` a des requêtes
   « apprises de la conséquence » dont le `queries_cos_to_hand = [0.9982, 0.9994, 1.0]` :
   elles ont **convergé exactement sur les primaires codées-main**. Et ce checkpoint porte
   `gates_failed: ["G-cluster", "G-slot"]` dans sa propre méta.
3. **Le monde forêt rend le critère cosinus insatisfiable.** 4 teintes de nourriture pour
   1 requête, et les troncs (`forest_appearance_var=0.15`) chevauchent la nourriture à **39,9 %**
   dans l'espace (depth, R, G, B). Aucun classifieur par-rayon ne peut les séparer — mesuré,
   3 variantes réfutées (MLP local 2,09 m, token-BCE 2,10 m, token-JEPA 1,50 m).

### 1.5 L'information EST dans le latent — mais pas linéairement

| sonde latent → position | erreur médiane |
|---|---|
| LINÉAIRE (128→2) | **2,01 m** |
| MLP (128→64→32→2) | **0,58 m** |
| écart | **×3,5** |

La littérature (LeCun/LeWM 2026, Zhang ICLR 2026) dit que dans un latent bien régularisé
l'écart linéaire/MLP est quasi nul — la position est une **direction linéaire**. Notre écart de
3,5× signale un latent **non factorisé** : VICReg empêche l'effondrement mais ne force pas la
structure gaussienne qui garantit l'identifiabilité linéaire.

---

## 2. LA CHAÎNE CAUSALE, RÉVISÉE

L'audit du 2026-07-24 posait cette chaîne :

```
A1 latent effondré → A2 slot codé-main → A3 -min_dist optimal → le critique appris échoue (×5)
```

**A1 est maintenant LEVÉ** (99,9 %). La chaîne s'est déplacée :

```
A1 ✅ RÉSOLU — le latent porte le type (99,9 %) et la position (0,58 m, non-linéaire)
      ↓
A2 ❌ INCHANGÉ — le slot lit la RÉTINE BRUTE avec des requêtes codées-main → 2,17 m en forêt
      ↓
   le planner vise des fantômes → 1 repas/vie → mort à 12 % du budget
      ↓
   le corpus ne contient presque aucun motif « approche → mange »
      ↓
   aucun module appris en aval n'a de signal (critique, valeur, arbitrage)
```

**Le WM a deux chemins de perception parallèles, et le planner lit le mauvais :**

```
rétine ─┬─▶ encodeur attention ──▶ LATENT   (type 99,9 %, position 0,58 m)  ← APPRIS ✅
        │                                                                     (non lu)
        └─▶ slot cosinus codé-main ──▶ position 2,17 m                       ← CODÉ-MAIN ❌
                                            │                                  (lu par le planner)
                                            ▼
                                        la décision
```

---

## 3. CE QUI EST CASSÉ DANS L'OUTILLAGE (et qui nous a coûté cher)

### 3.1 L'arsenal de diagnostic est AVEUGLE au WM actuel

**40+ diagnostics construisent `CommandWorldModel(...)` à la main** sans passer `retina_attention`.
Ils ne peuvent donc **pas charger** le WM courant. Seuls 13 fichiers utilisent `from_checkpoint`,
et l'essentiel date de la dernière session.

🚨 **`diag_info_matrix.py` en fait partie** — c'est-à-dire l'outil phare du projet, celui qui répond
à « où l'information meurt-elle dans le pipeline ? », la question centrale d'une archi JEPA.
Il force en plus `with_slot=True` (ligne 64), que le WM de base n'a pas.

### 3.2 La carte d'architecture ment

`tools/archi_hud/architecture.json` — que CLAUDE.md désigne comme « source de vérité » :
- `world_model.code` → `wm_objcentric_s1` : **ancien monde** (obs 277 vs 278 servis)
- `monde_foret.etat_detail` → « RIEN n'est encore collecté ni ré-entraîné » : **faux**
  (450 vies, 260 k ticks, 10 WM entraînés)
- `perception_slot.etat` → `pur` : **réfuté** par l'audit A2 (requêtes RGB codées, scoreur appris
  calculé puis jeté, readout 100 % géométrique)

### 3.3 CLAUDE.md est en retard d'une génération

| CLAUDE.md dit | le code sert |
|---|---|
| obs 277 / proprio 132 | **278 / 133** |
| rétine 360° | **cône 120°** |
| `kin_speed=0.8` | **6.0** |
| WM vivant `wm_objcentric_s2` | harnais forêt → `wm_foret_v2*` |

### 3.4 Le corpus est massivement sous-exploité

**260 497 ticks / 450 vies** disponibles en corpus forêt. Les 4 têtes entraînées le 2026-07-30
l'ont été sur `gate_foret_cl` = **9 282 ticks / 12 vies**. C'est 3,5 % du corpus — et c'est
la raison directe pour laquelle la value_head n'avait que 23 repas de signal.

### 3.5 Deux oracles vivants restent, sous-signalés

- `serve_planner_command.py:541` — le radar de **vérité-terrain** conditionne l'existence même
  d'un replan, quelle que soit la perception utilisée derrière.
- `command_planner.py:573` — la branche étiquetée « object-centric PUR » accepte
  `with_position_head`, tête **L2-supervisée sur `food_rel0`**. Aucune barrière n'empêche un
  checkpoint contaminé d'emprunter ce chemin.

---

## 4. OÙ EN EST L'ARCHITECTURE LeCun

| brique LeCun | Sylvan | état |
|---|---|---|
| Perception | encodeur attention | ✅ **apprise** (99,9 %) |
| ↳ readout de position | slot cosinus | ❌ **codé-main** — le goulot |
| World model | RSSM + rollout | ✅ fidèle (0,051 m) mais **déterministe** (pas d'incertitude) |
| Coût intrinsèque (IC) | drives faim/soif | ✅ légitimement conçu (propriété du corps) |
| Critique (TC) | `-min_dist` / coût survie analytique | ❌ **codé-main** — 5 tentatives échouées |
| Acteur | planner MPC | ✅ recherche générique pure |
| Configurateur | — | ❌ absent |
| Hiérarchie temporelle | — | ❌ absente (1 seule échelle, 0,88 m d'imagination) |

**Deux verrous seulement séparent l'archi actuelle d'une archi JEPA défendable** : le readout de
perception (A2) et le critique (TC). Et le second dépend du premier — un critique n'a rien à
apprendre tant que la décision est de la pure géométrie sur une position fausse.

---

## 5. LE POINT DE BASCULE

Le contrôle §1.3 dit que corriger la perception fait passer la survie de 380 à 1833 ticks.
Ce n'est pas seulement un gain de performance : c'est le **prérequis du critique appris** que la
doctrine du 2026-07-22 avait identifié —

> « Il faut d'abord RESSERRER LE MONDE jusqu'à ce que la valeur discrimine — que survivre
> redevienne incertain. »

Aujourd'hui la survie est à 12 % du budget (trop dur, l'entité ne fait rien d'intéressant).
Avec une perception correcte elle passerait à ~60 % — **exactement la bande où la valeur
discrimine**, ni saturée ni effondrée. La perception et le prérequis du critique se règlent
donc d'un seul geste.

---

## 6. MESURES DU 2026-08-02 (outillage réparé)

### 6.1 Le slot ne se dégrade pas : il s'effondre à PORTÉE

`diag_bilan.py`, corpus `gate_foret_cl`, vérité `food_rel0` :

| distance vraie | slot (servi) | readout latent | gain |
|---|---|---|---|
| < 2 m | 0,60 m | 0,45 m | ×1,3 |
| 2–4 m | 1,50 m | 0,48 m | ×3,1 |
| 4–6 m | 3,11 m | 0,68 m | ×4,6 |
| > 6 m | 4,10 m | **0,89 m** | ×4,6 |

Le slot voit bien de près et devient aveugle au-delà de 2 m — or la nourriture est à **3,16 m
médians**. Le readout latent, lui, reste **sous 1 m à toute distance**, donc sous le rayon de
bouche (1,0 m).

**Pourquoi** : à 3 m une baie de 0,18 m sous-tend ~3,4°, soit UN rayon de rétine (pas de 3,33°).
Le vote cosinus ray-par-ray est alors minoritaire face aux troncs qui passent le seuil. L'encodeur
d'attention, lui, pondère les 36 rayons ENSEMBLE et peut utiliser le contexte (« ce rayon rougeâtre
est isolé au milieu de verts »).

### 6.2 ⚰️ La consistance-de-transport seule est RÉFUTÉE dans ce monde

Le signal qui a fait émerger le slot en 2026-06 suppose l'objet IMMOBILE. Mesuré :

| gap | résidu de transport | = |
|---|---|---|
| 2 | 0,046 m | 0,023 × 2 |
| 8 | 0,184 m | 0,023 × 8 |
| 16 | 0,368 m | 0,023 × 16 |

Le résidu vaut EXACTEMENT `prey_speed × gap` : c'est le déplacement de la proie. Or les arbres
sont **statiques**, donc leur résidu est nul.

⇒ **L'optimum de la consistance-de-transport pure est de verrouiller sur un tronc.** Le monde a
évolué (proie mobile depuis v6) et a invalidé le mécanisme de pureté sur lequel la perception
reposait. C'est le même défaut que l'audit A4 signalait côté rêve, ici côté perception.

### 6.3 Ce qui reste comme ancre PURE : la conséquence

Inventaire des événements de conséquence, tous corpus forêt (457 vies, 271 731 ticks) :

| | nombre | densité |
|---|---|---|
| repas | **534** | 0,20 % |
| boissons | **670** | 0,25 % |
| dégâts | **0** | 0,00 % |

🚨 **Zéro dégât** : la pulsion danger n'a aucun signal vécu dans tout le corpus — un drive servi
mais jamais mordu.

**Un repas rétro-propagé rend toute la trajectoire d'approche.** En partant de « la nourriture
était à ma bouche quand j'ai mangé » et en remontant le temps par la seule ego-motion :

| k ticks avant le repas | erreur vs food_rel0 | < 1 m |
|---|---|---|
| 1 | 0,62 m | 91 % |
| 10 | 0,85 m | 65 % |
| 20 | 1,16 m | 44 % |
| 30 | 1,42 m | 36 % |

Valide jusqu'à ~20 ticks (au-delà, la dérive de la proie et l'incertitude d'ancre dominent).
Cela fournit **~3 200 positions étiquetées SANS aucune couleur** — uniquement « j'ai mangé » et
ma propre proprioception. C'est la première fois du projet qu'un signal de position pur ET dense
est disponible.

### 6.4 Ce que l'outillage réparé a coûté/rapporté

- `from_checkpoint` centralise désormais les DEUX calages du slot (seuils par-requête, angles du
  cône). Sans eux le même checkpoint rendait 1,42 m au serveur et 3,07 m aux diagnostics.
- `diag_bilan.py` : une commande, quatre sections (substrat / perception / vie / pureté).
- Le code mort des 4 approches réfutées est retiré (il fabriquait des poids aléatoires à chaque
  chargement) ; la brèche `with_position_head` dans le chemin « pur » du planner est fermée.
