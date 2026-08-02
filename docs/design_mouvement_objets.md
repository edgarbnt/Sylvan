# Design — MOUVEMENT DES OBJETS dans le WM (dette n°1) — pré-inscrit 2026-08-02

> Pré-inscription écrite AVANT tout diag/train (§1). Ouvre après le STOP du critique-de-rang, qui
> a montré que le rêve ne prédit pas la conséquence — et pointe ici.

## Mission
Le WM croit que les objets sont immobiles. Lui rendre le **mouvement propre des objets**, pour que
son rêve prédise ce qui va arriver au lieu d'une scène figée. C'est la **dette n°1** du projet, et
elle est déjà déclarée dans la carte.

## Le fait, mesuré
`[MESURÉ: command_wm.transport_slot]` Le slot n'est déplacé que par l'**ego-motion de l'agent**.
Le déplacement propre d'un objet n'est modélisé nulle part ⇒ dans le rêve, une proie qui fuit à
0,023 m/pas **est immobile et attend**.

`[MESURÉ: diagnostics/diag_portee_g0.py + sondes du 2026-08-02]`
- sur un horizon de 80 pas, l'hypothèse immobile accumule **~1,8 m** d'erreur, pour une bouche de 1,0 m ;
- le ralenti terminal qui en découle coûte **−64,5 points** de capture, le plus gros facteur mesuré ;
- l'échafaudage `SYLVAN_PLANNER_SPRINT` neutralise sa conséquence sans corriger la cause : **dette**.

`[MESURÉ: diag_critique_rang_g0.py, 63 triplets contrefactuels]` Aucune lecture du rêve ne corrèle
avec le résultat réel (meilleur |ρ| = 0,209 ; hasard corrigé pour 7 lectures : médiane 0,176,
p = 0,334). `[INFÉRÉ]` Un rêve qui croit la proie immobile ne peut pas savoir quelle commande
l'attrape — les deux constats convergent.

## Essayé → RÉFUTÉ (ne pas répéter)
| tentative | résultat |
|---|---|
| Estimer la vitesse par **différences finies sur le slot servi** | **0,0751 m/pas — 3,3× PIRE** que de supposer l'objet immobile (0,0230). Le bruit du slot écrase le mouvement réel et la différenciation l'amplifie. |
| La filtrer temporellement | **Impossible par construction** : le biais du slot dérive à autocorrélation 0,948, la direction de la proie à 0,936 sur 5 pas — **même échelle de temps**. C'est un problème d'IDENTIFIABILITÉ, pas de réglage. |
| Viser où la proie SERA (interception) | **+0,0 %** de capture à tous les ratios `[MESURÉ: diag_prey_interception.py]` |

⚠️ **Le plafond, lui, est énorme** : avec la position VRAIE, « répéter le dernier déplacement »
supprime **99 %** de l'erreur (0,0230 → 0,0001 m/pas), et la direction de la proie est
quasi-déterministe (autocorrélation +0,986 à 1 pas, +0,770 à 20). **Le signal existe dans le monde ;
ce qui manque, c'est un moyen de le lire à travers une perception qui dérive.**

## G0 — GATES GRATUITS (0 entraînement lourd, 0 Godot), dans cet ordre

### G0-A — le mouvement est-il lisible À TRAVERS le slot, à un décalage quelconque ?
Le bruit du slot et le mouvement de la proie dérivent à la même vitesse **à un pas**. Mais le
mouvement s'ACCUMULE linéairement alors que le bruit sature. Mesurer, pour des décalages Δ = 1…60 :

```
SNR(Δ) = médiane | déplacement VRAI de la proie sur Δ |  ÷  médiane | changement d'erreur du slot sur Δ |
```

| gate | critère |
|---|---|
| **PASS** | il existe un Δ avec **SNR ≥ 2,0** ⇒ une estimation de mouvement est extractible du slot servi à ce décalage |
| 🛑 **STOP** | SNR < 1,0 à **tous** les Δ ⇒ le slot servi ne peut pas porter le mouvement, et le chantier **REDIRIGE VERS LA PRÉCISION DE LA PERCEPTION** (−14,8 pts, déjà mesurés) |

### G0-B — l'information est-elle dans le CAPTEUR (plafond, étiquettes parfaites) ?
Indépendant de G0-A : une petite tête sur **deux rétines consécutives** peut-elle prédire le
déplacement propre de la proie ? Étiquettes = vérité-terrain, **oracle d'ÉVAL uniquement**, pour
établir un PLAFOND. Découpe **par épisode**.

| gate | critère |
|---|---|
| **PASS** | erreur angulaire médiane sur la direction ≤ **45°** (hasard = 90°) |
| 🛑 **STOP** | > 70° ⇒ l'information n'est pas dans la rétine ; aucune tête ne la trouvera |

### G0-C — existe-t-il une CIBLE auto-supervisée ? (payé seulement si G0-A et G0-B passent)
Le chantier ne vaut que s'il est **JEPA-pur** : la cible doit venir de l'observation, pas d'un
oracle. Candidate : la position future du slot au décalage Δ retenu par G0-A. Vérifier que
l'apprentissage sur cette cible bruitée retrouve ≥ 60 % de la performance obtenue avec les
étiquettes parfaites de G0-B.

## Critère de succès du chantier (le BUT)
`[MESURÉ comme référence: A/B sprint du 2026-08-02]` Le vrai juge n'est pas l'erreur de prédiction,
c'est que **l'échafaudage devienne retirable sans perte** : `SYLVAN_PLANNER_SPRINT=0` avec le WM
conscient du mouvement doit atteindre au moins la consommation par 1000 pas vécus obtenue avec
l'échafaudage (**3,60**), et la survie moyenne (**1358**).

🛑 **KILL** : consommation par temps vécu en baisse de **plus de 15 %** contre le bras échafaudé, ou
morts-danger **+3 ou plus** (magnitude, pas direction — faute de pré-inscription du matin).

## Réserve à dire d'avance
`[HYPOTHÈSE]` G0-A a une chance réelle d'échouer : le bruit positionnel du slot est du même ordre
que le déplacement de la proie, et c'est précisément ce qui a fait échouer la voie naïve. Si c'est
le cas, **le vrai chantier est la PRÉCISION DE LA PERCEPTION**, et le mouvement des objets devient
son aval. Ce serait le **quatrième** chemin indépendant à désigner la perception — après le coût de
capture (−14,8 pts), l'échec du critique-de-rang, et l'inséparabilité biais/mouvement. Je le dis
maintenant pour ne pas le « découvrir » après coup.

## Ce que ce chantier ne fait pas
Il ne corrige ni le rayon de braquage (−23,3 pts), ni l'arbitrage (clos en négatif), ni la survie
bimodale. Ce sont des sujets distincts.

---

## ⭐⭐ VERDICT G0 (2026-08-02, `diagnostics/diag_mouvement_g0.py`) : **STOP — REDIRECTION VERS LA PERCEPTION**

### G0-A ❌ — le mouvement n'est lisible à AUCUN décalage
| Δ | déplacement VRAI | changement d'ERREUR du slot | SNR |
|---|---|---|---|
| 1 | 0,023 m | 0,066 m | 0,35 |
| 10 | 0,230 m | 0,627 m | 0,37 |
| 30 | 0,690 m | 1,590 m | 0,43 |
| 60 | 1,380 m | 2,324 m | **0,59** |

L'asymétrie supposée est **réelle mais insuffisante** : le SNR monte bien avec Δ (0,35 → 0,59),
donc le mouvement accumule plus vite que le bruit — mais à peine. Le bruit croît en Δ^0,87 quand le
signal croît en Δ^1,0 : le SNR progresse en Δ^0,13, et n'atteint jamais 1,0 dans un horizon utile.
**Barre STOP pré-enregistrée (SNR < 1,0 partout) : franchie.**

### G0-B ❌ — l'information n'est pas dans le capteur non plus
Avec des étiquettes **PARFAITES**, une tête sur deux rétines + ego-motion prédit la direction de la
proie à : 85,3° (Δ=10) · 80,7° (20) · 71,8° (30) · 73,1° (45) · 68,9° (60) · 67,6° (90).
**Jamais près de la barre 45°, jamais loin du hasard 90°.** Le balayage montre que le résultat BOUGE
avec le paramètre — ce n'est donc pas un test dégénéré, c'est une absence d'information.

### Conséquence
La cause est identifiée et unique : **le bruit de la perception domine le mouvement à toutes les
échelles**. Ce n'est pas un problème de modélisation du mouvement, c'est un problème de PRÉCISION.

⇒ **Le chantier redirige vers la PRÉCISION DE LA PERCEPTION**, exactement comme la réserve du design
l'annonçait avant de lancer. C'est le **quatrième chemin indépendant** à y mener :
1. l'ablation de capture — le biais de visée coûte **−14,8 pts** ;
2. le G0 critique-de-rang — le rêve ne porte aucun signal sur le résultat (p = 0,334) ;
3. l'estimation naïve de vitesse — **3,3× pire** que de supposer l'objet immobile ;
4. ce G0 — SNR 0,59 max, et 68° avec des étiquettes parfaites.

### La cible est précise
`[MESURÉ, sondes du 2026-08-02]` L'erreur de gisement vaut **10,5°** quand un rayon touche vraiment
la cible, et **33°** quand aucun ne la touche — ce qui arrive **61 %** du temps. Et la gate de
visibilité servie est à **AUC 0,559**, quasi le hasard. **L'entité invente une position, et elle ne
sait pas qu'elle invente.** C'est là qu'est le prochain chantier.
