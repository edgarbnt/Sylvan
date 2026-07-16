# État des lieux — le critique (2026-07-15)

> Handoff court et honnête. Détail et sources : `docs/recherche_critique_argmax.md`.
> Carte vivante : `tools/archi_hud/architecture.json` (module `critique_appris`).

## Mission

Que la note des 33 plans imaginés soit, à terme, **inné (câblé, minimal) + correction apprise du vécu**
(forme LeCun `C = IC + TC`), et non plus une formule entièrement codée-main. Locomotion = prérequis donné.

## Le critique en une phrase

Le planner imagine 33 mouvements `(vx, ω)`, déroule chacun dans le world-model, **note** chacun, exécute
le meilleur. Le critique **est le donneur de notes** — rien d'autre.

## Ce qui est ÉTABLI (mesuré, pas supposé)

1. **Le critique ne peut PAS remplacer la formule** — démontré, pas supposé. Les 33 plans finissent quasi
   au même endroit (rêve 1,6 m contre ressources 2-8 m) → l'écart entre le meilleur et le 2ᵉ est **1e-5**,
   alors que l'erreur d'un réseau est **2e-4** (20-50× trop grosse). La formule y arrive car son erreur est
   **exactement zéro**. *(sonde `diag_critic_aggregation.py`)*
2. **Le cadrage « pureté = supprimer la formule » était faux.** LeCun : `note = coût inné (immuable) +
   critique (appris par-dessus)`. Le coût survie **est** un coût inné à ses yeux. Le `CLAUDE.md` PRINCIPE N°3
   le disait déjà. → la cible est `inné + correction`, pas `critique seul`.
3. **`inné + correction apprise` : bâti, et REFUSÉ par son gate.** La correction (le critique apprend
   l'*erreur* de l'inné) ne prédit la survie que **+0.023** de R² mieux que l'inné seul sur des vies jamais
   vues (gate ≥ +0.10 ; un pli sur 4 **négatif**). Code présent (`--labels residual`,
   `SYLVAN_PLANNER_COST=residual`), **NON promu**. *(`train_survival_critic.py`)*
4. **La cause n'est pas le cerveau, c'est le VÉCU.** 57 vies, **une seule politique déterministe** → l'entité
   revit la même vie (les spawns varient, la réaction non). Rien à apprendre d'un vécu qui se répète.

## État courant qui TOURNE

| donneur de notes | forage (repas+eau, 12 vies) | statut |
|---|---|---|
| **formule innée seule** (`SYLVAN_PLANNER_COST=survival`) | **34,5** | ✅ le vivant |
| valeur parfaite (oracle, sonde) | 36,5 | plafond de la fente |
| critique appris **à la place** | 16 | ☠️ prouvé impossible |
| **inné + correction** (`=residual`) | non mesuré | 🔨 bâti, gate refusé, non promu |

## Exploration en espace-commande : TESTÉE, mauvais levier (2026-07-15)

Hypothèse : l'entité revit la même vie → lui faire vivre des vies **variées** donnerait à la correction un
contraste à apprendre. Le bouton `SYLVAN_CMD_EXPLORE_STD` était un no-op (bruit tiré par replan, re-corrigé
aussitôt). **Fix implémenté** : `SYLVAN_CMD_EXPLORE_PERSIST=K` — tenir le biais K replans (K=5 ≈ 1 m de
déviation engagée). **A/B court (8 vies × 2, `diag_life_diversity.py`)** :

| | exploration OFF | persistante (std 0.3, K=5) |
|---|---|---|
| ω saturé aux bornes ±0.6 | 44 % | 22 % → le bruit **atteint** bien la commande |
| **dispersion de survie** (le juge : diversité des issues) | 1120 | **742 (×0.66)** |

**Verdict = négatif propre.** L'exploration atteint la commande (le no-op est corrigé) mais **COMPRIME** les
issues au lieu de les diversifier : tout le monde meurt un peu plus tôt, sans structure. **Le monde est
simple et la politique quasi-optimale → perturber au hasard ne peut que dégrader.** Le bruit de commande
n'est pas le bon levier. *(errance non concluante : métrique polluée par les refills intra-vie.)*

## Le vrai fork (décision d'owner)

La marge du critique est **structurellement petite ICI** : l'inné forage déjà 34,5 contre un plafond de
36,5 (et l'écart restant est surtout métabolique, `diag_metabolic_ceiling.py`). Trois voies :

1. **Accepter l'inné comme point de fonctionnement.** Honnête : dans ce monde plat sans danger, la survie ≈
   géométrie, que l'inné capture déjà → un critique appris n'a presque rien à ajouter, par construction.
2. **Exploration au niveau du PLAN** (dernier levier d'exploration) : choisir un candidat non-argmax et le
   **tenir** — l'entité fait des CHOIX différents (aller à l'eau quand la bouffe primait), pas juste un ω
   bruité. Test gratuit : re-collecter + `diag_life_diversity` (dispersion de survie ↑ ?) avant tout gate.
3. **Enrichir le MONDE** (obstacles, ressources qui s'épuisent, danger) : alors l'issue dépend de plus que
   la géométrie → le résidu devient GRAND et apprenable → le critique reprend son sens. « Enrichir le monde
   avant le cerveau. »

**Gate déjà écrit** (`train_survival_critic --labels residual`, 2 min, critère +0.10) : à rejouer dès qu'un
corpus **réellement varié** existe (voie 2 ou 3). C'est lui qui dira si `inné + correction` prend vie.

## Direction choisie (2026-07-15, owner) : ENRICHIR LE MONDE — zone nocive (danger)

Décision owner : Sylvan doit être un système où l'expérience compte → voie 3. Premier élément = **zone
nocive** (une région qui abîme la santé). Choisi car la **place est prouvable** : le coût inné n'a aucun
terme de danger → l'entité fonce dedans en aveugle, coût inévitable par construction. Pas de piège collision.

**Bâti** : `godot/scripts/world/hazard_manager.gd` (disque sur le trajet spawn→bouffe, opt-in
`SYLVAN_HAZARD_COUNT`, défaut OFF = zéro régression). Branché dans `main.gd` par 4 lignes **NON stagées**
(chantier HUD owner — hooks locaux à intégrer côté owner ; toute la logique est dans le manager, stageable).
Gate gratuit : `diagnostics/diag_hazard_gate.py` (critères pré-enregistrés : aveuglement ≥50 %, coût réel).

**Méthode anti-boucle** (ce qui rend ce chantier différent) : le cher (WM-retrain pour percevoir le danger,
puis composant qui apprend à l'éviter) est **gaté derrière la preuve gratuite que la place existe**. On ne
paie l'étape N+1 que si l'entité aveugle SOUFFRE mesurablement du danger. Le baseline aveugle (santé perdue /
morts par danger) = le chiffre que l'entité percevante+décidante devra battre ensuite.

**Gate PASSÉ et CONSÉQUENT (2026-07-15)** : `diag_hazard_gate.py`, 12 vies OFF vs ON, plusieurs niveaux de
dégâts. Résultat : la santé est du **slack** (rien ne la lit avant 0 : ni planner, ni récompense, ni corps →
vérifié) → un danger sous-létal ne change rien (la faim tue avant). À **dégâts 0.5** (défaut verrouillé), la
zone devient **létale** : traverser vide la barre (100 dégâts) → **7/12 vies aveugles TUÉES par le danger**
(vs 0 sans danger ; morts de soif 8→0). Éviter = retour au régime normal → monde survivable *si contourné*.

- **BASELINE AVEUGLE = 7/12 (58 %) tuées par un danger invisible.** C'est le chiffre à faire tomber vers 0
  par une entité qui PERÇOIT et CONTOURNE. Payoff net, non ambigu (≠ marge floue du critique).
- **Config verrouillée** : `SYLVAN_HAZARD_COUNT=1` (r=1.3, dégâts=0.5, frac=0.55). Défaut OFF = zéro régression.

**ÉTAPE 1 FAITE + VÉRIFIÉE (2026-07-15) : le danger est PERCEPTIBLE.** `hazard_manager.gd` est devenu un
`Node3D` qui pose un cylindre **violet** `(0.6,0.12,0.85)` (mesh + Area3D couche-8 `retina_color`, calqué
sur food_manager) au centre de chaque zone. Vérif gratuite (collecte hazard-ON + parse `wm.retina0`) : la
rétine renvoie exactement le violet, distinct du rouge(bouffe)/bleu(eau) ; danger VU dans **79 % des frames**.
Zéro ligne changée dans la rétine (elle encode déjà RGB). main.gd : +1 ligne `add_child(hazard_manager)` (local, non stagé).

**ÉTAPE 2 RÉSOLUE SANS RETRAIN (2026-07-15) — le slot-danger marche sur le WM GELÉ.** Sonde gratuite
`diag_hazard_slot.py` : le slot lit la rétine BRUTE (pas le latent) et localise par attention-couleur
GÉOMÉTRIQUE → un 3ᵉ slot requête-VERT localise le danger sans ré-entraîner le WM (leçon slot-1 : « le slot
était déjà là »). Résultats : (A) séparation couleur PARFAITE, 0 fuite (253 rouge→rouge, 2364 bleu→bleu,
19498 vert→vert) ; (B) positions bouffe/eau bit-identiques 2-res vs 3-res ; (C) danger localisé 100 % des
frames (saillance 0.80, 2.41 m). **Le violet était un piège** (cos 0.57 rouge / 0.81 bleu > seuil 0.55 →
aurait corrompu bouffe+eau) → corrigé en VERT (`hazard_manager.gd`, `slot_head.py` 3ᵉ requête `[0,1,0]`).

⚠️ **Caveat trouvé par la sonde (§2)** : 19498 rayons verts vs 253 rouges → le cylindre (rayon 1.3) **occulte
la bouffe** qu'il garde (confirmé au gate : à 0.5, soif satisfaite mais morts de faim → eau atteignable, bouffe
non). Réaliste mais confond « éviter » et « voir la ressource ». À l'étape suivante : **réduire le rayon**
(ou décaler la zone) pour un test propre de « éviter le danger TOUT EN forageant ».

**Occlusion : réglage géométrique RÉFUTÉ (2026-07-15, 3 négatifs convergents).** Sweep rayon+placement pour
rendre le monde « évitable ET forageable » : r=0.8/0.5 → bouffe visible (0.78/frame) MAIS entrée 9%/0% (l'aveugle
rate le petit disque → 0 mort-danger, baseline perdu) ; funnel (frac 0.8, près bouffe) → PIRE : bouffe bloquée
(10 morts de faim) et 2 morts-danger seulement. CAUSE FONDAMENTALE : un disque opaque « sur le trajet vers la
bouffe » EST « sur la ligne de vue vers la bouffe » → occulte par construction. Rayon/placement = mauvais leviers.
DÉCISION anti-boucle : **garder r=1.3** (seule config conséquente, 7/12 ; défaut, zéro code). L'occlusion y est
MINEURE pour l'aveugle (le danger tue avant ; +2 morts de faim). Son impact sur une entité qui PERÇOIT+CONTOURNE
est inconnu et se résout probablement seul (contourner = ne plus faire face au cylindre = bouffe re-visible). On
ne résout pas un problème non confirmé. Réserve si confirmé : champ de fins piliers verts (rayons passent entre).

**ÉTAPE 2b FAITE + A/B = NÉGATIF QUI TUE LE PARI « DIFFÉRER L'OCCLUSION » (2026-07-16).** WM 3-slots construit
sans retrain (`build_hazard_slot.py`, slot danger=vert idx 2), terme codé-main « évite le vert » branché
(`command_planner.py`, échafaudage `SYLVAN_HAZARD_AVOID`). A/B évite OFF vs ON (WM 3-slots, danger r=1.3) :
**les deux bras IDENTIQUES** — danger 0, faim **11** (vs 3 sans danger), soif 1, entrée **9 %**. L'évitement ON=OFF
car il n'y a rien à éviter : le gros cylindre vert **occulte la bouffe** → l'entité ne voit plus sa nourriture →
erre → meurt de FAIM (11/12) → et par accident n'entre plus dans le danger (9 %).

⭐ **RÉTROSPECTIVE : le baseline 7/12 était avec le danger INVISIBLE** (bouffe visible → l'entité fonçait dedans).
Dès que le danger devient VISIBLE (obligatoire pour le percevoir), l'occlusion domine et casse le forage.
**L'occlusion n'était pas à différer — c'est LE verrou, confirmé.** Un danger perceptible DOIT être non-occultant.

**CHAMP DE PILIERS = OCCLUSION RÉSOLUE (2026-07-16, `ddb8f5e`).** Cylindre plein → N fins piliers verts (dégâts
= disque inchangé ; visibilité = poteaux fins, rayons passent entre). Sonde slot : bouffe visible 2185 rayons
(vs 253), danger localisé 100 %, slot-danger ↔ slot-bouffe 2.05 m (distincts). Un disque plat au sol ne marcherait
pas (rétine horizontale ne voit pas le sol).

**A/B ÉVITE OFF vs ON (piliers, WM 3-slots) — JALON + DILEMME (2026-07-16) :**
| | sans danger | évite OFF | évite ON |
|---|---|---|---|
| morts danger | 0 | **8/12** | **5/12** |
| morts faim | 3 | 3 | 7 |
| entrée zone | — | 64 % | 45 % |

⭐ **PROUVÉ (jalon net) : la chaîne perception → slot → décision marche DE BOUT EN BOUT, ZÉRO retrain WM.** Baseline
conséquent rétabli (OFF 8/12) ; l'évitement RÉDUIT morts-danger (8→5) et entrée (64→45 %) → l'entité perçoit ET
évite le danger. **PAS atteint : gain de survie net** — l'évitement échange danger contre faim (3→7). Causes (non
masquées) : (1) le danger GARDE la bouffe (55 % du trajet) → éviter = ne plus manger = dilemme risque/récompense
RÉEL ; (2) horizon planner myope (~0.8 m vs danger 2-4 m) → évitement TARDIF, l'entité fuit au lieu de contourner
(même mur d'horizon court que le critique) ; (3) monde marginal → vies condamnées de toute façon (change la cause,
pas le nombre).

**⭐⭐ VERDICT D'ÉLIMINATION (2026-07-16, journée entière) : LA FENTE DU PLANNER EST LE VERROU, PAS LE JUGE.**
Six règles codées-main « oracle » testées pour obtenir « contourne le vert ET mange » (mesure = LE BUT : repas
>10 ET morts-danger ≤2 ; réf sans danger = 15 repas ; parseur survie+repas/boissons ajouté à diag_hazard_gate) :
| règle | morts-danger | repas |
|---|---|---|
| répulsion 200/500/1000 | 2-8 | 3-6 |
| rêve ×2 (h160) | 4-6 | 2-7 |
| détour ligne binaire | 5 | 3-7 |
| détour gradué (profondeur de coupe) | 6 | 9 |
| + centre corrigé (+1 m) | 6 | 4 |
| mur-vert rétine brute 300/600 | 6/3 | 3/4 |
**Aucune ne casse le troc évitement↔repas.** Le mur-vert@600 ÉVITE vraiment (dégâts médians 0, boissons 22 :
l'entité campe à l'eau) mais ne mange pas. Chaque échec a été DIAGNOSTIQUÉ (pas deviné) : minimum local de la
répulsion ; gradient détour 6-37 pas vs spreads 300-500 (trace 88 replans, SYLVAN_HAZARD_DEBUG) ; centre estimé
depuis UN point perçu = ±1 m d'erreur pour une zone de 1.3 m ; paysage de coupe PLAT sur 53 % des replans.
→ **33 arcs myopes (commande constante, ~0.8 m) rejoués goulûment ne peuvent pas COMPOSER un contournement,
quel que soit le score.** Entraîner le critique-résidu dans cette fente = garanti de rejouer l'échec du critique.

**FORK RÉSOLU (owner, 2026-07-16) : « PETIT H-JEPA » — hiérarchie waypoint à 2 niveaux.** Recherche
approfondie faite (4 axes : LeCun §4.6-4.8, Director/HIQL/SoRB/HWM, classiques HRL, robotique TangentBug) →
**`docs/recherche_hjepa_waypoint.md`** (le document de référence, avec croisement point-à-point sur nos faits
mesurés). Verdict : full H-JEPA = non (ouvert même pour LeCun) ; 2-niveaux waypoint = **licencié par le
blueprint** (sous-buts = « cibles pour les états du niveau bas », §4.7 ; proposeur de sous-buts = trou
explicitement laissé ouvert, §6/§8.1), massivement précédenté (SoRB : même politique gelée, échoue loin en
plat, réussit avec waypoints), et répare NOS DEUX murs au même étage : le détour devient un MODE discret
(TangentBug : l'échappée d'un minimum local est un changement de mode, pas un blend) ET l'écart d'action
redevient large (HIQL fig.8 : « des sous-buts différents mènent à des valeurs plus dissemblables que des
actions primitives » → le critique renaît à cet étage). Bonus rétrospectif : nos 2 négatifs de la semaine
(exploration-commande qui comprime ; blends scalaires qui calent) sont des résultats CONNUS (Director :
bonus worker nuit, manager requis ; Koren-Borenstein : minima locaux inhérents).

**CHANTIER SUIVANT (session dédiée)** : spec §6 du doc recherche + prompt de lancement
**`docs/prompt_chantier_hjepa.md`**. Gates pré-enregistrés inchangés : G1 danger repas>10 ET morts≤2
(réf 15 ; meilleur juge plat 9/6) ; G0 non-régression monde plat.
Échafaudages actifs (tous opt-in, défaut OFF, déclarés) : SYLVAN_HAZARD_AVOID / _DETOUR / _CENTER_SHIFT /
_GREENWALL / _DEBUG (trace). WM 3-slots : wm_objcentric_kin_haz. À retirer une fois l'appris validé.

## Critère de succès = le BUT

Forage (repas + boissons sur 12 vies), jamais la survie médiane (plafond épars = **métabolique**,
`diag_metabolic_ceiling.py`). Référence à battre : **34,5**. Plafond de la fente : **36,5**.
