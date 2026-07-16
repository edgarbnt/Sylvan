# Roadmap — les prochains chantiers, jusqu'au monde v3 (écrit 2026-07-17)

## Mission
Ordonner les chantiers entre l'état actuel (sprint appris 45/8, pureté cartographiée, main a1e764f)
et le **monde v3** : un monde beaucoup plus complet, ressemblant au vrai (apparences variables,
topologie, ressources épuisables), où une entité doit percevoir-chercher-mémoriser-arbitrer pour
vivre. Chaque chantier est gaté ; l'ordre encode les DÉPENDANCES, pas des préférences.

## À lire d'abord
- `docs/design_purete_hjepa.md` — le critère officiel (« survit à un changement de monde ? ») et
  l'inventaire clé-apparence.
- `docs/prompt_chantier_perception_consequence.md` — le chantier n°1, prêt à lancer.
- `docs/design_critique_sprint.md` — la méthode qui marche (têtes composées, gates, juge poolé).

## La méthode (constante, non négociable)
Pré-enregistrement avant tout run ; diag gratuit avant tout train ; budget 1 train + 1 re-train
diagnostiqué ; juge closed-loop poolé (bruit ±5 repas/24 vies) ; négatif = commité ; promotion à
≥3 seeds (dette historique à résorber au passage) ; carte à jour dans le même commit.

## ⚠️ ORDRE RÉVISÉ (2026-07-17, owner) — voir `docs/design_monde_incremental.md`
L'objectif COURANT = **zéro connaissance du monde codée-main dans l'entité**. Vis-à-vis de CET
objectif, l'ordre ci-dessous est FAUX : jour/nuit (chantier 2) est orthogonal à la pureté (il
re-consolide des têtes déjà apprises, ne touche aucun reste codé-main). Le chemin DIRECT =
rouvrir la reconnaissance des types via l'ingrédient « apparences variées » (un « v3 simplifié » :
mêmes 3 types, objets variés). Il n'y a pas de « monde v3 » monolithique — une PILE d'ingrédients
découplés, chacun gaté sur son chantier. Jour/nuit reste valable mais sur sa propre piste (cycle
de vie), pas sur le chemin de la pureté. Détail + plan de réouverture + gate-capacité + sources :
`docs/design_monde_incremental.md`.

## Les chantiers, dans l'ordre (⚠️ ordre pré-révision — lire l'encart ci-dessus)

### 1. Perception par la CONSÉQUENCE (EN COURS, branche feat/perception-consequence)
« Dangereux = ce qui a précédé mes dégâts ». État au 2026-07-17 : la lunette saillance-danger
APPRISE du vécu est VALIDÉE offline (MIL max-pool sur rétine brute, labels = dégâts ; AUC 0.997,
identique à la règle verte sur 100 % des 14 003 décisions, portée ρ̂=0.63 m apprise) et les têtes
dg sont dé-contaminées à l'EXACT (0 ligne divergente). Le juge du remplacement-des-MARGES a dit
NON (29/14) : le standoff 1.0 est une PRÉFÉRENCE DU CORPS (jumeau spatial de P2-bis : survivre
exige l'aversion, pas l'espérance vécue) ; lunette innocentée. **VOLET DANGER CLOS (2026-07-17
soir) : bras MIXTE (lunette apprise + marges-standoff du corps) JUGÉ PASS 41/9** (s1 23/3 bat le
vivant 19/5 ; poolé vs gate ≥40/≤10, réf 45/8) → **PROMU** : `SYLVAN_WP_SALIENCY` +
`sprint_critic_decont` = config vivante, green_points = secours, `saillance_danger` = pur en
carte. Caveat : promotion seeds 1+2 (le seed-3 de la règle ≥3 = dette à payer). **VOLET 2 « nourrissant »
JOUÉ ET CLOS EN NÉGATIF INSTRUCTIF (2026-07-17 soir, 3 causes cartographiées §P6)** : le fit
gradient n'identifie pas une requête (jauge/init) ; la MESURE retrouve 2 apparences vraies sur 3
(eau au millième) mais food = contaminé par la géométrie engouffrée (explaining-away = forme non
licenciée) et surtout VERROU STRUCTUREL : les requêtes main sont des SÉPARATEURS idéalisés plus
écartés que les couleurs vraies (cos(bleu-vrai, vert-vrai)=0.61 > 0.55) — propriété d'appareil,
pas d'apparence. Requêtes main = déclarées-datées ; à rouvrir avec le monde v3 (apparences
variables = le contraste qui rend le chantier décidable), formes candidates précises notées.
**Sortie du build 1 RÉVISÉE : la clé-apparence DANGER est dissoute (l'étage décisionnel) ; les
requêtes-ressource restent le dernier échafaudage d'apparence, déclaré.** Le monde v3 volet
« apparences variables » devra attendre leur purification — ou la forcer.

### 2. Jour/nuit v1 — têtes seules (le plus petit cycle qui VIT)
Jour = vies avec ε au MANAGER (machinerie existante ; la collecte déterministe est prouvée
auto-confirmante, 2026-07-08) ; nuit = re-train gaté des têtes rapides (critique-sprint, P̂repas,
douleur̂, P̂mort) sur le vécu du jour. Gate par nuit : non-régression poolée vs la veille.
Prérequis : chantier 1 (ne pas consolider la lunette couleur). **Sortie : une entité qui
s'améliore en vivant — le premier cycle de vie complet.** PAS de consolidation M2→M1 ici (v2).

### 3. CHERCHER + MÉMOIRE SPATIALE (le mur de capacité)
Le fait : 43 % des replans épars n'ont rien de visible → errance (`no_food_command`). En monde
réaliste, « rien de visible » est le cas majoritaire. Deux briques liées :
(a) politique de recherche à l'étage waypoint — sous-buts d'EXPLORATION scorés par valeur
épistémique quand rien n'est visible (première pulsion non-homéostatique : curiosité) ;
(b) mémoire spatiale — zones de respawn vues, couverture (« où ai-je déjà regardé ») — le module
mémoire du blueprint, gardé exprès pour après un substrat sain (§4). Gate d'entrée : monde-sonde
ressources hors-vue au spawn, mesurer temps-à-trouver vs errance baseline (le G-place du chantier).

### 4. Jour/nuit v2 — consolidation M2→M1 (quand la pression vitesse existe)
La nuit gagne un 3ᵉ composant : BC du réflexe sur les décisions fraîches du planner (amortized
inference). Lucidités mesurées : la distillation ne dépasse pas son maître (2026-07-13) — c'est
un gain de VITESSE, pas de capacité ; et l'autre sens du pont (M1→M2, quand le réflexe rend la
main) exige un déclencheur principiel incertitude/surprise — le pont-panique a échoué (juillet).
À ouvrir quand : mondes plus grands/multi-agents, ou réflexe-danger plus rapide que la cadence
de replan.

### 5. Dettes au fil de l'eau (sur signal, jamais « au cas où »)
- multi-seed ≥3 sur toute promotion (transverse) ;
- drive-dynamics head (le suivi analytique des drives dans le planner — dette nommée du README) ;
- proposeur waypoint appris (P3 — si la topologie du monde v3 casse anneau+tangents) ;
- readout slot : reste géométrique déclaré (le fitter l'a déstabilisé — ne rouvrir que sur preuve).

### 6. MONDE v3 (un chantier de DESIGN à part entière)
Leçon monde v2 : « enrichir le monde avant le cerveau » — chaque ingrédient doit créer une PLACE
mesurable (pattern G-place : prouver par une sonde que la capacité visée gagne X avant de la
construire), pas du décor. Ingrédients candidats, chacun appuyant un chantier d'amont :
- **apparences variables/réalistes** (teste le chantier 1 : la perception-conséquence survit) ;
- **arène large + ressources hors-vue** (rend CHERCHER/mémoire décisifs, chantier 3) ;
- **ressources épuisables + zones de respawn** (rend la mémoire rentable, pas juste utile) ;
- **topologie** : murs, couloirs, impasses (teste le proposeur — P3 sur signal) ;
- **lumière jour/nuit environnementale** (module la perception, prépare des rythmes de vie) ;
- éventuellement **menace mobile** (réflexe-danger → justifie jour/nuit v2).
Règles : viabilité MÉTABOLIQUE mesurée AVANT de juger les agents (leçon plafond 1400 : portée
soutenable vs distance de spawn — sinon on juge des vies condamnées par l'arithmétique) ;
un ingrédient à la fois, chaque ajout opt-in défaut OFF avec baseline bit-identique.

## Critère de succès = le BUT (à chaque étage)
Toujours des vies mesurées (repas, morts, temps-à-trouver), poolées, contre des réfs vivantes,
au plancher de bruit connu. Jamais un proxy offline seul. Le monde v3 « réussi » = l'entité y
survit par perception+recherche+mémoire+arbitrage appris, avec un corps donné, des drives câblés,
et un WM gelé — les seules choses codées-main restantes étant des propriétés du corps, déclarées.
