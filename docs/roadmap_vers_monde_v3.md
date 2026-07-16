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

## Les chantiers, dans l'ordre

### 1. Perception par la CONSÉQUENCE (prêt — prompt écrit)
« Dangereux = ce qui a précédé mes dégâts » : saillance apprise sur rétine brute, remplace la
règle « danger = vert » + dé-contamine les entrées des têtes (douleur̂, P̂mort). Puis volet 2
(licence séparée) : « nourrissant = ce qui a soulagé le drive » — les requêtes-couleur des slots
WM remplacées par le lien consommation appris. **Sortie : plus aucune variable clé-apparence dans
la boucle décisionnelle.** Prérequis de TOUT enrichissement du monde (sinon on débogue monde et
perception en même temps). Juge : parité avec la réf vivante (≥40 repas ET ≤10 morts poolés).

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
