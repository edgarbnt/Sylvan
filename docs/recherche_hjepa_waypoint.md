# Recherche — un « petit H-JEPA » (waypoints à 2 niveaux) : ce que dit la littérature, croisé avec notre cas

> Session de RECHERCHE du 2026-07-16, à la suite du verdict d'élimination (commit `8a3d80a` : la fente
> MPC est le verrou, pas le juge). Question de l'owner : « ce qu'il faudrait, c'est le H-JEPA — est-ce
> raisonnable ? peut-être un petit H-JEPA pour ce problème ? et une autre solution, est-ce pur LeCun ? »
> 4 axes sondés : LeCun v0.9.2 (§4.6-4.8), hiérarchies de world-models qui marchent, classiques HRL,
> robotique waypoint. Sources vérifiées, incertitudes flaggées.

## Réponses courtes aux 3 questions de l'owner

1. **Le H-JEPA complet est-il raisonnable maintenant ? NON.** LeCun lui-même : *« Hierarchical planning
   is a largely unsolved problem »* (§7.1) ; « si un H-JEPA peut être construit et entraîné » est listé
   comme question ouverte (§8.1). C'est la frontière de recherche, pas un chantier.
2. **Un petit H-JEPA (2 niveaux, waypoints) ? OUI — licencié par le blueprint et massivement précédenté.**
   Détail ci-dessous.
3. **Une autre solution (2-segments, etc.) serait-elle « pure LeCun » ? Oui aussi** — le papier ne prescrit
   pas la procédure de recherche de l'acteur (*« does not prescribe a particular way »*, §8.1). Mais sa
   réponse aux horizons longs, c'est la hiérarchie (§4.6-4.7) ; le reste est rustine.

---

## 1. Ce que LeCun prescrit — et ce qu'il laisse ouvert (v0.9.2, numérotation réelle : H-JEPA=§4.6, planif hiérarchique=§4.7)

**La structure.** Deux niveaux couplés : le haut prédit à long terme dans une représentation grossière, le
bas à court terme. Les « actions » du haut *« ne sont pas des actions réelles mais des **cibles pour les
états prédits du niveau bas** »* (§4.7). La satisfaction d'un sous-but est mesurée par un **module de coût
par niveau** `C(s, a2)` qui *« mesure à quel point l'état satisfait la condition »*. Le bas *« infère alors
une séquence d'actions qui minimise les coûts de sous-buts »* (légende fig. 16). Le papier bénit
explicitement la lecture « état-cible » : *l'analogie du servomécanisme proportionnel à état-cible* (§4.7).

**Ce qui est OUVERT (silences explicites du papier)** : comment apprendre la hiérarchie (échelles de temps,
abstraction) ; comment entraîner `C(s, a2)` ; la terminaison des sous-buts (pas de « jusqu'à atteint », pas
de timeout) ; et surtout **la décomposition en sous-buts elle-même** : *« comment le configurateur apprend à
décomposer une tâche en sous-buts […] je laisse cette question ouverte »* (§6) ; *« precisely how to do that
is not specified »* (§8.1).

**Un niveau haut codé-main est-il conforme ?** Rien ne l'interdit : (a) le blueprint contient déjà un module
câblé par conception (l'IC immuable, §3.2) ; (b) le proposeur de sous-buts remplit un trou explicitement
non spécifié ; (c) le papier cite **sans critique** des travaux à sous-buts en « paramètres de pose »
(Gehring 2021, §7.1) ≈ nos waypoints spatiaux ; (d) *« un latent discret peut représenter plusieurs routes
alternatives »* (§4.6) + recherche dirigée/élagage (§4.8) → **proposer une poignée de waypoints discrets et
les scorer est dans la lettre du texte.** La seule friction est une aspiration (*« ces représentations
intermédiaires devraient aussi être apprises »*, §4.7) → notre niveau haut analytique = **échafaudage déclaré,
à remplacer par l'appris** — pas une violation.

**Notre planner actuel est mot pour mot le module §3.1.2 du papier** (« essentially […] MPC with receding
horizon ») → il devient le niveau bas **inchangé**.

## 2. Ce qui marche dans la littérature (design éprouvé, chiffres vérifiés)

| système | espace de sous-buts | cadence haut | contrat « atteint » | garde d'atteignabilité | preuve hiérarchie > plat |
|---|---|---|---|---|---|
| **Director** (2206.04114) | codes discrets 8×8 d'un auto-encodeur de buts (latents bruts ÉCHOUENT en récompense éparse) | K=8 fixe | max-cosinus (L2 bien pire) | codebook ≈ états VISITÉS + retour du manager évalué à travers le worker imaginé | résout Ant Maze pixels où Dreamer plat plafonne |
| **HIRO** (1805.08296) | **offsets spatiaux bruts** (navigation !) | c=10 | −‖s+g−s'‖ ; seuil+timeout | relabeling (bas-niveau APPRENANT — disparaît si gelé) | Ant Maze 0.99 vs ~0 |
| **HAC** (1712.00948) | états | ≤H=5/niveau | sparse + HER | **test de sous-but : raté ⇒ récompense −H** | 3>2>1 niveaux |
| **HIQL** (2307.11949) | représentations 10-d | **k=25-50** (ablaté 1-100 ; petit k DÉGRADE) | AWR | sous-buts régressés des données | précision au but ≥50 pas : bat plat ET hiérarchie-sans-abstraction-temporelle |
| **Puppeteer** (2405.18418) | commandes du tracker | k réglable | tracking | **bas-niveau pré-entraîné puis GELÉ**, réutilisé sur 8 tâches | humanoïde visuel 56-DoF |
| **SoRB** (1906.05253) | ~100-1000 obs du replay | 1 waypoint (Dijkstra) | — | nœuds = états visités ; **MaxDist sensible** | LA démo : même politique, échoue loin en plat, réussit avec waypoints, **zéro retrain du bas** |
| **SGM** (2003.06417) | graphe éparsifié | — | — | **nettoyage des arêtes qui échouent à l'exécution** | murs fins : SoRB 28% → SGM 100% ; « coarse+vérifié > dense+optimiste » |
| **PRM-RL** (1710.03937) | waypoints échantillonnés main, 0.1-0.4/m² | arête de graphe | — | arête ssi ≥85% de succès sur 20 rollouts | navigation 100 m+ |
| **HWM** (2604.03208, **juin 2026, co-signé LeCun** ; pas encore peer-reviewed) | latents prédits du WM long-horizon | macro-actions CEM 2 échelles | matching latent | sous-buts = prédictions du WM (sur-variété par construction) | **Franka 0%→70% ; Push-T 17%→61% ; 3× moins de compute que plat** — motivation verbatim la nôtre |

**Constantes trans-systèmes** : (i) **grossier gagne** — cadences 8-50 pas, 3-11 sous-buts, plus fin NUIT
(HIQL, Hieros sur jeux simples, Director) ; (ii) **bas-niveau gelé = précédent solide** (Puppeteer, SoRB,
PRM-RL) et **tue la non-stationnarité** qui fait souffrir HIRO/HAC ; (iii) sous-buts **spatiaux bruts
marchent en navigation** (HIRO, PRM-RL) ; (iv) trois gardes copiables : sous-buts sur-variété (Director/
SoRB ; contre-exemple LEAP : espace non contraint + scoreur appris = **sous-buts adversariaux**, le planner
exploite la valeur), vérification empirique (HAC −H ; PRM-RL 85%/20 ; SGM cleanup), et cadence grossière.

## 3. Les deux résultats théoriques qui nous concernent directement

**HIQL = la version publiée de notre pari écart-d'action.** Claim exact (§4.2-4.3) : *« la politique haute
reçoit un signal plus fiable parce que **des sous-buts différents mènent à des valeurs plus dissemblables que
des actions primitives** »* — mesuré (fig. 8, Procgen Maze) : le bruit de la valeur croît avec la distance
au but ; à ≥50 pas, la hiérarchie AVEC abstraction temporelle garde la meilleure précision, et la hiérarchie
SANS abstraction temporelle n'aide PAS. → **c'est le mécanisme isolé : l'étage waypoint est l'endroit où
notre critique (mort sur des écarts de 1e-5) peut enfin discriminer.** (Chaîne « cadence grossière ⇒ écart
d'action plus grand ⇒ argmax tolérant à l'erreur (Farahmand) » = assemblage de 3 résultats, pas un théorème
unique — flaggé.)

**Contre-point à absorber : Nachum 2019** (« Why Does Hierarchy (Sometimes) Work So Well? », 1909.10618) —
en RL model-free, le gros du bénéfice de la hiérarchie est **l'exploration**, pas l'entraînement hiérarchique.
Pour nous c'est une BONNE nouvelle double : notre problème est l'axe crédit/SNR (celui de HIQL), ET l'étage
waypoint donnera par surcroît l'exploration structurée qui nous manquait — **notre échec d'exploration en
espace-commande (2026-07-15 : le bruit COMPRIME les vies) est exactement la leçon de Director : le bonus
d'exploration au niveau WORKER nuit (« jambes chaotiques »), il FAUT le mettre au niveau MANAGER.** Deux de
nos négatifs de la semaine sont des résultats connus, retrouvés indépendamment.

## 4. La robotique classique : nos 6 échecs sont canoniques, et le fix est un MODE

- **Champs de potentiel** (Khatib 1986 ; Koren & Borenstein 1991) : minima locaux, oscillations, GNRON —
  notre journée d'élimination est la reproduction exacte de ces classes d'échec. *Aucun blend scalaire ne
  répare un minimum local.*
- **TangentBug** (Kamon, Rimon & Rivlin, IJRR 1998) — capteur de distance limité (≈ notre rétine 36 rayons) :
  la solution est **deux MODES discrets** — « motion-to-goal » et « boundary-following » — avec mémoire de
  la distance minimale au but et condition de sortie. **L'échappée est un changement de mode, pas un champ.**
  Notre étage waypoint = ce mode, en version sous-but.
- Engagement : les systèmes pratiques **commitent** le waypoint (seuil d'atteinte + hystérésis) — un local
  planner qui re-choisit à chaque pas dithered. (Cohérent avec notre leçon commitment de 2026-07-04.)

## 5. LE CROISEMENT — chaque fait mesuré chez nous ↔ la littérature

| fait Sylvan (mesuré) | écho littérature | conséquence design |
|---|---|---|
| beeline prouvé, planner §3.1.2 | SoRB : « même politique, échoue loin, réussit avec waypoints, zéro retrain » | **niveau bas GELÉ, inchangé** |
| écart d'action 1e-5 → critique mort | HIQL : sous-buts dissemblables ⇒ signal fiable | **le critique renaît à l'étage waypoint** |
| 6 juges scalaires échouent (troc évitement↔repas) | Koren-Borenstein : minima locaux inhérents ; TangentBug : fix = mode discret | **étage discret, pas un 7ᵉ blend** |
| exploration commande = compression des vies | Director : bonus worker nuit, manager requis ; Nachum : hiérarchie ≈ exploration | **explorer = varier les WAYPOINTS** (corpus contrasté gratuit) |
| slot = point le plus proche, ±1 m | LEAP : espace de sous-buts non contraint = adversarial | waypoints = **positions au sol proposées par NOUS** (pas reconstruites du danger) ; le score les évalue par la rétine (ligne dégagée) |
| bas-niveau gelé | Puppeteer/PRM-RL | pas de non-stationnarité → pas besoin de relabeling HIRO |
| monde 2-8 m, vitesse ~0.5 m/s, replan 10 pas | cadences K=8-50 partout | waypoint tenu ~100-200 pas Godot (seuil atteinte ~0.8-1 m + timeout) |

## 6. Le design « petit H-JEPA v0 » pour Sylvan (spec du chantier)

**Étage haut (échafaudage DÉCLARÉ, serveur)** — à chaque décision de sous-but (au spawn, à l'atteinte, au
timeout, ou si la cible change) :
1. Candidats : la cible directe + un anneau de **6-8 waypoints** autour de l'entité (R ≈ 2-3 m) — positions
   au sol, espace spatial brut (HIRO/PRM-RL).
2. Score de chaque candidat (tout existe déjà) : ligne entité→waypoint **dégagée de vert-proche** (rétine
   brute, style mur-vert) + ligne waypoint→cible dégagée + longueur totale du trajet en pas → queue de
   survie analytique. Le danger n'est PAS reconstruit (pas de centre/rayon) — seulement « cette direction
   est-elle verte-proche ? ».
3. **Commit** : le waypoint choisi devient la CIBLE du niveau bas (mécanisme override existant) jusqu'à
   atteinte (seuil ~0.9 m) ou timeout (~150-200 pas) ; puis re-décision. Hystérésis contre le dithering.
**Étage bas : le planner actuel, INTOUCHÉ** (coût survie, slots, tout).
**Gates pré-enregistrés** (les mêmes que la journée d'élimination — on ne déplace pas les poteaux) :
- **G1 (monde-danger)** : repas > 10 ET morts-danger ≤ 2 (réf sans danger : 15 repas ; meilleur juge plat : 9 repas / 6 morts).
- **G0 (non-régression monde plat)** : sans danger, forage ≥ baseline (le waypoint « direct » doit gagner trivialement).
**Étape apprise ensuite (le but)** : le **critique note les waypoints** (peu d'options, très dissemblables →
gros écarts — HIQL) ; corpus contrasté GRATUIT en variant les waypoints (l'exploration au bon étage) ;
gate résidu +0.10 R² déjà écrit ; à terme, remplacer le proposeur/scoreur main par l'appris (l'aspiration
§4.7), puis un vrai prédicteur grossier (H-JEPA réel).

## 7. Verdict

- **Full H-JEPA : non** (ouvert même pour LeCun).
- **Petit H-JEPA waypoint : oui** — licencié par le blueprint (sous-buts = états-cibles ; proposeur = trou
  laissé ouvert ; routes discrètes scorées = §4.6+§4.8), précédenté par une décennie de systèmes qui
  marchent (SoRB/PRM-RL/Director/HIQL/Puppeteer/HWM), aligné avec la robotique canonique (TangentBug), et
  il répare **les deux murs mesurés de la semaine** (détour inexprimable + écart d'action trop fin) au même
  étage. Le niveau haut naît analytique (échafaudage déclaré) et a un chemin d'apprentissage clair.

## Incertitudes flaggées

HWM très récent (possiblement non peer-reviewed) ; chaîne écart-d'action = assemblage, pas théorème ;
constantes Mann & Mannor non extraites ; k par-env de HIQL non extrait ; THICK au niveau abstract seulement.
Et le risque propre à NOUS : le dilemme danger-garde-la-bouffe reste dur même avec waypoints (le waypoint
contournant coûte du métabolisme dans un monde marginal) — G1 tranchera, pas l'enthousiasme.
