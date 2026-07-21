# Prompt de démarrage — CONSTRUIRE LE MONDE (et le geler)

> Copier-coller le bloc ci-dessous comme premier message de la nouvelle session.

---

Salut. On reprend Sylvan, et on change de nature de travail : **cette session ne touche pas au
cerveau, elle construit le MONDE.**

Lis d'abord `ETAT_DES_LIEUX.md`, puis `docs/design_audit_peremption.md` (§conclusion) et
`docs/design_obstacle_affordance.md` (§G3 dégelé). Ne lance rien avant d'avoir lu.

## Pourquoi — le constat qui commande tout

La session du 2026-07-21 a réfuté **sept** pistes de capacité, chacune sur critère pré-inscrit :
correction de `nominal_speed` (−14,7 pts), suppression du plateau de la queue (lavage), critique
appris en monde plat (−0,567), en monde varié sur 80 vies (−0,106), avec token de danger (−0,210),
cône de vision (aucun hors-vue durable), prédicteur d'obstacle en forêt (**6× plus** de collisions),
et son ré-entraînement (effondrement en détecteur de proximité).

**Elles ont toutes échoué pour la même raison de fond : rien dans le monde ne les demande.**

- **mémoire** — inutile : portée de la rétine **12 m** ⩾ taille du monde **9 m**, tout est toujours vu ;
- **valeur apprise** — inutile : survivre ≈ géométrie, et l'inné est *exact* pour classer
  (écart d'action 1e-5 ≪ erreur d'un réseau) ;
- **prédiction** — inutile : le corps cinématique obéit exactement à (vx, ω), donc la trajectoire se
  calcule **analytiquement** — vérifié, `diag_candidate_divergence.py` la retrouve sans le WM ;
- **évitement** — la sélectivité « apprise » du prédicteur n'était qu'une propriété du monde
  d'entraînement (un mur cyan **unique** en monde food-only). En forêt dense, la même tête apprend
  « tout bloque de très près ».

⇒ **L'entité navigue parce que naviguer suffit.** Ce n'est pas un défaut de son cerveau.

## Ce qu'on veut

**Un monde qu'on construit une bonne fois, proprement, et qu'on GÈLE.** C'est la condition dure : le
2026-07-21, le monde a été modifié six fois en trois heures et chaque changement a invalidé les
mesures précédentes — impossible d'accumuler quoi que ce soit. Un monde figé, c'est ce qui permet à
une capacité de se prouver **et de rester prouvée**.

Quatre exigences, déduites des échecs ci-dessus — pas d'une liste de souhaits :

| pour que ceci compte | il faut |
|---|---|
| **chercher** | un monde **plus grand que la perception** |
| **se souvenir** | des ressources **qui durent et s'épuisent** |
| **prédire** | des conséquences **différées**, non calculables analytiquement |
| **éviter** | des dangers **discriminables par l'apparence**, pas seulement proches |

## Ce que je te demande de faire

**1. RECHERCHE — en workflow multi-agent. Je t'y autorise explicitement.**
Cherche sur internet ce qui fait un bon monde pour une entité ALife / foraging :
- environnements de référence en ALife et RL incarné, et ce qu'ils contiennent (topologie,
  ressources, cycles, risques) ;
- ce qui, dans la littérature, rend mémoire / exploration / planification **nécessaires** plutôt que
  décoratives ;
- écologie du fourrageur réel (patchs, épuisement, temps de repousse, théorème de la valeur
  marginale) — c'est le meilleur guide pour un monde qui exige des décisions ;
- **assets low-poly CC0** : si les KayKit actuels ne suffisent pas, trouve mieux. Le thème forêt est
  gardé. Un monde beau donne envie de le regarder, et **regarder a corrigé mes métriques deux fois**.

Fais-en une synthèse courte et opinionée, avec une recommandation — pas un catalogue.

**2. PROPOSE un design, puis attends mon accord.** Ne construis rien avant que j'aie tranché.

**3. CONSTRUIS proprement.** Pas un empilement de flags : un monde cohérent, avec des paramètres
lisibles, un défaut qui EST la config vivante, et le visuel qui correspond à ce que l'entité perçoit.

## Contraintes dures — lis-les, elles ont coûté cher

- **Ne touche ni au WM, ni au corps, ni au planner.** Cette session, c'est le monde. Le WM n'a pas
  besoin d'être ré-entraîné pour un obstacle (précédent mesuré : l'info est déjà dans le latent).
- **Le visuel ne doit pas mentir.** La rétine lit `retina_color` du corps, pas le maillage : un arbre
  au tronc brun perçu comme vert ferait joli et tromperait l'observateur.
- **Choisis les apparences par la MESURE.** Les requêtes du WM sont rouge (bouffe) et bleu (eau),
  seuil 0,55. Une apparence dont la « fuite » après seuil est > 0 sera confondue avec une ressource.
  Mesuré : vert foncé **0,0000** ✅ · gris 0,0424 · **brun foncé 0,2271** ❌ (un tronc brun est
  perceptuellement rougeâtre — le choix « naturel » est le pire).
- **Ne fabrique pas la situation que tu veux mesurer.** Placer un obstacle *entre* l'agent et sa
  cible à chaque épisode, c'est truquer le test. Concevoir la **structure du monde**, oui ; viser
  l'agent, non.
- **Une ressource ne doit jamais être emmurée** : le keep-out doit être proportionnel au rayon de
  l'occulteur. Sinon on mesure un échec du monde en croyant mesurer l'entité.
- **Réglages mesurés à conserver** : 45 arbres = fenêtre navigable (18 → aucun effet ; 54 → immobile
  85 %, 0 repas). Espacement mini 1,3 m. Échantillonnage **uniforme en aire** (`r = √u` — sinon la
  densité part en 1/r et les arbres s'empilent sur le spawn).

## Méthode — ce qui a marché aujourd'hui

- **Mesure avant de croire.** J'ai eu tort **huit fois** en une session, souvent avec de bons
  arguments. Ce qui a attrapé chaque erreur : un test gratuit, jamais un raisonnement.
- **Ne conclus jamais sur un mécanisme sans avoir tracé le chemin d'exécution.** Quatre de mes huit
  erreurs venaient d'une lecture partielle (une branche inerte, un wrapper au lieu du fichier
  délégué, une sortie au lieu d'une autre).
- **Un log doit PROUVER ce qui est servi.** Trois fois un réglage a semblé appliqué sans l'être
  (`FOOD_COUNT` en dur dans le visualiseur, `far_align` allumé partout, `SYLVAN_PLANNER_SPEED`
  overridé par personne).
- **Pré-inscris les critères, et ne déplace jamais la barre.** Y compris quand le critère se révèle
  mal spécifié : on le dit, on ne le corrige pas après coup.
- **Regarde.** L'owner a trouvé à l'œil trois bugs invisibles dans mes métriques (densité concentrée
  au spawn, aucun espacement entre arbres, trou central systématique) et a invalidé un « déficit »
  que je m'apprêtais à traiter en priorité (68 % des ratés lointains sont en fait des choix corrects).

## Ce qu'il ne faut PAS refaire

- Remplacer le coût designé par une valeur apprise (réfuté 3×, et on sait pourquoi : le signal de
  valeur est **99,6 % constant**, l'inné est exact pour classer).
- Rebrancher `obstacle_affordance/obstacle_best.pt` tel quel en monde dense (`ρ̂ = 0,63 m` → réagit au
  contact → 6× plus de collisions).
- Fonder un chantier sur le « déficit de portée » sans conditionner sur la qualité du choix.
- Chercher à fabriquer de l'occlusion avec la rétine actuelle : sans aucun arbre, il y a déjà 3309
  éclipses de durée médiane **5 ticks** — c'est du scintillement d'échantillonnage (36 rayons à 10°),
  et aucune disposition d'arbres ne l'a changé.

Commence par la **recherche**, et reviens avec une synthèse opinionée et une proposition de design.
Ne construis rien avant mon accord.
