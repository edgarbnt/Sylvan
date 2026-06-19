# 🧬 BLUEPRINT : PROJET AGI INCARNÉE (Projet "Sylvan" - Architecture JEPA/MoE)

## 1. Résumé Exécutif (Project Manifest)

* **Objectif Principal :** Créer une Intelligence Artificielle Généralisée (AGI) de type "Vie Artificielle", incarnée dans une **morphologie quadrupède** (type chien/lézard). L'entité doit apprendre à se mouvoir, s'adapter et survivre (homéostasie) dans une forêt 3D hostile et imprévisible, en s'appuyant exclusivement sur une intelligence spatiale et physique, sans module de langage textuel.

> **🔄 PIVOT MORPHOLOGIE (2026-06-08) — humanoïde bipède → QUADRUPÈDE.** Décision du propriétaire après que l'équilibre bipède ait monopolisé tout le projet (firststeps → J0 → J_marche → survie v1-v7 : chaque objectif de survie s'effondrait car la chute à 100% ne donne aucun gradient). *« Le corps est un moyen »* au service de l'intelligence émergente, pas une fin. Le quadrupède a un centre de gravité bas + 4 appuis → il ne tombe quasiment plus → gradient stable → la locomotion s'apprend vite → on atteint enfin le vrai sujet (foraging / survie / boucle world-model). **Résultat immédiat : J0-quad (debout) validé du 1er coup ; la marche apprise from scratch en ~25 min (fwd 0.54, vs le bipède jamais sous 100% de chutes en survie).** Le saut/grimper ne seront PAS codés en dur : ils doivent **émerger du besoin de survie** (nourriture sur relief, obstacles). Le corps humanoïde est sauvegardé dans `_body_backup_humanoid_20260608/`.
* **Approche Scientifique :** Architecture JEPA (Joint Embedding Predictive Architecture) combinée à un réseau massif fractionné (Mixture of Experts). L'agent filtre le bruit visuel, apprend les lois de la physique par "l'imagination latente", et développe une réflexion organique émergente.
* **Stack Technologique Définitive :**
  * **Moteur du Monde (L'Usine à Données) :** Godot Engine 4.x + Godot Jolt (Moteur physique rigide).
  * **Cerveau Mathématique :** Python 3.10+, PyTorch.
  * **Pont de Communication :** Godot RL Agents + *Replay Buffer* asynchrone sur SSD.

---

## 2. Architecture du "Cerveau" : Le Paradigme V-M-C et MoE

L'architecture reproduit l'évolution biologique : elle traite l'information de manière sélective et route les problèmes complexes vers des sous-réseaux spécialisés pour ne pas saturer la puissance de calcul.

### [V] L'Encodeur Visuel (L'Instinct de Filtrage)
* **Technologie :** CNN ou Vision Transformer (ViT) léger.
* **Rôle :** Capte les pixels de la caméra Godot et la proprioception (l'angle des os), et écrase tout cela en un **Vecteur Latent** compressé ($z_t$).
* **Mécanisme d'Attention :** Filtre le bruit décoratif (ex: les feuilles qui bougent au vent) pour ne conserver que la topologie de survie immédiate (les pentes, les obstacles, les dangers).

### [M] Le Modèle du Monde (L'Imagination Spatiale via MoE)
* **Technologie :** RSSM (*Recurrent State Space Model*) massif structuré en **Mixture of Experts (MoE)**.
* **Architecture interne :** * Un **Routeur** analyse le vecteur latent $z_t$.
  * Il distribue l'information à un ou deux **Experts** spécialisés (ex: un expert "gravité", un expert "collision", un expert "ressource") parmi des dizaines d'experts inactifs.
* **Rôle :** Prédire mathématiquement l'état latent suivant ($\hat{z}_{t+1}$) et la récompense de survie attendue. Grâce au MoE, ce réseau possède une immense capacité de compréhension sans surcharger la carte graphique (Sparse Activation).

### [C] Le Contrôleur (Le Moteur d'Action)
* **Technologie :** Réseau Actor-Critic (type PPO).
* **Rôle :** Il ne regarde jamais directement le jeu Godot. Il pilote les moteurs virtuels des articulations en s'entraînant exclusivement à l'intérieur des "rêves" générés par le Modèle [M]. Il simule des millions de futurs possibles en quelques millisecondes.
* > ⚠️ **RÉVISION 2026-06-03 — voir §6.** Le « s'entraînant *exclusivement* dans
  > les rêves » est **amendé**. L'entraînement 100% imagination provoque le
  > *model-exploitation* (l'actor gagne dans un rêve faux et tombe dans le réel).
  > On passe à un contrôle **ancré (grounded)** : World Model JEPA sacré, mais
  > imagination **vérifiée** contre la réalité + échafaudage décroissant. Détails,
  > raisons et garde-fous en §6.

---

## 3. L'Environnement : La Boucle d'Homéostasie (Godot 4)

L'environnement impose la nécessité de survivre. L'intelligence de Sylvan n'est pas programmée, elle émerge pour répondre à ces jauges biologiques.

* **La Vulnérabilité (Énergie) :** Une jauge `energie` qui se vide avec le temps (métabolisme) et s'accélère avec l'effort physique. À 0, l'épisode se termine (mort virtuelle).
* **L'Instinct de Conservation (Santé) :** Une jauge `sante` qui baisse en cas de chute ou de contact avec un danger naturel (ronces, chutes d'arbres).
* **Les Ressources :** Des éléments interactifs (ex: sphères bleues pour l'eau, cubes rouges pour la nourriture) générés procéduralement pour restaurer les jauges.

---

## 4. Le Cycle Circadien et Stratégie Matérielle (Optimisation AMD RX 5700 XT)

L'apprentissage ne se fait pas en temps réel. Pour simuler la consolidation de la mémoire biologique et contourner les limites matérielles, Sylvan évolue selon un cycle strict d'Éveil et de Sommeil.

### Phase 1 : Le Jour (L'Éveil et la Collecte)
* **L'Action :** Godot tourne sur le PC (en mode normal ou accéléré *Headless*). Le cerveau de Sylvan est "gelé" en mode Inférence (lecture seule).
* **L'Expérience :** Sylvan interagit avec le monde. Il effectue des actions (parfois bruitées aléatoirement pour l'exploration).
* **La Mémoire à court terme :** Chaque seconde de sa journée est envoyée et stockée sur le disque dur SSD dans le *Replay Buffer*.

### Phase 2 : La Nuit (Le Sommeil et l'Apprentissage)
* **La Pause :** La simulation Godot est mise en pause.
* **Le Rêve (PyTorch sur GPU/CPU) :** Le cerveau s'allume en mode "Entraînement". PyTorch pioche massivement dans le *Replay Buffer* (souvenirs du jour + souvenirs anciens).
* **La Consolidation (MoE & Rêve Latent) :** 1. Le Modèle du Monde [M] ajuste ses synapses (experts) en calculant ses erreurs de prédiction. Les experts sont chargés en VRAM via quantification (Int8) ou appelés depuis la RAM classique.
  2. Le Contrôleur [C] s'entraîne dans l'espace latent mathématique mis à jour pour trouver de nouvelles stratégies de survie.

### Phase 3 : Le Réveil (La Mise à Jour)
* **Le Transfert :** Les nouveaux poids mathématiques du réseau sont sauvegardés.
* **La Réalité :** Godot est relancé. Sylvan se réveille avec l'expérience acquise durant la nuit, prêt à affronter son environnement avec de meilleurs réflexes.

---

## 5. Feuille de Route d'Apprentissage (Curriculum)

Le projet évolue par paliers pour éviter l'effondrement prédictif du Modèle du Monde.

* **Étape 1 : L'Agitation et la Matrice (Le Chaos)**
  * Configuration de l'environnement Godot procédural.
  * Mouvements aléatoires (Motor Babbling) pour remplir le Buffer.
  * Le Modèle [M] s'entraîne la nuit pour comprendre l'inertie du corps et la gravité.
* **Étape 2 : La Locomotion Émergente**
  * Objectif temporaire de l'Acteur [C] : "Garde le buste droit et utilise de l'énergie".
  * Sylvan apprend à se lever et à marcher dans ses rêves mathématiques. Validation de jour dans Godot.
* **Étape 3 : La Quête de Survie (ALife Pure)**
  * L'objectif d'avancement est supprimé. La seule directive devient : "Maintiens tes jauges d'énergie et de santé au-dessus de zéro".
  * Sylvan utilise l'attention de son réseau pour repérer les ressources, et son Modèle du Monde pour planifier des trajectoires sûres à travers la forêt 3D.

---

## 6. Décision architecturale centrale : *grounded*, pas « 100% imagination » (2026-06-03)

> Cette section **amende** la formulation « le contrôleur s'entraîne exclusivement
> dans les rêves » de §2.[C]. La vision reste intacte ; l'implémentation se durcit.

### La vision, réaffirmée
> Une entité qui **comprend le monde parce qu'elle l'a expérimenté** — pas parce
> qu'on le lui a expliqué. Plongée dans un environnement, elle survit (tenir,
> se mouvoir, manger, boire, homéostasie), et de cette expérience **émerge une
> intelligence**. Le tout dans l'architecture **JEPA / World Model de LeCun**.

### Les deux choses qu'on confondait
| Concept | Statut |
|---|---|
| World-model appris **par l'expérience** (JEPA, prédiction latente) | **Sacré, non-négociable** — c'est là que vit la compréhension |
| Contrôleur entraîné **uniquement** dans l'imagination (Dreamer pur) | **Rejeté** — fragile, et *pas* une exigence de LeCun |

**« Contrôle 100% imagination » n'est pas du JEPA, c'est un choix d'implémentation
de Dreamer.** Et c'est le mur frappé en pratique : l'actor « tient en équilibre »
dans le rêve (return imaginé ~53.5) et **tombe dans le réel** → *model-exploitation*
(le contrôleur exploite les erreurs du world-model pour gagner dans un rêve qui ne
transfère pas).

### Notre position : ancré (grounded) — et c'est *plus* fidèle à LeCun
- Chez LeCun, le world-model reste **ancré dans le réel en permanence** ; la
  planification l'utilise ; la policy **amortit** la planification (Mode 2
  délibéré → Mode 1 réflexe). L'imagination n'est jamais coupée de l'expérience.
- On **vérifie sans cesse que le rêve transfère au réel** (cf. §8). On ne fait
  jamais confiance à un succès imaginé non vérifié.

### L'échafaudage n'est pas de la triche
> **Métaphore directrice : une béquille décroissante = un parent qui tient le vélo.**

Un nourrisson n'émerge pas en savane ; on le tient debout, il **expérimente**
l'équilibre dans un cadre survivable, son modèle s'affine, puis on lâche. Il a tout
vécu de ses propres muscles ; on a juste **ordonné ses expériences**. L'émergence
pure sans échafaudage sur une tâche dure = zéro expérience positive = n'apprend
jamais. Ce n'est pas plus noble, c'est impossible.

### Mapping LeCun (rappel)
Perception/Encoder [V] · World Model JEPA [M] · Actor [C] · Cost/Critic ·
**Intrinsic cost = homéostasie énergie/santé (déjà dans le code)** · Configurator
(futur multi-tâches) · mémoire court-terme = état RSSM.

---

## 7. Architecture en couches : garder vs réécrire

| Couche | État | Verdict |
|---|---|---|
| **1. Physique + corps** (Godot/Jolt, articulations, proprio, actuation) | **QUADRUPÈDE** (tronc + 4 pattes × 3 DOF) — debout + marche OK | refait en 2026-06-08 (voir pivot §1) ; humanoïde sauvegardé |
| **2. Contrat de données** (replay, policy server, env) | marche | `proprio_dim=94`, `action_dim=12`, `metrics_dim=7`, `vision=12` (policy input 106). Contrat à ne pas casser EN COURS de phase ; il a changé une fois au pivot quadrupède (74/10→94/12). |
| **3. World Model** (RSSM/JEPA) | s'entraîne | **GARDER, raffiner** (+ mesurer sa précision, §8) |
| **4. Entraînement contrôleur** (récompense, imagination, gate, curriculum) | **tous les bugs étaient là** | **RÉÉCRIRE PROPRE** |
| **5. Instrumentation / validation** | le visuel mentait | **RÉÉCRIRE, au centre** |

« Recommencer à zéro » = réécrire proprement **les couches 4-5 sur ce blueprint**,
pas reconstruire le bipède. ~90% du bénéfice pour ~20% du risque.

---

## 8. Jalons vérifiables (méthode anti-échecs-silencieux)

On a perdu des jours car **plusieurs échecs silencieux s'empilaient** (gate gelé +
récompense plate + visuel menteur, chacun masquant les autres). Parade : avancer
par **jalons à pass/fail honnête**, sans passer au suivant tant que le précédent
n'est pas vérifié de façon non-ambiguë.

- **J0 — Sanity du substrat.** Un contrôleur *direct* (sur rollouts réels) fait-il
  tenir/rééquilibrer le bipède, même grossièrement ? *But :* prouver que physique +
  observation + récompense sont corrects et qu'une politique de balance EXISTE.
  *Pass :* balance nettement > hasard, debout visible (visuel honnête).
- **J1 — World Model prédictif ET qui transfère.** Mesurer l'erreur de prédiction
  du WM **et** |return_imaginé − return_réel| pour une même politique.
  *Pass :* erreur latente bornée + écart imagination/réalité sous seuil. **C'est le
  garde-fou anti-model-exploitation qu'on n'avait pas.**
- **J2 — Équilibre actif émergent (grounded).** WM validé + échafaudage décroissant
  → l'agent apprend à se tenir/rééquilibrer → on retire la béquille → il tient
  **sans aide**. *Pass :* fall% en chute, épisodes longs **sans béquille**, et le
  visuel honnête confirme du **vrai** équilibre actif (ni freeze, ni reward-hack).
- **J3 — Locomotion.** Réintroduire un objectif de déplacement (height-gated) une
  fois l'équilibre solide. *(≈ Étape 2 du §5)*
- **J4+ — Survie / ALife.** Manger, boire, homéostasie ; émergence sous coûts
  intrinsèques. *(= Étape 3 du §5, le grand objectif)*

Chaque jalon : **un critère chiffré, mesuré par un outil qui ne peut pas mentir.**

---

## 9. Instrumentation honnête (obligatoire)

La leçon la plus chère. En permanence :
1. **Voyant « le gradient coule-t-il ? »** — détecter automatiquement un actor figé
   (suivre le *mouvement* des pertes, pas la valeur absolue). Aujourd'hui c'est
   l'humain qui a vu l'actor loss bloquée à -53.58 ; ça doit être un voyant rouge.
2. **Validation = régime d'entraînement à l'identique.** Toute différence
   béquille/curriculum/config entre training et visuel est un bug. *(Le visuel
   appliquait une béquille fantôme reflex≈0.68 absente du training — corrigé.)*
3. **Mesure continue du transfert imagination → réalité.** Sans ça, le
   model-exploitation est invisible.
4. **Diagnostic comportemental** (≠ métriques agrégées) : freeze, crouch, sens de
   chute, stepping, survie. *(`diagnose_run` le fait — garder/étendre.)*
5. **Aucun cap/troncature silencieux** (gate, top-N, no-retry) : tout se logge.

---

## 10. Catalogue des anti-patterns (landmines déjà payées)

> À relire avant toute nouvelle campagne d'entraînement.

- **Gate d'acceptation qui gèle (« identical rows »).** Promotion du stable
  seulement si le score augmente *strictement*, sans tolérance ni échappatoire →
  fige tout le run au premier pic chanceux → la collecte rejoue une politique stale.
  *Corrigé* (tolérance ε + staleness-escape). **Cycles identiques → suspecter la
  promotion AVANT la récompense.**
- **Récompense plate sur l'horizon.** Termes *retardés* (uprightness, hauteur,
  tilt) constants tant que la chute (hors horizon) n'a pas commencé → actor sans
  gradient (loss figée au plafond « debout »). Il faut des signaux **anticipés**
  (leading), action-dépendants à *chaque* pas (vitesse horizontale du COM).
- **Pénalité leading *désarmée* sans déséquilibre.** Un terme de vitesse ne crée un
  gradient que si la trajectoire **imaginée** a de la vitesse. Imaginer rester
  immobile → pénalité jamais déclenchée. Injecter l'instabilité : **perturbations +
  `imagination_noise`**. Récompense / perturbations / noise sont **interdépendants**.
- **Béquille fantôme dans la validation.** `run_visual.sh` béquillait par défaut →
  « ne tombe jamais » en visuel vs « 100% chutes » en training. *Corrigé.*
- **Model-exploitation.** L'actor gagne dans un rêve faux. Parade : mesurer le
  transfert (§9.3), ancrer (§6), valider le WM (J1).
- **Émergence sans échafaudage.** Zéro béquille sur tâche dure → zéro expérience
  positive → n'apprend jamais. Échafauder puis retirer.

---

## 11. Philosophie de la récompense (drives intrinsèques)

- **Dense et action-dépendante à chaque pas**, jamais seulement terminale.
- **Leading > lagging** : signaux qui réagissent *avant* la conséquence
  (vitesse/élan du COM, COM au-dessus du polygone d'appui) > signaux qui ne bougent
  qu'une fois le mal fait (orientation, hauteur).
- **Intrinsèque** : à terme la « récompense » de survie dérive de l'homéostasie
  (énergie, santé), pas d'un bonus arbitraire. Balance et locomotion sont des
  sous-objectifs au service de « ne pas mourir ».
- **Garde-fous anti-dégénérescence** : un terme ne doit pas ouvrir un hack (p.ex.
  pénaliser la vitesse sans garder hauteur/uprightness → « s'immobiliser couché »).

---

## 12. État actuel & décisions ouvertes (2026-06-03)

**Corrigé cette session :** gate d'acceptation (tolérance + staleness-escape) ·
récompense leading (pénalité vitesse horizontale COM, `active_balance_v2`) ·
`run_visual.sh` (béquille fantôme → défaut zéro) · `run_sylvan.sh` (TTY headless).
**En cours :** run **R5** (gate réparé + récompense leading + perturbations +
`imagination_noise=0.30`, sans béquille) — premier run avec tous les fixes empilés.

**Décisions ouvertes à trancher :**
- Périmètre du rewrite : couches 4-5 seules *(recommandé)* vs from-scratch total.
- J0 avec un contrôleur model-free jetable (sanity substrat) : acceptable, ou
  puriste-JEPA jusqu'au bout ?
- Vision : `vision_shape=[0]` aujourd'hui (proprio seul) ; quand introduire la
  vraie JEPA visuelle (le [V] / MoE du §2) ?
- Forme de l'échafaudage : couple-torse décroissant (existe) vs assist gravité vs
  réduction temporaire de difficulté physique.

**Prochaine étape immédiate :** mettre **J1 en place en premier** (mesure du
transfert imagination→réalité) — le garde-fou manquant qui rend tout le reste
*débogable*. Mémoire détaillée : `memory/sylvan-stability-crutch.md`.

---

## 13. Honnêteté sur le « JEPA » : on en a les BASES, pas (encore) le moteur strict (2026-06-13)

> Acté à la demande du propriétaire après lecture comparée de `JEPAConcept.md` et du
> world-model qu'on vient de valider (Phase 4). **But : garder le cap théorique sans
> se mentir sur où on en est.**

**Ce qu'on a construit (Phase 4, `CommandWorldModel`) est un world-model à la *Dreamer*
(RSSM + têtes de reconstruction), PAS un JEPA strict au sens de LeCun.** C'est un cousin
qui partage l'architecture cognitive mais pas le moteur technique. Distinction précise :

| Critère JEPA (LeCun, cf. `JEPAConcept.md`) | Notre état (2026-06-13) |
|---|---|
| Prédire dans l'**espace de représentation abstrait** (pas l'input) | ⚠️ **Partiel** : on a bien un `predicted_next_encoded` (latent→latent), mais la perte est **dominée par la reconstruction** en espace d'entrée (proprio, radar, énergie, déplacement). C'est le chemin « génératif » que LeCun juge inférieur. |
| **Non-contrastif anti-collapse** (VICReg) | ❌ Absent. On évite le collapse « gratuitement » car les décodeurs forcent le latent à tout retenir → on **contourne** le problème dur du JEPA. |
| **Variables latentes** pour l'incertitude | ❌ RSSM **déterministe** (cf. `rssm.py`). |
| **H-JEPA** (hiérarchie multi-échelle) | ❌ Une seule échelle de temps. |
| Architecture (World Model + Cost intrinsèque + Actor + Mode-1/2) | ✅ **Bien alignée** — c'est le squelette à 6 modules (§6 mapping). |

**Pourquoi ce choix assumé :** le reconstruction-based est *ce qui a marché* cette semaine
(WM qui imagine 4 m de nav à ~13 % d'erreur, [[sylvan-rearchitecture]]). Un JEPA pur
(latent-only + VICReg + latentes stochastiques) est plus dur à stabiliser et aurait pu
coûter des semaines sans garantie. **On prend le chemin court vers le foraging d'abord.**

**Cap : se rapprocher du vrai JEPA petit à petit, par étapes incrémentales et réversibles,
APRÈS avoir prouvé le foraging émergent (Phase 5).** Ordre pressenti, chacun mesuré contre
le jalon de précision open-loop (ne rien casser) :
1. **Déplacer le poids de la perte** reconstruction → prédiction latente (réduire les têtes
   de reconstruction, muscler `encoded_predictor`).
2. **Ajouter VICReg** (variance/covariance/invariance) pour tenir les représentations sans
   décodeur → vrai anti-collapse non-contrastif.
3. **Latentes stochastiques** dans le RSSM → gestion de l'incertitude (le monde imprévisible).
4. **Hiérarchie (H-JEPA)** : un niveau lent (planif abstraite) au-dessus du niveau rapide.

Règle inchangée : **fiabilité/résultat d'abord, pureté théorique ensuite.** On documente à
chaque étape de combien on s'est rapproché — et on a le droit de s'arrêter au « assez JEPA »
si le Dreamer-like suffit au north-star.

---

## 14. Principe de GÉNÉRALITÉ : cœur agnostique, coût modulaire (2026-06-13)

> Acté après une question juste du propriétaire : « dire d'aller en ligne droite vers la bouffe,
> est-ce que ça handicape les missions futures (il n'y aura pas que la bouffe) ? » La réponse
> a clarifié OÙ doit vivre la spécificité d'une tâche pour que l'entité reste libre/générale.

**La règle d'or — où vit la connaissance d'une mission :**

| Composant | Doit être | Pourquoi |
|---|---|---|
| **World-Model** (imagination) | **AGNOSTIQUE** | Il prédit des conséquences physiques (position, énergie, radar) — il ne sait pas que « bouffe = bien ». Sert toute mission à l'identique. **Aucune connaissance de tâche ne doit y fuir.** |
| **Planificateur / Acteur** (délibération) | **AGNOSTIQUE** | Il optimise *n'importe quel* coût qu'on lui donne. « Savoir construire un trajet efficace (s'orienter puis foncer) » est un muscle GÉNÉRAL — il sert à aller vers l'eau, fuir un danger, etc. **Améliorer le planificateur (ex. M1 beeline) ne spécialise PAS vers la bouffe.** |
| **Coût** (ce qu'il veut là, maintenant) | **SPÉCIFIQUE à la tâche, mais MODULAIRE/interchangeable** | C'est le SEUL endroit où « bouffe » a le droit de vivre. Demain : ajouter des pulsions (eau, douleur, curiosité) sans toucher au cœur. |

**Conséquence pratique :** le « aller droit » (M1) vit dans le planificateur → **général, zéro handicap**. La vraie spécificité-bouffe est le terme **« distance à la bouffe »** qu'on a ajouté à la main dans le coût (proxy de densité pour l'énergie, qui est creuse). Il est **mis en quarantaine dans le coût**, jamais ailleurs.

**Réponse de LeCun à la généralisation multi-missions (les modules qui restent à construire) :**
- **Configurateur** : selon le contexte, règle *quelle pulsion domine* (faim haute → poids énergie ; danger → poids douleur). Un SEUL WM + un SEUL planificateur + PLUSIEURS pulsions coexistantes. « Aller vers X » n'est jamais codé — ça **émerge** quand la pulsion X domine.
- **Critique appris (*trainable critic*)** : remplacera à terme le proxy « distance à la bouffe » codé main par un réseau qui *apprend* à prédire le bien/mal à long terme → signal dense pour *n'importe quelle* pulsion, sans coder le proxy de chaque mission.

**Garde-fou à tenir en permanence :** avant d'ajouter quoi que ce soit, demander « est-ce que ça met de la connaissance de tâche dans le WM ou le planificateur ? » Si oui → STOP, ça doit aller dans le coût. Tant que WM + planificateur restent agnostiques, la généralité est **structurellement sauve** : passer à multi-missions = ajouter des pulsions + remplacer le proxy par un critique, **sans réécrire le cœur**.
