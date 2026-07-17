# Découverte autonome apparence → conséquence — synthèse de recherche

> **Nature du document.** Synthèse d'un run de deep-research *arrêté puis récupéré*
> (`wf_0f49414b-94b`). Le digest brut (≈135 blocs de notes + pages web récupérées,
> ~546 k car., avec du bruit/boilerplate) vit à
> `~/.claude/projects/-home-edgarbrunet-Documents-PERSO-SylvanV1/49f3dc57-.../subagents/workflows/wf_0f49414b-94b/RECOVERED_digest.md`.
> Daté **2026-07-17**. **MATÉRIEL DE RÉFÉRENCE — PAS une décision engagée, pas un plan validé.**
> Aucun run, aucun code : c'est une carte de mécanismes éprouvés et CPU-friendly pour éclairer
> un futur chantier « perception→conséquence auto-découverte ».
>
> **Question.** Comment un agent (WM JEPA gelé + slots object-centric couleur-requête + planner
> MPC en `(vx, ω)`, PyTorch-CPU) peut-il découvrir, **de façon autonome et ouverte, à partir du
> seul vécu (zéro label, zéro règle codée par objet)**, les associations entre **apparences**
> (couleurs) et leurs **conséquences comportementales** — sur 5 axes.
>
> **Honnêteté (cf. CLAUDE.md §2).** Le digest est riche sur les axes 1, 2, 5 ; correct sur l'axe 4 ;
> **le plus mince sur l'axe 3** (l'idée « absorber la dynamique d'obstacle DANS le WM » n'a qu'une
> seule référence vague, tout le reste plaide pour un prédicteur d'affordance *séparé*). Plusieurs
> sources des axes 3-5 sont nommées par méthode sans métadonnées bibliographiques complètes dans le
> digest ; c'est signalé au fil du texte et dans la liste des sources.

---

## Axe 1 — Apprendre plusieurs canaux de conséquence hétérogènes à la fois

Homéostasie (food→énergie, water→soif, hazard→santé) **et** affordances physiques (obstacle bloque,
herbe traversable), le tout comme famille ouverte de prédictions apprises du vécu.

**Mécanismes les plus solides (par ordre d'affinité avec le substrat) :**

- **Horde / GVF demons** [Sutton, Modayil, Delp, Degris, Pilarski, White, Precup, *Horde: A Scalable
  Real-time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction*, AAMAS 2011].
  C'est **le** cadre canonique. Un grand nombre de « démons » indépendants apprennent chacun une
  General Value Function **off-policy** depuis un unique flux sensorimoteur partagé. Chaque démon =
  4 fonctions-question (politique π, pseudo-récompense r, pseudo-terminaison γ, terminal-reward z),
  *sans relation avec une tâche de base* → une famille **ouverte et hétérogène** de canaux (restaure-
  énergie, restaure-soif, dégât, contact-obstacle) est exactement un ensemble de démons.
  - *CPU-friendly, prouvé* : temps/mémoire **constants par pas** (coût par démon linéaire dans le nb
    de features actives) ; démontré à des **milliers de démons** sur laptop ; **1000 politiques
    apprises en temps-réel, cycle 85 ms** sur 4 cœurs. Off-policy = multiplicateur d'efficacité
    (8 démons appris en 4× le temps d'un seul).
  - *Composition* (clé pour l'axe 3) : la valeur apprise d'un démon peut servir de pseudo-reward/
    terminal-reward à un autre → on construit « près d'un obstacle » = P(pic du capteur de contact
    sous quelques secondes) purement du vécu.
  - **Prérequis / failure mode** : l'off-policy massif **exige des méthodes gradient-TD** — GTD(λ)/
    GQ(λ) convergent sous échantillonnage off-policy + approx. linéaire, le TD(λ) classique **non**.
    Le *cumulant* (pseudo-reward) est **spécifié**, c'est la *valeur* qui est apprise (nuance de
    vocabulaire à garder honnête).

- **Successor Features (SF)** [Barreto et al., *Successor Features for Transfer in RL*, NeurIPS 2017,
  arXiv:1606.05312]. Représentation `Q^π = ψ^π·w` qui **découple dynamique et récompense** :
  `r = φ(s,a,s')ᵀw`. Ajouter une pulsion = ajuster un nouveau `w` sur des SF partagées, **sans
  retrain du substrat** — l'incarnation exacte du split CLAUDE.md « WM = substrat lent / tête de
  valeur = couche rapide ». **Generalized Policy Improvement (GPI)** combine/réutilise les politiques-
  drives déjà apprises quand une nouvelle pseudo-reward apparaît.
  - **Prérequis / caveats (le digest insiste, fact-check ×3)** : (a) récompense **linéaire dans les
    features** (`r=φᵀw`) ; (b) `ψ^π` est **dépendant de la politique** (GPI a besoin d'une *librairie*
    de SF-par-politique) ; (c) SF est une **représentation de valeur, PAS un WM forward** → « WM gelé
    = SF » est une analogie de régime (dynamique fixe + récompense variable), pas mécanistique.

- **UVFA** [Schaul et al., *Universal Value Function Approximators*, ICML 2015]. Un seul réseau
  `V(s,g;θ)` factorisé en deux flux (`φ(s)`, `ψ(g)`, combinés par dot-product) généralise la valeur
  **sur les buts** → représente une famille *effectivement non bornée* de GVF dans **un** objet, coût
  d'apprentissage lié à la complexité du domaine, pas au nombre de démons.
  - **Caveat fort (bloc de réfutation dédié)** : la généralisation UVFA est **intra-famille** (les
    buts sont des *goal-states* d'un même espace) — elle **n'est PAS** la découverte ouverte de
    *nouvelles modalités* de conséquence (énergie vs soif vs santé vs traversabilité sont des
    récompenses de nature différente, hors du goal-embedding appris). Ça, c'est territoire Horde/GVF,
    pas UVFA. UVFA n'« découvre » pas, il évalue des buts qu'on lui donne ; entraînement RL instable.

- **HRA — Hybrid Reward Architecture** [van Seijen, Fatemi, Romoff et al., *Hybrid Reward Architecture
  for Reinforcement Learning*, NIPS 2017, Microsoft Maluuba]. Décompose une récompense en composantes,
  **une tête de valeur par composante**, chacune ne dépendant que d'un **sous-ensemble de features** →
  basse dimension, apprentissage facile. Précédent concret pour l'axe 4 : sur Ms. Pac-Man, HRA démarre
  à **0 tête** et en **instancie une en ligne** au premier passage/pellet → **~1800 GVF** à la fin
  (famille de têtes qui *croît* avec les apparences rencontrées). Split « **où × conséquence** » (très
  proche du substrat) : GVF de navigation `[0,1]` vers la position de l'objet **×** poids = récompense
  obtenue à la consommation ; la GVF-transport s'apprend **même sans objet présent**.
  - **Prérequis / failure mode capital** : HRA **ne découvre PAS** la décomposition — elle est fournie
    en connaissance-domaine, et chaque composante doit dépendre de peu de variables. **Donc HRA seul
    viole la contrainte « zéro label / zéro règle »** → il lui faut un **étage amont de découverte
    non-supervisée d'objets/indices** (= axes 2 et 4).

- **Object-Centric GVFs** [Nath, Subbaraj, Khetarpal, Ebrahimi Kahou, *Discovering Object-Centric
  Generalized Value Functions From Pixels*, ICML 2023, arXiv:2304.13892]. **La correspondance la plus
  serrée avec les slots.** Slot-attention découvre des features-objets ; un *Question Network* produit
  **une cumulant (pseudo-reward) par slot** ; des têtes GVF servent le contrôle — des « démons »
  auto-générés, **sans tâche auxiliaire codée-main**.
  - **Caveats** : nombre de GVF = **hyper-paramètre K fixe** (auto-généré mais **pas** ouvert/croissant)
    ; slot-attention **fusionne les objets de couleur similaire** dans un slot (pertinent : Sylvan
    requête-couleur) ; validé pixels/Atari, **pas** sur rétine bas-dim ni drives homéostatiques.

**→ Mapping Sylvan.** Le split « WM gelé = dynamique lente / une tête par pulsion sur le latent » est
littéralement l'architecture SF/HRA/Horde. Recette : garder le WM + slots couleur, poser **une tête-
GVF par canal** (chaque slot couleur porte sa propre prédiction de conséquence, façon OC-GVF), les
apprendre **off-policy en parallèle** (Horde) sur le vécu de la politique de déploiement. Le split
« où × conséquence » de HRA calque exactement `slot(position) × poids-drive-appris` — et permet
d'apprendre le transport **indépendamment** de l'événement-repas (ce que Sylvan fait déjà via slot +
readout géométrique). Le WM restant gelé, ajouter une pulsion = fitter un `w`/une tête, **jamais**
retrain du WM (aligné §3 CLAUDE.md).

---

## Axe 2 — Attribution de crédit / compétition d'indices co-occurrents

« Baie dans un buisson » : quel indice prédit le repas ? Blocage Rescorla-Wagner, contingence ΔP,
identifiabilité par décorrélation, rejet d'un distracteur neutre.

**⚠️ Résultat NÉGATIF central (le plus important de l'axe, à ne pas ignorer).** Dans un apprentissage
de contingence **incident** couleur-mot (indices prédictifs non pertinents pour la tâche de couverture,
seulement corrélés), **ni le surombrage ni le blocage n'apparaissent** (deux grandes études en ligne :
pas de coût de surombrage cue-composé vs cue-simple ; blocked ≡ blocking). L'indice redondant/bloqué
**est appris incidemment** ; la « compétition » apparente serait une **suppression descendante** par
des processus de décision explicites qui *savent déjà* que l'indice bloquant prédit l'issue. **Implication
directe pour Sylvan** : un apprenant purement associatif/incident **ne rejettera PAS spontanément** un
indice neutre co-occurrent (le buisson autour de la baie) comme le prédit R-W classique →
**l'identifiabilité demande un mécanisme explicite de décorrélation/attention/structure, pas juste la
co-occurrence.** [source « colour-word / incidental contingency », deux études en ligne ; auteur/année
non portés par le digest].

**Mécanismes proposés :**

- **Rescorla-Wagner modifié à init intermédiaire (incertitude)** [R-W classique = Rescorla & Wagner
  1972 ; variante à init non-nulle nommée par le digest, auteur non porté]. Initialiser la force
  associative d'un indice **nouveau** à une valeur **intermédiaire** encode l'incertitude sur son statut
  causal → seul ce R-W modifié capture l'**effet de redondance** (parmi des indices co-occurrents, un
  bloqué X finit noté > un non-corrélé Y ; fit R²≈0.96-0.98). **CPU-cheap, règle de mise à jour
  candidate concrète pour slot→drive.** *Caution* : R-W **non modifié**, fitté sur le jeu **complet**
  d'indices, n'explique pas mieux le blocage que le simple Bush-Mosteller → l'adéquation cue-competition
  du R-W vanilla est **surestimée**.

- **Apprentissage de structure causale** [Tomov et al., *Neural Computations Underlying Causal Structure
  Learning*, J. Neurosci. 2018, doi:10.1523/JNEUROSCI.3336-17.2018 ; adapté de Gershman 2017]. La
  compétition d'indices est gouvernée par la **structure causale supposée**, pas par la co-occurrence
  brute : compétition entre **causes candidates** (apprentissage prédictif) mais **pas** entre effets
  (apprentissage diagnostique). L'inférence de structure est **neuralement et computationnellement
  distincte** de l'apprentissage associatif (division du travail : substrat structurel lent ↔ têtes
  associatives rapides). Bien modélisé comme **inférence bayésienne sur un espace discret de structures
  latentes**, qui permet la généralisation à des combinaisons indice/contexte inédites.

- **Cadre à étages de sélectivité** [Boddez, Haesen, Baeyens, Beckers, *Selectivity in associative
  learning: a cognitive stage framework for blocking and cue competition phenomena*, Frontiers in
  Psychology 2014]. Décompose le blocage en **acquisition vs expression** : un indice redondant est-il
  vraiment *écarté* ou seulement *non exprimé* ? → informe **où** placer l'attribution de crédit dans
  le pipeline slot (au temps d'apprentissage ou au read-out). L'attention est modulée par la
  **prédictivité relative** (Mackintosh 1975 : plus d'attention à un meilleur prédicteur) ; variante
  Pearce-Hall : attention aux indices suivis d'une issue **surprenante**.

- **Décorrélation / contingence ΔP / revaluation rétrospective.** L'identifiabilité s'obtient en
  **décorrélant** indice et contexte à travers les blocs (un indice cause l'issue dans *tous* les
  contextes, un autre jamais → marque le neutre). La **revaluation rétrospective** [Aitken, Larkin,
  Dickinson, Q.J.Exp.Psychol. 2001] : après A+X, voir **A seul** réduit *rétroactivement* le crédit de
  X → une **passe de replay batchée** (re-visiter des épisodes stockés) réalise une correction
  rétrospective qu'une règle purement forward/online rate.

**→ Mapping Sylvan.** Ne pas compter sur la co-occurrence pour dé-pondérer un distracteur neutre :
prévoir soit (a) une init intermédiaire + R-W modifié sur le lien slot→drive (cheap, online), soit
(b) — plus robuste — une **passe jour/nuit d'inférence de structure** (bayésienne, sur épisodes
vécus) qui **décorrèle** les apparences confondues que l'update online ne peut pas démêler, exactement
le pattern « substrat structurel lent + têtes rapides ». Cohérent avec le blocage Rescorla-Wagner déjà
noté comme verrou dans MEMORY (monde-incrémental). Décision « re-cluster jour/nuit vs online » ↔ axe 4.

---

## Axe 3 — Affordance d'obstacle par erreur de prédiction du WM

Absorber la dynamique d'obstacle **dans** le WM (le planner l'évite par rollout, sans coût codé) **vs**
un coût/affordance appris **séparé**. **⚠️ Axe le plus mince du digest** : presque toutes les preuves
plaident pour un **prédicteur séparé** ; l'« absorber-dans-le-WM » n'a qu'une référence vague.

**Mécanismes :**

- **Affordances de Montesano** [Montesano et al., *Learning Object Affordances* — nommé « le papier
  affordance canonique » par le digest ; réf. canonique IEEE T-RO 2008]. Réseau bayésien de
  dépendances entre **(action, features-objet, effet)**, appris de l'expérience motrice, puis utilisé
  comme **modèle forward** pour prédiction/planification/imitation. **Fait l'attribution de crédit de
  l'axe 2 gratuitement** : empiriquement, **la couleur est détectée non pertinente** pour chaque action
  tandis que forme/taille sont retenues (elles changent vitesse/contact observés). Données
  **interventionnelles gratuites** (l'agent *choisit* action+objet = variables d'intervention →
  condition d'identifiabilité). Effets = **changements saillants** de l'état perceptif, clusterisés en
  ensemble **ouvert** (vitesse nulle/petite/grande ; contact persistant). Zéro label.

- **GVF « près d'un obstacle »** (Horde, cf. axe 1) : P(pic capteur-contact sous quelques secondes) —
  l'affordance d'obstacle **par prédiction du contact**, apprise, composable.

- **Traversabilité par erreur de reconstruction** [travaux off-road/véhicule ; auteur/année non portés
  par le digest]. Un **auto-encodeur entraîné UNIQUEMENT sur le terrain réellement traversé** (sûr)
  signale les obstacles comme **erreur de reconstruction élevée** au test — zéro label, purement du
  vécu de traversée (sol vs végétation séparés à **81-85 %** vs labels). C'est le **jumeau exact de la
  « lunette saillance-danger » apprise** que Sylvan utilise déjà pour le danger. **Caveats** : (a) c'est
  une **anomalie d'apparence-reconstruction, PAS une erreur de prédiction de dynamique du WM** ; (b)
  plafonne bien sous le supervisé ; (c) **contamination** — les régions occluses (végétation) doivent
  être filtrées sinon elles entrent à tort dans l'ensemble « sûr » (analogue au risque d'attribution
  quand des apparences co-occurrent à l'interaction — cf. axe 2).

- **Traversabilité self-supervisée depuis commandé-vs-réel** [self-supervised traversability + MPC ;
  auteur non porté]. Labels **bootstrappés du wheel-slip / mouvement commandé-vs-réel** — *exactement*
  le signal « le WM prédit un déplacement, le déplacement réel ≈ 0 ⇒ collision » — puis fourni à un
  **planner model-predictive**. **Voie médiane** entre absorber-dans-le-WM et coût-appris-séparé, que
  Sylvan pourrait brancher sur l'étage waypoint/critique.

- **Évitement model-based émergent** [WM-based navigation ; référence unique, vague, sans nom d'auteur
  dans le digest]. L'évitement **émerge du planning dans un modèle de dynamique appris**, sans terme de
  coût-obstacle explicite → seule preuve du côté « absorber la collision dans le WM pour que les rollouts
  `(vx, ω)` contournent tout seuls ». **À traiter comme piste, pas comme méthode prouvée** (le digest
  est mince ici).

**→ Mapping Sylvan.** Deux options honnêtes : (1) **Prédicteur d'affordance séparé** (le mieux
étayé) — un auto-encodeur/GVF « traversabilité » appris du vécu commandé-vs-réel, lu par le planner
comme un coût, dans la lignée exacte de la lunette saillance-danger déjà vivante et pure. (2)
**Absorber dans le WM** — séduisant (le planner MPC évite par rollout, zéro coût codé) mais **peu
étayé** : nécessiterait re-collecter/re-entraîner le WM avec des interactions d'obstacle pour que la
dynamique `(vx,ω)→déplacement` encode le blocage — coûteux et à gater derrière un test gratuit (cf.
§1 CLAUDE.md ; et attention au SIGNAL D'ALERTE §3 : ne pas re-entraîner le WM pour *une* ressource).
Le signal commandé-vs-réalisé est le pont naturel des deux options.

---

## Axe 4 — Découverte non-supervisée de classes d'objets, K non-stationnaire, continu

Nombre de classes **ouvert et croissant**, stabilité continual (pas d'oubli catastrophique), **re-cluster
jour/nuit batch vs online**.

**Mécanismes :**

- **STAM / Unsupervised Progressive Learning** [*Unsupervised Progressive Learning* + architecture STAM ;
  la correspondance la plus serrée avec l'axe 4 ; auteur/année non portés par le digest]. Apprenant de
  représentation **online** sur flux **non-stationnaire non-labellisé** où le **nombre de classes croît**,
  **sans stockage ni replay**. Recette : **clustering online + détection de nouveauté + oubli des
  outliers + mémoire de prototypes (centroïdes)** — stocke des *features prototypiques*, pas des exemples.
  CPU-friendly ; évite l'oubli catastrophique par **persistance de prototypes** (pas de replay/régularisation
  de gradient) ; benchmarké contre MAS (régularisation) et GEM (replay). **Failure mode signalé** : STAM
  **« oublie les outliers »** → risque de **jeter des événements rares mais significatifs** (argument fort
  pour une passe **batch jour/nuit** qui les capture). *NB* : la proposition STAM préliminaire (colonne
  corticale + predictive-coding hiérarchique) **n'a ni formulation math ni expériences** — pas de preuve
  empirique de fragilité online vs batch dans ce papier-là.

- **Détection de nouvelle classe par Page-Hinckley sur ELBO** [détection de dérive de concept ;
  auteur non porté par le digest]. Test de changement Page-Hinckley appliqué à la **vraisemblance ELBO**
  du modèle, exploitant la **continuité temporelle** du flux d'objets → **lisse le bruit ELBO** et
  **réduit drastiquement la sur-segmentation** vs CURL (par-outlier). Preuve chiffrée : sur MNIST
  ~**10 composantes** (vraies classes = 10, AMI 0.746 / ARI 0.70) **vs CURL 94-120, SOINN 1204** → le
  clustering online naïf par-outlier **sur-segmente catastrophiquement**. Buffer rempli après un
  changement de catégorie détecté, non utilisé immédiatement (anti-overfit).

- **ART + iCVIs** [Adaptive Resonance Theory + incremental Cluster Validity Indices]. Croissance de
  prototypes online sous **seuil de vigilance** ; les iCVIs = contrôleur online de création/fusion de
  prototypes → utile pour décider les seuils de croissance et pour un **check de validité batch qui
  gate un re-cluster jour/nuit**. Sensible au drift de paramètre/volume de données.

- **Argument théorique décisif « sur-clusteriser online, consolider en batch »** : un clusterer
  **strictement online single-pass sans fusion NE PEUT PAS** se remettre d'un split erroné précoce ;
  **autoriser des clusters en excès** (sur-clusteriser online puis fusionner plus tard) rend le problème
  **traitable**. C'est **l'argument formel** pour « croître libéralement online, consolider en passe
  jour/nuit », et explique pourquoi l'online pur est fragile **précisément sur les classes rares/tardives**.

- **Dérive de représentation = fausse nouveauté** : la **dérive interne** d'objets connus non-revus
  longtemps provoque de fausses détections de nouveauté et de la sur-segmentation — la fragilité online
  qui motive le re-cluster batché. Réglages open-world récents utilisent l'**UQ** pour distinguer vrai-
  nouveau vs bruit avant de committer une classe.

**→ Mapping Sylvan.** Converge fortement vers le choix déjà pressenti dans MEMORY (monde-incrémental,
« reconnaissance des types AVANT jour/nuit ; jour/nuit orthogonal à la pureté ») : **croître les
prototypes libéralement en ligne, mais consolider/re-clusteriser dans une passe batch jour/nuit** (où
Page-Hinckley/ELBO + un CVI décident le K, et où les événements rares — un danger vu 3 fois — ne sont
pas « oubliés comme outliers »). Les prototypes = banque de requêtes-couleur ; le K découvert alimente
le nombre de têtes-GVF de l'axe 1. Cohérent avec le résultat déjà vivant `build_typed_slots.py`
(K=3 **mesuré**, pas fitté).

---

## Axe 5 — Motivation intrinsèque pour la découverte ouverte, sans casser la survie

**Mécanismes :**

- **Curiosité régularisée par l'homéostasie** [combinaison curiosité info-théorique + terme homéostatique ;
  auteur/année non portés par le digest]. La récompense de curiosité (erreur de forward-model) est rendue
  **cohérente avec un drive homéostatique** qui garde les variables corporelles critiques **bornées** —
  curiosité *régularisée* par l'homéostasie, pas juste additionnée. Formalise « sonder l'inconnu sans
  casser la survie ». Résultat notable : l'ajout du terme homéostatique **augmente** le gain d'information
  global (l'homéostasie ne fait pas que protéger, elle **améliore** l'objectif d'exploration). Fond
  survie : variables physiologiques à **set-points**, récompense = **réduction de drive** (Hull/
  **Keramati-Gutkin**), actions continues — forme principielle du coût multi-drive contre lequel arbitrer
  la curiosité.

- **Plan2Explore** [Sekar et al., *Planning to Explore via Self-Supervised World Models*, ICML 2020 —
  nommé par le digest ; « best substrate match »]. Motivation intrinsèque = **désaccord d'un ensemble
  dans l'espace latent** (≈ gain d'information *attendu*) **dans un WM appris** ; la politique
  d'exploration est entraînée **purement sur des rollouts imaginés** — exactement « WM + planner MPC
  sur rollouts imaginés ». Comme le désaccord s'évalue sur des trajectoires rêvées, l'agent peut
  **planifier** pour visiter un objet de conséquence inconnue **avant d'agir**. Une phase
  self-supervisée task-agnostique donne un WM ré-utilisable pour des tâches **spécifiées plus tard**
  (zéro/few-shot), presque au niveau d'un oracle. **Caveats** : Plan2Explore vanilla est **reward-free
  (exploration pure)** → **doit être mélangé au coût homéostatique** sinon la curiosité **tue** l'entité
  (l'arbitrage même de la question) ; validé sur pixels/Dreamer, **pas** sur rétine bas-dim (transfert
  non testé).

- **RND — Random Network Distillation** [Burda et al., *Exploration by Random Network Distillation*,
  ICLR 2019 — nommé par le digest]. Bonus = **MSE d'un prédicteur entraîné à imiter un réseau-cible
  aléatoire FIXE** ; élevé sur les états nouveaux, décroît avec la familiarité. **Le plus CPU-friendly
  et le plus reproduit** : un forward pass par batch, compatible avec toute optim de politique. **Évite
  le « noisy-TV »** (cible = fonction *déterministe* de l'observation courante, pas une prédiction de
  next-state stochastique → ne fixe pas sur des canaux d'apparence bruités de la rétine). **Design-clé
  survie** : RND sépare têtes de valeur **intrinsèque / extrinsèque** avec **des discounts différents**
  pour que la nouveauté **ne noie pas** le retour de survie — l'arbitrage exact visé. Preuve : SOTA
  Montezuma's Revenge (8152, 1er > humain moyen sans démos/état interne).

- **Empowerment vs entropie vs gain d'info** [comparaison en environnement ouvert *Crafter* ; auteur
  non porté]. Parmi trois objectifs intrinsèques, **seuls Entropie et Empowerment** corrèlent avec le
  progrès d'exploration humain — **le Gain d'Information, non**. Profils temporels distincts :
  **l'entropie monte vite puis plafonne** (utile **tôt**, pour *trouver* les objets), **l'empowerment
  monte continûment** (à prioriser **tard**, pour *maîtriser* les affordances) → suggère un **planning**
  (nouveauté d'abord, contrôle ensuite). L'empowerment biaise vers des cibles **contrôlables**
  (traversable, food fiable) plutôt que vers l'imprévisible pur (qui attire vers l'incontrôlable/
  dangereux = risque de survie).

**→ Mapping Sylvan.** L'erreur de prédiction du WM gelé = signal de curiosité **déjà disponible** ;
le coût multi-drive de survie = terme homéostatique. Le plus sûr/cheap : **RND** (têtes intrinsèque/
extrinsèque à discounts séparés) bolté sur les features slot/rétine, **plafonné** pour ne jamais
dominer la survie. Plus ambitieux et mieux aligné au substrat : **Plan2Explore** (désaccord d'ensemble
évalué sur rollouts `(vx,ω)` imaginés) — mais **impérativement mélangé** au coût homéostatique, et à
gater derrière un test gratuit (le digest avertit : reward-free = mortel ; régime pixels non transféré).
Schéma d'ordonnancement plausible : entropie/nouveauté tôt (trouver les couleurs inconnues),
empowerment tard (maîtriser leurs conséquences).

---

## Synthèse transversale (une phrase par axe)

1. **Un WM gelé + une tête-GVF par pulsion** (Horde/SF/HRA/OC-GVF), off-policy, coût constant/pas —
   le split « substrat lent / têtes rapides » de CLAUDE.md est déjà l'état de l'art.
2. **La co-occurrence ne suffit PAS** à rejeter un distracteur neutre (négatif colour-word) → prévoir
   une **décorrélation/structure explicite**, idéalement en passe batch jour/nuit + init-incertitude R-W.
3. **Affordance d'obstacle = plutôt un prédicteur séparé appris** (recon-error / commandé-vs-réel, façon
   lunette-danger) ; « absorber dans le WM » est séduisant mais **peu étayé** dans le digest.
4. **Croître les prototypes online, consolider en batch jour/nuit** (argument formel + Page-Hinckley/ELBO
   contre la sur-segmentation) — convergent avec la décision monde-incrémental déjà notée.
5. **Curiosité plafonnée et régularisée par l'homéostasie** (RND cheap ; Plan2Explore aligné-substrat
   mais à mélanger au coût de survie) — jamais reward-free.

---

## Sources

Citées telles qu'elles apparaissent dans le digest récupéré ; métadonnées complétées seulement quand
le digest les porte. « (méthode nommée) » = le digest nomme la méthode sans auteur/année complet.

**Axe 1**
- Sutton, Modayil, Delp, Degris, Pilarski, White, Precup — *Horde: A Scalable Real-time Architecture for
  Learning Knowledge from Unsupervised Sensorimotor Interaction*, AAMAS 2011 (PDF intégral dans le digest).
- Barreto et al. — *Successor Features for Transfer in Reinforcement Learning*, NeurIPS 2017,
  arXiv:1606.05312 (+ caveats : *Advantages and Limitations of using Successor Features*, ResearchGate
  318849333 ; *Successor Feature Representations*, arXiv:2110.15701, openreview MTFf1rDDEI ;
  *Non-Linear Rewards for SFs*, openreview 2KSsaPGemn2).
- Schaul et al. — *Universal Value Function Approximators*, ICML 2015 (proceedings.mlr.press/v37/schaul15).
- Nath, Subbaraj, Khetarpal, Ebrahimi Kahou — *Discovering Object-Centric Generalized Value Functions
  From Pixels*, ICML 2023, arXiv:2304.13892.
- van Seijen, Fatemi, Romoff et al. (Microsoft Maluuba) — *Hybrid Reward Architecture for Reinforcement
  Learning*, NIPS 2017 (code : github.com/Maluuba/hra).

**Axe 2**
- Rescorla & Wagner 1972 (R-W classique) ; variante **R-W modifié à init intermédiaire** (méthode nommée) ;
  Bush-Mosteller (comparaison).
- Tomov et al. — *Neural Computations Underlying Causal Structure Learning*, J. Neurosci. 2018,
  doi:10.1523/JNEUROSCI.3336-17.2018 (PMC6083455) ; s'appuie sur Gershman 2017, Griffiths & Tenenbaum 2005.
- Boddez, Haesen, Baeyens, Beckers — *Selectivity in associative learning: a cognitive stage framework
  for blocking and cue competition phenomena*, Frontiers in Psychology 2014 (PMC4228836) ; Mackintosh 1975,
  Pearce-Hall (attention).
- Aitken, Larkin, Dickinson — *Re-examination of within-compound associations in retrospective revaluation
  of causal judgements*, Q.J.Exp.Psychol. 2001 (pmid 11216299) — revaluation rétrospective.
- Étude(s) **incident colour-word contingency** montrant l'absence de blocage/surombrage (méthode nommée,
  deux grandes études en ligne ; auteur/année non portés par le digest).

**Axe 3**
- Montesano et al. — *Learning Object Affordances* (réseau bayésien action-features-effet ; « papier
  affordance canonique » ; réf. canonique IEEE T-RO 2008).
- Traversabilité par **erreur de reconstruction** d'auto-encodeur sur terrain traversé (81-85 % vs labels ;
  auteur non porté) ; **traversabilité self-supervisée commandé-vs-réel + MPC** (auteur non porté) ;
  traversabilité SVM behaviour-based (auteur non porté).
- **Navigation model-based à évitement émergent** (rollouts dans un WM appris ; référence unique et vague,
  sans nom d'auteur dans le digest — piste, non prouvée).
- Composition GVF « près d'un obstacle » (Horde, ci-dessus).

**Axe 4**
- **Unsupervised Progressive Learning / architecture STAM** (méthode nommée ; clustering online + nouveauté
  + oubli-outliers + prototypes ; benchmarks MAS/GEM ; auteur/année non portés).
- Détection de nouvelle classe **Page-Hinckley sur ELBO** (vs CURL/SOINN sur MNIST ; auteur non porté).
- **ART + iCVIs** (vigilance + validity indices incrémentaux, méthode nommée).
- Argument théorique **over-cluster-online-puis-merge-batch** (méthode/résultat nommé) ; dérive de
  représentation → fausse nouveauté ; open-world continual à UQ (méthode nommée).

**Axe 5**
- **Curiosité info-théorique + terme homéostatique** (méthode nommée ; améliore le gain d'info) ;
  **Keramati & Gutkin** — RL homéostatique par réduction de drive (nommé).
- Sekar et al. — *Planning to Explore via Self-Supervised World Models* (**Plan2Explore**), ICML 2020
  (nommé par le digest).
- Burda et al. — *Exploration by Random Network Distillation* (**RND**), ICLR 2019 (nommé ; Montezuma 8152).
- Comparaison **Entropy / Information Gain / Empowerment** en environnement *Crafter* (méthode nommée ;
  auteur non porté).

*(Bruit écarté : Bloc 140 = dump base64 d'image JPEG, sans contenu textuel.)*
