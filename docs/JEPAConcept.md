Résumé : A Path Towards Autonomous Machine Intelligence (Yann LeCun, 2022)
Ce document propose une architecture cognitive permettant aux machines d'apprendre, de raisonner et de planifier de manière autonome, en s'inspirant de l'efficacité de l'apprentissage animal et humain. L'objectif est de dépasser les limites actuelles du Machine Learning (qui nécessite trop de données et d'essais-erreurs) grâce à des modèles prédictifs et une motivation intrinsèque.

Concepts Fondamentaux
World Models (Modèles du Monde) : La capacité fondamentale de l'agent à posséder un modèle interne du fonctionnement du monde. Cela lui permet de prédire les conséquences de ses actions, de raisonner par anticipation et de combler les informations manquantes sans avoir à interagir physiquement à chaque fois.

Motivation Intrinsèque : Le comportement de l'agent n'est pas dicté par des règles codées en dur ou une supervision externe stricte, mais par la recherche de la minimisation d'un "coût" interne (analogue à l'évitement de la douleur ou à la recherche de satisfaction).

Self-Supervised Learning (Apprentissage Auto-Supervisé - SSL) : L'agent apprend majoritairement par simple observation du monde (pour comprendre la physique, la permanence des objets, etc.) afin de minimiser le nombre d'interactions coûteuses ou dangereuses requises pour apprendre une tâche.

Architecture du Système (Les 6 Modules)
L'agent est composé de six modules interconnectés et différentiables :

Configurator (Configurateur) : Le contrôleur exécutif. Il reçoit les inputs de tous les modules et configure leurs paramètres et leur attention pour la tâche spécifique en cours.

Perception : Reçoit les signaux des capteurs et extrait une représentation abstraite et utile de l'état actuel du monde.

World Model (Modèle du Monde) : Le cœur du système. Il simule et prédit les états futurs plausibles du monde en fonction des actions imaginées par l'agent. Il gère l'incertitude du monde réel.

Cost (Coût / Énergie) : Mesure le niveau "d'inconfort" de l'agent via un scalaire (l'Énergie). Il est composé du Intrinsic Cost (immuable, définit les pulsions de base comme la faim ou l'évitement du danger) et du Trainable Critic (qui apprend à prédire les coûts futurs pour anticiper les récompenses/punitions).

Short-Term Memory (Mémoire à Court Terme) : Stocke les états passés, actuels et futurs prédits, ainsi que leurs coûts associés. Sert de contexte pour l'apprentissage et la planification.

Actor (Acteur) : Propose des séquences d'actions, utilise le World Model pour en simuler les résultats, et calcule via des méthodes de gradient la séquence optimale qui minimisera le coût futur avant d'envoyer la commande aux effecteurs.

Les Deux Modes de Pensée
L'architecture reproduit un fonctionnement cognitif dual (similaire aux Systèmes 1 et 2 de Daniel Kahneman) :

Mode-1 (Réactif) : Action immédiate basée sur la perception. L'Acteur utilise une "politique" (policy) directe sans passer par des simulations complexes. Rapide et peu coûteux en énergie.

Mode-2 (Raisonnement et Planification) : Processus lourd où l'agent imagine des séquences d'actions, simule leurs résultats via le World Model, évalue le coût et optimise son plan. Les compétences apprises avec effort en Mode-2 sont ensuite "compilées" pour devenir des réflexes en Mode-1.

Le Moteur Technique : JEPA et H-JEPA
L'architecture rejette les modèles "génératifs" classiques (qui tentent de prédire chaque détail, comme chaque pixel d'une vidéo, ce qui est inefficace et sujet à l'erreur face à l'imprévisibilité du monde).

JEPA (Joint Embedding Predictive Architecture) : Une architecture qui effectue ses prédictions dans un espace de représentation abstraite. Les encodeurs suppriment les détails non pertinents ou aléatoires de l'environnement pour ne garder que l'information utile.

Variables Latentes : Utilisées par le JEPA pour représenter l'incertitude (les éléments qui ne peuvent pas être déduits du passé, comme le choix d'un autre conducteur de tourner à gauche ou à droite).

H-JEPA (Hierarchical JEPA) : Un empilement de modèles JEPA permettant de faire des prédictions à plusieurs niveaux d'abstraction et sur de multiples échelles de temps. Le bas niveau gère les détails à très court terme (millisecondes), le haut niveau gère la planification abstraite à long terme (heures).

Le Paradigme d'Apprentissage
Contre les méthodes contrastives : L'auteur argumente que les méthodes d'apprentissage contrastives (qui nécessitent de comparer des exemples positifs avec des exemples négatifs générés artificiellement) s'effondrent face à la haute dimension (le "fléau de la dimension").

Privilégier les méthodes régularisées (Non-contrastives) : L'entraînement du JEPA doit utiliser des critères comme VICReg : maximiser le contenu informatif des représentations (pour éviter qu'elles ne s'effondrent sur une valeur constante), tout en limitant la capacité d'information de la variable latente pour forcer le modèle à réellement apprendre la dynamique du monde.
