# Recherche : locomotion omnidirectionnelle rapide & fluide (salamandre/gecko, 8 envs CPU)

Deep-research multi-agents, 2026-06-16. 19 sources primaires, 91 claims extraits, 25 vérifiés
(adversarial 3-vote), 21 confirmés, 4 écartés. Question : faire apprendre à une policy RL
command-conditionnée (8 CPU, PPO/SAC, base CPG) un déplacement RAPIDE, STABLE, omnidirectionnel —
surtout **tourner-en-avançant** (course en courbe, pas pivot) — sans que la moyenne déterministe
fige le virage.

## Contexte empirique (établi côté Sylvan, voir [[sylvan-cpg-openloop-wall]])
Régler les boutons du CPG en boucle ouverte (foulée, cadence, servo kp/vitesse, flexion colonne,
S-wave) NE donne PAS un gait rapide+stable+directionnel : grande foulée → bascule/spirale ;
servo non limitant. Le feedback RL est essentiel même en ligne droite (0.09→0.27 m/s). Vrai mur :
faire que le RL stabilise+accélère un VIRAGE comme il le fait pour la ligne droite, sans le figer.

## Leviers classés (pour 8 CPU)

### 1. Curriculum de commandes adaptatif (Grid) — confiance HAUTE, cause-racine
Échantillonner uniformément de grandes commandes FAIT échouer le RL (commandes fortes → reward
trop faible → moyenne déterministe fige le virage). Fix : grille (vx ~0.5 m/s × omega ~0.5 rad/s),
**démarrer près de zéro, élargir une cellule (4-connectée) seulement quand le reward franchit un
seuil**. C'est LA réponse à notre échec.
Sources : Margolis "Rapid Locomotion" RSS 2022 / IJRR 2024 (arxiv 2205.02824), curric refs
(2211.00458, 2505.10022, 2212.03238, nature s41598-024-79292-4).

### 2. Augmentation par symétrie (réflexion sagittale) dans PPO — confiance HAUTE, prouvé sur NOTRE morpho
Groupe de symétrie par réflexion sur les relations de phase ; augmentation des données PPO.
Sur le quadrupède type-salamandre exact : reward diagonal 4.57→6.55, variance 0.79→0.28.
Gratuit en envs. Miroiter gauche↔droite + inverser signes latéraux / omega / vy / colonne.
Source : papier Nankai nov. 2025 (arxiv 2511.08299v1).

### 3. Reward à transplanter — confiance HAUTE
Noyaux EXPONENTIELS sur la vitesse du COM (linéaire 1.0·dt, angulaire 0.5·dt), énergie 1e-3·dt,
dt=0.02. **Mesurer au COM, PAS à l'IMU** — l'oscillation de la colonne trompe l'IMU.
⚠️ Nuance Sylvan : nos noyaux exp s'étaient effondrés (vloss=0) FAUTE de curriculum (commandes
trop dures → gradient nul). Le curriculum (#1) les rend viables. Les deux vont ensemble.

### Fork d'architecture — NON tranché, choisir UN
- **Route A (colle à la base CPG)** : CPG-RL — policy module amplitude+fréquence par oscillateur,
  **+ amplitude latérale (y) pour l'omnidirectionnel**, couplages/biais de phase (couplage fort,
  poids ~10, fixe l'identité de démarche + couche de style séparée → 9 démarches sans réglage
  par démarche). Sources : CPG-RL RA-L 2022 (2211.00458), Visual CPG-RL ICRA 2024 (2212.14400),
  AllGaits 2024 (2411.04787).
- **Route B (Ijspeert mars 2026, NOTRE morpho exacte)** : résidu PPO DIRECT en espace articulaire
  sur posture nominale, pas de commande CPG, faible reward de phase Bézier, virage par asymétrie.
  Les auteurs : moduler les params CPG *limite* le répertoire. MAIS entraîné à 8192 envs GPU,
  0.23 m/s réel. Source : 2603.16683.
→ Pour 8 CPU, l'archi compte moins que #1+#2. Garder CPG+résidu (Route A légère), ABANDONNER le
  turn-fade (laisser le résidu posséder le virage).

### Colonne / S-wave — confiance HAUTE
Un CPG distribué modulable (burst generators) sert marche/virage/transitions ; seulement 2-3
"descending drives" fixent fréquence/déphasage de la chaîne axiale via deux valeurs → mappe sur
notre colonne 2-segments (S-wave). Sources : Frontiers neurorobotics 2020 (604426).

### AMP — dernier recours
Remplace les rewards de style à la main par un reward de style adversarial + reward de tâche
(vitesse/yaw). La référence n'a PAS besoin d'être du mocap → des rollouts CPG analytiques marchent.
Reward de tâche encore à régler. Sources : AMP Peng 2021 / Escontrela 2022 (2203.15103),
BCAMP MDPI 2025 (mdpi 15/6/3356).

## Écarté (3-vote)
- Vitesse de phase ±0 comme "enabler" clé de l'omnidirectionnel (1-2).
- "Phase-coverage reward" requis pour éviter le collapse (1-2).
- Couplages CPG comme variables apprenables/explorables au sens fort (0-3).
- Conditionner le discriminateur AMP sur la commande (0-3).

## Questions ouvertes
- L'augmentation par symétrie compose-t-elle avec colonne flexible 2-segments + couplage asymétrique ?
- À 8 CPU : modulation params CPG vs résidu PPO direct, lequel converge plus vite ?
- Un rollout CPG boucle-ouverte est-il une référence AMP assez riche (éviter mode-collapse) ?
- Comment coupler les 2-3 drives de la colonne au curriculum limbe/démarche pour que l'ondulation
  aide le virage ?

## Plan d'implémentation Sylvan (ordre)
- **Phase A** : curriculum de commandes adaptatif (#1) + reward COM exp (#3) + suppression turn-fade,
  réentraînement.
- **Phase B** : augmentation par symétrie dans la boucle PPO (#2 ; construire la carte miroir
  102 obs / 13 actions).
