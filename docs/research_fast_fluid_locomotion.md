# Recherche : locomotion RAPIDE & FLUIDE (salamandre sprawlée, CPG+PPO, 8 CPU)

Deep-research multi-agents 2026-06-16. ~100 agents, sources primaires vérifiées (3-vote).
Question : pousser ~0.15 → ~1.0 m/s (droite) / 0.8 (courbe), fluide, sur notre corps sprawlé
(trot diagonal CPG + résidu PPO borné). Contexte : virage-en-avançant DÉJÀ résolu (symétrie +
curriculum). Échecs empiriques injectés : servo↑, cadence↑, foulée↑, S-wave naïf horloge-locké,
de-sprawl warm-start. Voir [[sylvan-phase-a-progress]].

## VERDICT : c'est de la COORDINATION, pas la morphologie
Les tétrapodes sprawlés réels (iguane du désert) atteignent plusieurs m/s. La lenteur vient du
contrôle/couplage, pas d'un plafond du corps. Tout ce qu'on a essayé a échoué pour des raisons
connues et documentées.

## Pourquoi chaque tentative a échoué (HAUTE confiance, 3-0)
1. **Vitesse = foulée ET fréquence montent ENSEMBLE pendant que le duty factor CHUTE** (iguane :
   50→350 cm/s = foulée 13→39 cm + fréq 3.9→8.6 Hz + duty 63%→34%). → cranker UN seul bouton
   (cadence seule, foulée seule) déstabilise forcément. (Fieler & Jayne 1998, JEB)
2. **Haute vitesse = posture PLUS DRESSÉE** : à grande vitesse la longueur effective de patte ~double
   (genou tendu, fémur plus vertical, talon décolle/digitigrade), la hanche monte. → réduire le sprawl
   PUIS warm-starter a échoué (chute) car **changer la posture = changement de RÉGIME cinématique, pas
   une perturbation** → exige un ré-entraînement DE ZÉRO ou un curriculum recuit. La morpho n'est PAS
   le mur ; "se dresser à la vitesse" est le mode naturel.
3. **RACINE de notre échec S-wave (d)** : l'ondulation latérale EST un vrai mécanisme d'allongement de
   foulée, MAIS seulement bien phasée : marche = **onde STATIONNAIRE en S, nœuds verrouillés aux
   ceintures (girdles)**, chaque appui synchronisé à l'angle extrême de la ceinture pour que **le pied
   ancré serve de PIVOT**. Une S-wave libre/horloge-lockée (ce qu'on a fait) SCRUBE au lieu d'allonger.
   (Ijspeert et al. Science 2007 ; Frontiers neurorobotics 2021)

## Le FIX concret (HAUTE confiance)
4. **Couplage corps-pattes piloté par le CONTACT, pas par l'horloge.** Un oscillateur de phase par
   patte avec retour de **force de réaction au sol (GRF)** : `φ̇_i = ω − σ·N_i·cos(φ_i)` (N_i = GRF du
   pied, tire la phase vers l'appui ~3π/2 tant que chargé). Ça auto-organise le timing des appuis ET
   fait plier le corps vers l'avant-patte ancrée. **Aucun couplage inter-oscillateur nécessaire — la
   physique du corps porte la coordination.** (Owaki/Ishiguro, "Tegotae"). C'est notre règle
   implémentable pour le couplage colonne.
5. **Transitions de démarche (marche→trot→galop) ET bascule onde stationnaire→onde progressive** émergent
   d'un SEUL paramètre de vitesse ω (vitesse angulaire intrinsèque des oscillateurs de pattes). Le TYPE
   d'onde doit matcher la vitesse cible (stationnaire à vitesse modérée → progressive postérieure à haute
   vitesse).
6. **CPG-RL : faire de l'AMPLITUDE et la FRÉQUENCE des oscillateurs l'espace d'action du RL.** Garder le
   CPG codé main, laisser le résidu PPO moduler ω/amplitude + fournir le couplage colonne piloté-contact.
   (Ijspeert lab ; démontré sur quadrupède DRESSÉ A1, pas sprawlé → transfert ondulation non prouvé.)
7. **Bio** : un seul circuit spinal partagé (CPG axial type-lamproie + oscillateurs de pattes plus LENTS),
   modulé par un drive descendant tonique (bas=marche, haut=nage). Colonne et pattes = sous-circuits
   couplés, pas indépendants → un seul commande haut-niveau "vitesse/drive".

## Caveats
Base de sources étroite (lignée Ijspeert/Owaki/Ishiguro EPFL/Tohoku, haute qualité + sur-morpho mais
narrow). Les papiers CPG-salamandre n'utilisent PAS de RL → "ω comme action RL" = extrapolation
raisonnable, pas un résultat démontré. La seule démo CPG-RL action-space est sur A1 dressé. Chiffres
bioméca de l'iguane (dressable, ~3.5 m/s, plus rapide qu'une salamandre) → DIRECTION prouvée, MAGNITUDE
pour notre gabarit incertaine.

## Questions ouvertes
- 1.0 m/s atteignable pour NOTRE gabarit précis, ou faut-il jambes plus longues / posture dressable ?
  (direction prouvée, magnitude non).
- Phase exacte appui↔angle-extrême-ceinture pour un trot DIAGONAL ; le résidu peut-il la découvrir du
  contact seul, ou faut-il l'encoder partiellement dans le CPG pour éviter le scrubbing ?
- Le retour GRF (`φ̇=ω−σN cos φ`) déstabilise-t-il l'entraînement à 8 envs ? scheduling de σ et du gain
  de couplage colonne le long du curriculum de vitesse (stationnaire→progressive) ?
- Le passage à posture dressée (+ éventuel trot→bound) exige-t-il from-scratch, ou un curriculum recuit
  posture+vitesse porte-t-il la policy actuelle sans la chute 20-40% du warm-start ?

## Plan d'action Sylvan (priorisé)
1. **Couplage colonne piloté-contact** (#4) — remplacer la S-wave horloge-lockée par une cible de flexion
   pilotée par les contacts pieds (plier vers l'avant-patte en appui ; nœuds aux ceintures). On A déjà les
   foot_contacts. C'est LE mécanisme de propulsion qu'on avait faux + la fluidité. PRIORITÉ 1.
2. **Curriculum de vitesse couplé** (#1) — récompenser foulée ET fréquence ensemble + duty factor qui
   baisse à haute vitesse commandée ; ω (fréquence) modulé par la commande/le résidu (#6).
3. **Posture-vitesse** (#2) — lier sprawl/longueur-effective à la vitesse commandée ; tout changement de
   posture = from-scratch ou curriculum recuit (PAS warm-start).
4. **Transition de démarche** (#5) si besoin de pousser au-delà.
