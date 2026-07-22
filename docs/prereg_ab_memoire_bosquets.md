A/B MEMOIRE — CRITERES PRE-INSCRITS (ecrits AVANT lancement, 2026-07-22)

Monde : 4 bosquets x 2 baies, repousse 2500, mono-pulsion, espacement 9-11 m (config CALIBREE).
Bras  : MEM=off  vs  MEM=on (--egomotion-head + --slot-memory).
Taille: 2 graines x 20 vies = 40 vies par bras.

METRIQUE PRIMAIRE = REPAS par vie (la survie sature et est bruitee ; les repas ne saturent pas).
  PASS    : (ON - OFF) >= +0.8 repas median poole, ET la direction tient sur CHAQUE graine prise seule.
  PARTIEL : gain >= +0.8 poole mais direction inversee sur une graine -> non concluant, re-mesurer.
  NUL     : un bras a >= 60 % des vies au plancher de famine (2000 pas) -> le monde est casse, verdict void.
  KILL    : ON < OFF sur la primaire -> la memoire NUIT, negatif banke.

METRIQUE SECONDAIRE = fraction d episodes pleins.
  Confirmation attendue : (ON - OFF) >= +10 points. Ne peut PAS sauver un echec sur la primaire.

CE QUE CE TEST NE DIT PAS : il juge la memoire dans CE monde mono-pulsion. Il ne dit rien du
multi-drive (mur arithmetique banke) ni de la generalisation a un autre agencement de bosquets.

--- AVENANT (2026-07-22) : meme pre-inscription appliquee au CONE ---
Condition : FOV=120 (vrai cone, 36 rayons redistribues a 3,33 deg) + KIN_TURN=6.0 (x4).
Justification mesuree AVANT lancement : hors-vue 6,2 % -> 73,2 % ; l entite survit (3/3
au-dessus du plancher) car un tour complet coute 13 % du budget au lieu de 52 %.
METRIQUES ET BARRES INCHANGEES (repas moyens, direction par graine, plancher). Aucune barre
n est deplacee : seule la condition change, et elle est declaree ici avant le run.

--- CAVEAT D INTERPRETATION, ecrit AVANT que le resultat du cone soit connu (2026-07-22) ---
VERIFIE DANS LE CODE : le cout du planner ne contient AUCUN terme d information / incertitude /
epistemique (0 occurrence dans command_planner.py et serve_planner_command.py). Il note chaque
candidat par la survie attendue ETANT DONNE la croyance courante. Tourner pour regarder n ameliore
donc jamais le score, puisque le score ignore que regarder changerait la croyance.
(`explore_target` l.1040-1072 n est PAS un comportement : c est la machinerie epsilon du chantier
arbitrage, etiquetee dans le code "decision FORCEE, pas la politique".)

CONSEQUENCE SUR LA LECTURE DU RESULTAT :
- Le balayage n est PAS code -- rien n a ete ecrit pour ca. Mais il ne peut pas EMERGER non plus.
- L entite tourne pour se DEPLACER ; son cone balaie le monde par effet de bord, pas par intention.
- Donc si la memoire ne paie pas sous le cone, DEUX explications restent ouvertes et ce test ne
  les separe PAS : (a) la memoire est inutile ici, (b) l entite ne REGARDE jamais, donc elle
  n encode presque rien a memoriser. Ne pas conclure (a) sans avoir teste le regard.
- Symetriquement, si la memoire paie, c est un gain PASSIF (retenir ce qu on a vu en allant
  ailleurs), pas une perception active. Ne pas le revendiquer comme de la perception active.
