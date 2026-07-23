GATE HORS-LIGNE DU CRITIQUE — rang INTRA-ETAT (2026-07-23)

POURQUOI. Les 3 echecs du critique sont UN bug de mesure : on jugeait au R2 POOLE, domine par la
variance INTER-etat, alors que le planner compare 117 candidats DANS LE MEME ETAT. Un modele qui
predit la moyenne par etat et classe au hasard obtient deja un bon R2 poole — c est litteralement
le « inne +0,437 » qu on prenait pour une reference.

LE PIEGE RESOLU. Verifier un classement exige la vraie cible pour PLUSIEURS candidats du meme
instant ; un corpus n en observe qu UN. Ni reve du WM (circulaire) ni re-collecte (chere) : le monde
est GELE et le corps CINEMATIQUE, donc la cible des 117 candidats se CALCULE (simulateur valide par
diag_consequence_g0). Le gate mesure donc un PLAFOND : cibles exactes, zero bruit d observation.
Un echec y est decisif ; un succes est necessaire, pas suffisant.

CRITERES PRE-INSCRITS : pairwise > 0,65 ET Kendall tau > 0,30 ET (appris - inne) > 0,05.
CONTROLE : cibles permutees intra-etat doivent retomber dans [0,45 ; 0,55] (sinon fuite).

--- RESULTAT (120 etats, 21 candidats, replan=60, delta=600, split PAR ETAT) ---
  predicteur                 pairwise   Kendall tau   regret@1
  inne (proxy « va au plus proche »)  0,243      -0,514      7,51
  tete apprise (ridge)                0,712      +0,424      5,92
  controle permute                    0,476      -0,047      6,59

Selfcheck : oracle 1,000 / hasard 0,406 / predicteur inverse 0,000 -> la metrique est saine.
Controle permute a 0,476, DANS la fenetre -> pas de fuite.

VERDICT PARTIEL : **la tete CLASSE** (0,712 et tau 0,424, au-dessus des barres absolues), et c est
un resultat propre, controle.

🚨 **MAIS LE « +0,469 vs l inne » N EST PAS REVENDICABLE.** Mon proxy d inne (« aller vers le
bosquet non-vide le plus proche ») classe SOUS LE HASARD (0,243, tau -0,514). C est un HOMME DE
PAILLE — le meme piege qu avec `greedy` la veille — car le VRAI cout du planner porte un terme
d urgence (urgency_w=6.0) module par la faim, que mon proxy ignore. DETTE : brancher le vrai cout
analytique (command_planner) avant toute comparaison.

⭐ DECOUVERTE, non prevue et interessante : les candidats qui S ECARTENT du bosquet le plus proche
(om=-0,6 -> 1,34 repas) battent ceux qui foncent dessus (om=+0,6 -> 1,13). Sur 600 ticks avec 4
bosquets, **aller au plus proche n est PAS optimal** : l ENCHAINEMENT compte davantage. C est
exactement le signal recherche depuis le debut — un monde ou le glouton cesse d etre suffisant.
⚠️ Hypothese « c est l enchainement de bosquets » NON VERIFIEE ; hypothese concurrente (ecretage a
100) REFUTEE par mesure (correlation negative aussi forte chez les affames : -0,385).

PROCHAIN PAS : brancher le vrai cout analytique comme reference, puis seulement juger le gain.
