A/B REPLAN-EVERY — PRE-INSCRIT (2026-07-23, avant lancement)

POURQUOI. Le G0 conséquence (diag_consequence_g0.py) mesure qu a replan=10 la variance intra-état
vaut 1,9 % : le planner replanifie si souvent qu il EFFACE sa propre decision avant qu elle ait une
consequence. Aucun critique appris ne peut classer des candidats qui menent tous au meme endroit.
En simulation, replan 60-120 fait passer la variance a 12,7 % / 38,9 % SANS degrader la faim
moyenne (77,5 -> 78,6) ; a 300 la faim s effondre a 58,2 (fausse solution, refusee).

CE QU ON TESTE ICI, ET CE QU ON NE TESTE PAS.
On teste : l entite SURVIT-ELLE a un planner moins reactif ? Ce n est PAS un test d amelioration.
L objectif n est pas plus de repas, c est de verifier que le regime « decisions consequentes » est
VIABLE en vies, pas seulement en simulation avec un relais glouton.

CONFIG : preset bosquets_v2 (le monde gele, corps NON modifie -- l ablation du 2026-07-22 a prouve
que kin_turn est indifferent : 1,40 repas a 1.5 comme a 6.0). Memoire ON. 2 graines x 20 vies/bras.
BRAS : replan-every = 10 (reference) | 60 | 120.

METRIQUE PRIMAIRE = repas MOYENS par vie, poolee sur les 2 graines (la survie sature, la mediane
d un petit compte entier saute d une unite -- deux pieges deja payes).

CRITERES, appliquant la doctrine du 2026-07-22 (purete/consequence sans effondrement) :
  VIABLE   : repas(bras) >= repas(10) - 0.3  -> le regime consequent tient, on peut rouvrir le
             critique appris dessus. On retiendra le replan le PLUS GRAND qui reste viable.
  KILL     : repas(bras) <= repas(10) - 0.8  -> allonger le replan casse l entite ; la consequence
             ne s obtient pas par ce levier, et il faudra revenir au corps (cher).
  ENTRE-DEUX : non concluant, re-mesurer avant de trancher.
CONTROLE : si un bras a >= 40 % de vies au plancher de famine (2000 pas), verdict NUL (monde casse).

CE QUE CE TEST NE DIT PAS : il ne dit rien de la qualite d un critique appris. Il ouvre ou ferme la
porte, il ne la franchit pas.

--- RESULTAT (2026-07-23) : replan=60 est GRATUIT, replan=120 coute ---
Preset bosquets_v2, memoire ON, 2 graines x 20 vies. Application VERIFIEE dans les corpus :
1 replan tous les 10 / 60 / 120 ticks, conforme dans les trois bras.

| replan | repas moy | pleins | plancher | survie med | ecart vs 10 | IC95 |
|   10   |   1,40    |  50 %  |   10 %   |    2900    |      —      |   —  |
|   60   |   1,40    |  50 %  |   10 %   |    2900    |   **+0,00** | [-0,28, +0,30] |
|  120   |   1,12    |  35 %  |   22 %   |    2800    |     -0,27   | [-0,58, +0,03] |

VERDICT : les deux sont formellement VIABLES (>= -0,3). Mais replan=120 FROLE la barre (-0,27) et
son plancher de famine DOUBLE (10 % -> 22 %), les pleins tombent de 50 % a 35 %. La degradation est
reelle meme si le critere passe.

⇒ **RETENU : replan-every = 60.** Il est GRATUIT (ecart exactement 0,00, tout identique : pleins,
plancher, survie mediane) et le G0 conséquence lui donne deja 12,7 % de variance intra, au-dessus
de la barre de 10 %. C'est le choix dominant : toute la conséquence necessaire, aucun cout mesure.
On ne retient PAS le plus grand viable comme la pre-inscription le prevoyait — la regle disait
« le plus grand qui reste viable », mais elle n avait pas anticipe qu un bras puisse etre viable
sur la primaire tout en doublant la mortalite. Ecart a la pre-inscription DECLARE, pas masque.

VERIFICATION anti-artefact : les runs sont bien distincts (8/20 vies de duree differente entre
rp=10 et rp=60) — le +0,00 est un vrai resultat, pas deux fois le meme run.

FAIT NOTABLE : multiplier par SIX le temps entre deux decisions ne change RIEN a la performance.
Le planner replanifiait donc six fois plus souvent que necessaire. C est coherent avec le G0 :
a rp=10 la decision n engageait rien, donc la reprendre si souvent n apportait rien non plus.

PROCHAIN PAS : le critique appris redevient testable, sur bosquets_v2 + replan=60, avec la cible
RESIDU (repas dans la fenetre, bimodale, validee) et la metrique de RANG INTRA-ETAT (jamais le R2
poole). Le gate hors-ligne reste a ecrire.
