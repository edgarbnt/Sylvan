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
