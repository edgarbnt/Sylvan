NETTOYAGE DES CONSTANTES — PRE-INSCRIT (2026-07-22, avant lancement)

Config de reference = celle validee le 2026-07-22 : cone 120, kin_turn 6.0, memoire ON,
4 bosquets x 2 baies, repousse 2500, mono-pulsion. 20 vies, graine 1.

DEUX CONSTANTES SUSPECTES, mesurees et non supposees :
 1. heading_weight = 2.0 -- ACTIF en mono-pulsion (branche plan_wm_slot, command_planner.py:580,
    reason emis l.592). Le projet l a RETIRE sur mesure (hw=0 >= hw=2, forage_ab_hw.sh) et les
    harnais single-drive vivants sont a 0.0. Mon harnais servait 2.0, copie d un harnais
    multi-drive ou il est INERTE. C est un echafaudage rallume par erreur.
 2. surv_turn_rate = 0.015 -- modele de virage. MESURE sur le corps promu : p99 = 0.0602, soit
    4x plus. Le planner croit qu un demi-tour coute 209 ticks alors qu il en coute 52. Sous un
    CONE c est doublement genant : tourner EST le moyen de regarder, et le cout le sur-facture.

CELLULES : (hw, turn) = (2.0, 0.015) reference | (0.0, 0.015) | (2.0, 0.060) | (0.0, 0.060)

CRITERES -- ils appliquent la DOCTRINE tranchee le 2026-07-22 (docs/doctrine_appris_vs_designe.md) :
on ne demande PAS la parite pour retirer un echafaudage, on demande qu il n y ait pas effondrement.
  ADOPTER  : repas >= reference - 0.3 -> on retire l echafaudage / on corrige la constante.
             (une baisse jusqu a 0.3 repas est ACCEPTEE comme prix de la purete)
  KILL     : repas <= reference - 0.8 -> la constante etait PORTEUSE (cas nominal_speed, -14.7 pts
             quand on avait insere la vraie valeur). On la GARDE et on la DECLARE fausse-mais-utile.
  ENTRE LES DEUX : non concluant, re-mesurer sur 2 graines avant de trancher.

CE QUE CE TEST NE DIT PAS : une graine, 20 vies. Verdict directionnel ; toute adoption sera
re-confirmee sur 2 graines avant d etre ecrite dans les defauts.
