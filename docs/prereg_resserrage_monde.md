RESSERRAGE DU MONDE — PRE-INSCRIT (2026-07-22, avant lancement)

POURQUOI. Dans la config validee, 95-100 % des episodes sont PLEINS et l entite fait ~4 repas pour
un besoin de 3,75. Consequence mesuree le jour meme : le nettoyage des constantes a rendu trois
intervalles de confiance contenant TOUS zero -- rien ne discrimine. C est le meme mur que celui qui
avait tue le critique appris (signal de valeur constant a 99,6 % : `time.clamp(max=cap)` fait
saturer tout candidat qui survit, donc la cible n a rien a classer).
Tant que la survie sature, ni le nettoyage des constantes ni un critique appris ne sont mesurables.

LEVIER CHOISI = LA REPOUSSE, et l arithmetique le designe :
  arriver a un bosquet avec 55 d energie -> manger 2 baies (+80, ecrete a 100) -> traverser 10 m
  (-45) -> arriver a 55. Le cycle est EXACTEMENT a l equilibre. Il ne tient que parce que la
  repousse (2500 ticks) remplit les bosquets a temps. La ralentir augmente la probabilite d arriver
  sur un bosquet VIDE = l echec pertinent pour la DECISION, et celui que la memoire doit eviter.
On ne touche NI au drain NI au restore NI a la vitesse : ce sont des proprietes du corps, et les
changer rendrait incomparables tous les chiffres du jour.

BALAYAGE : regrow dans {2500 (actuel), 4000, 6000, 8000}. Config par ailleurs identique a celle
validee (cone 120, kin_turn 6.0, memoire ON, 4 bosquets x 2 baies, mono-pulsion, seed 1, 12 vies).

CRITERES -- pre-enregistres :
  CALIBRE : 30 % <= episodes pleins <= 70 %  ET  vies au plancher de famine < 20 %
            ET ecart-type des durees de vie > 400 ticks (il FAUT de la dispersion, c est l objet)
  SATURE  : episodes pleins > 80 %  -> trop facile, la valeur ne discriminera pas
  MUR     : plancher >= 40 %  OU  pleins < 20 %  -> on mesurerait un mur, pas une decision
  On retient le regrow CALIBRE le PLUS FAIBLE (le moins de changement par rapport a l existant).

CE QUE CE TEST NE DIT PAS : une graine, 12 vies. Directionnel. Le reglage retenu sera reconfirme
sur 2 graines avant d etre ecrit comme defaut.

--- AMENDEMENT (2026-07-22) : LEVIER 1 REFUTE, et le calcul le disait ---
RESULTAT du balayage repousse {2500,4000,6000,8000} : 83-100 % d episodes pleins, ecart-type des
durees de vie 0-171 ticks, ~4 repas partout. AUCUNE cellule calibree. Meme a 8000 (au plus une
repousse par vie), l entite survit a 83 %.

CAUSE, calculable AVANT le run et que je n avais pas faite : besoin d une vie = 3,75 repas, stock
INITIAL = 4 bosquets x 2 baies = 8 baies, soit 2,1x le besoin. L entite n a donc JAMAIS besoin qu un
bosquet repousse : elle visite des bosquets neufs et la vie s arrete avant epuisement. La repousse
etait DECORATIVE. Negatif banke : ne pas re-tenter la repousse comme levier tant que le stock
initial depasse le besoin.

LEVIER CORRIGE = LE STOCK INITIAL. Condition pour que revisiter (donc le timing, donc la memoire)
soit obligatoire : stock initial < 3,75 baies.
BALAYAGE 2 : berries dans {4, 3, 2} a repousse 2500, 4 bosquets, reste identique.
Effet secondaire recherche : avec moins de baies que de bosquets, certains bosquets sont VIDES des
le depart tout en gardant leur buisson-marqueur visible -- c est exactement l aliasing qu on veut.
CRITERES INCHANGES (30-70 % pleins, plancher < 20 %, ecart-type > 400).

--- RESULTAT DU BALAYAGE 2 (stock initial) + CORRECTION D UN CHIFFRE FAUX ---
| baies | pleins | plancher | ecart-type | repas | verdict |
|   8   |  92 %  |   0 %    |     33     | 3,67  | SATURE  |
|   4   |  92 %  |   0 %    |     83     | 2,17  | SATURE  |
|   3   |  92 %  |   0 %    |     55     | 2,00  | SATURE  |
|   2   |  33 %  |  17 %    |    348     | 1,17  | entre-deux |

🚨 CHIFFRE FAUX QUE J AI REPETE TOUTE LA SESSION : « besoin metabolique = 3,75 repas ». FAUX.
J avais divise le drain TOTAL (3000 x 0,05 = 150 points) par le restore (40) en OUBLIANT le
RESERVOIR INITIAL. L entite demarre a 100 : il ne lui manque que 50 points = 1,25 REPAS.
Verification sur les donnees, exacte : a B=3, deux repas donnent 100 + 80 - 150 = 30 points
restants -> elle finit l episode (92 % pleins, mesure). A B=2, 1,17 repas donne -3 -> elle meurt
juste avant (survie mediane 2800, mesuree).
A RETIRER EXPLICITEMENT : le commentaire « le bras ON fait 3,80 repas pour un besoin de 3,75, il
mange exactement ce qu il faut » (§13) etait une COINCIDENCE, pas une confirmation.

VERDICT : B=2 est le seul regime qui discrimine (durees de vie
[2000, 2000, 2420, 2610, 2730, 2800, 2800, 2800, 3000, 3000, 3000, 3000] = vraie dispersion),
mais il ECHOUE le critere d ecart-type : 348 contre 400 exige. Barre NON deplacee malgre les
52 ticks d ecart et malgre une distribution visiblement dispersee.
=> non concluant par pre-inscription. A re-mesurer sur 2 graines et 20 vies avant d etre adopte.

--- CONFIRMATION B=2 (2 graines x 20 vies) + CRITERE DECLARE MAL POSE ---
                pleins   plancher   ecart-type   repas
  graine 1       50 %      10 %        306        1,40
  graine 2       60 %      20 %        395        1,40
  POOLE n=40     55 %      15 %        354        1,40

VERDICT PRE-INSCRIT : NON CALIBRE (2 criteres sur 3 ; l ecart-type echoue, 354 < 400).
LA BARRE N EST PAS DEPLACEE ET LE RUN N EST PAS REFAIT.

🚨 MAIS LE CRITERE ETAIT MATHEMATIQUEMENT MAL POSE, et c est une propriete de ma pre-inscription,
pas une re-lecture du resultat. La duree de vie est BORNEE : plancher de famine 2000, plafond
d episode 3000. Sur [2000, 3000] :
    dispersion UNIFORME (ce qu on cherche)      -> ecart-type 289
    bimodale 50/50 aux extremes                 -> ecart-type 500  = MAXIMUM ABSOLU
    mesure                                      -> ecart-type 354
Exiger > 400 revient donc a exiger du TOUT-OU-RIEN, pas de la dispersion. Une distribution
parfaitement etalee ECHOUE ce critere par construction. Les 354 mesures sont DEJA plus disperses
qu une uniforme.

CRITERE BIEN POSE, declare ici pour les pre-inscriptions FUTURES (jamais applique retroactivement
a ce run) : fraction de vies NI au plancher NI au plafond >= 25 %. Mesure ici : 30 % (12/40).

DECISION LAISSEE A L OWNER : adopter B=2 (55 % pleins, 15 % plancher, 30 % de traine
intermediaire, seul regime non sature du balayage) est un JUGEMENT apres une pre-inscription
ECHOUEE -- a ne jamais relire comme « B=2 a passe son gate ». Ne PAS relancer avec le critere
corrige : le choisir en connaissant deja la reponse serait de la ceremonie, pas de la rigueur.
