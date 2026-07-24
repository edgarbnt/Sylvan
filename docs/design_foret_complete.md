# LA FORÊT — document de référence complet

**Statut** : pré-inscription. Aucun run n'a été lancé sur le monde forêt. Tout ce qui suit est soit
une **mesure déjà payée**, soit une **hypothèse explicitement étiquetée comme telle**.
**Date** : 2026-07-24. Remplace/complète `design_foret_bases_et_limites.md` (qui reste la version courte).

---

## 0. LE PROBLÈME, DANS LES MOTS DE L'OWNER

> « une difficulté trop plate ou trop linéaire, qui empêche l'apprentissage »

C'est exactement ce que la journée du 2026-07-24 a mesuré, et le diagnostic se scinde en deux :

**PLAT** = une décision ne change rien. Mesuré : au régime vécu, forcer le PIRE choix ne modifie
l'avenir que dans 17 % des cas (33 % avec le levier périssable) — le planner replanifie et rattrape.

**LINÉAIRE** = ce qui compte est *dérivable*. Mesuré sur cinq leviers : dès que la relation entre ce
qu'on perçoit et ce que ça vaut a une forme close, **une formule ajustée égale ou bat un réseau**.

| levier testé | part de la marge oracle captée | verdict |
|---|---|---|
| plus proche (`-min_dist`, l'existant) | 28 % | référence |
| prédiction / interception | — | **forme close** : « vise devant » s'écrit |
| hétérogénéité (choix entre proies) | formule 64,3 % / MLP 64,0 % | **redondant** |
| **arbitraire (valeur par type)** | **formule 49,5 % / appris 69,7 %** | ✅ **seul cas où apprendre ≠ calculer** |

**La forêt doit donc produire une difficulté qui n'est ni plate ni linéaire.** Ce document liste ce
qu'on y met pour ça, ce qu'on s'interdit, et comment on le vérifiera.

---

## 1. LE CRITÈRE (dérivé des mesures, pas du goût)

Toute mécanique candidate doit passer **quatre questions**. C'est le filtre qui a tué quatre idées
séduisantes pour zéro run :

1. **Est-elle PERCEPTIBLE ?** — si l'information n'atteint pas la rétine, rien ne peut l'apprendre.
2. **Survit-elle à l'ENCODEUR ?** — piège mesuré : le type est lisible à 82,9 % dans la rétine brute
   et 29,5 % après l'encodeur. Perceptible ≠ représenté.
3. **Est-elle NON-DÉRIVABLE ?** — s'il existe une formule sur des grandeurs physiques perceptibles,
   la formule gagnera. Test gratuit disponible : `diag_choice_headroom.py` / `diag_arbitrary_headroom.py`.
4. **Change-t-elle l'ISSUE ?** — contre-exemple mesuré : la maturité est bien dans le latent (R² 0,65)
   et n'améliore la prédiction du retour **en rien** (ablation −0,030).

Une mécanique qui échoue à l'une des quatre est de la décoration.

---

## 2. CE QU'ON VEUT DANS LA FORÊT

### 2.1 Le substrat visuel — et la règle qui le gouverne

**Assets** : `ForestLowPolyAssets/` (KayKit Forest Nature Pack, **CC0**, 211 fichiers glTF : arbres,
rochers, buissons, herbes). Style cohérent avec le loup (Quaternius, CC0). On garde le **low-poly** :
on simule des millions de ticks, le coût de rendu compte réellement.

**La variété doit venir des INSTANCES, pas du nombre de modèles** — taille, teinte, densité, rotation.
C'est la variation qui nourrit l'encodeur ; dix modèles identiques n'apprennent rien de plus qu'un.

🚨 **RÈGLE ABSOLUE : LE VISUEL NE DOIT PAS MENTIR.** Aujourd'hui `forest_manager.gd` est explicitement
« VISUAL-ONLY [...] NO collision, NO physics » et n'existe **qu'en mode visuel** : la forêt est
invisible à l'agent et absente de l'entraînement. Tout élément dessiné doit avoir sa contrepartie
perceptible (couche rétine bit 7 + `retina_color`) et, s'il bloque, sa collision (bit 2).

### 2.2 Structure spatiale — une forêt n'est pas un tirage uniforme

*(codé ce soir dans `forest_solid.gd`, opt-in, NON VÉRIFIÉ)*

- **Peuplements** — processus de **Neyman-Scott/Thomas** (standard en écologie spatiale) : des centres,
  puis des arbres dispersés autour selon une gaussienne. Les graines tombent près du parent.
- **Espacement minimal** — les arbres se concurrencent pour la lumière et l'eau ; ils ne se collent pas.
- **Clairières** — disques d'exclusion, comme les trouées laissées par un arbre tombé.
- **Sous-bois guidé par la lumière** *(à faire)* — les buissons poussent **là où la canopée est ouverte**,
  donc en bordure de clairière. C'est écologiquement vrai **et** ça crée le « couvert au bord de
  l'ouvert », la configuration exacte qui rend l'affût possible.
- **Vérification** : indice de **Clark-Evans** = (distance moyenne au plus proche voisin) / (attendue
  sous Poisson). < 1 = groupé, = 1 = aléatoire, > 1 = régulier. Loggé, pour **prouver** l'arrangement
  au lieu de l'affirmer.

**Pourquoi ça compte au-delà du réalisme** : une forêt structurée produit une occlusion **NON
UNIFORME** — couloirs de visibilité, écrans denses, ouvertures. C'est ce qui fait qu'**une position
vaut mieux qu'une autre**. Un semis uniforme n'est qu'un brouillard homogène où toutes les positions
se valent : encore une difficulté plate.

### 2.3 Dynamique — pour que le world-model ait enfin quelque chose à modéliser

**Le constat qui l'impose** : le déplacement prédit par le WM est reconstructible à **R² 0,985 depuis
la commande SEULE**. Son prédicteur n'a jamais appris autre chose que sa propre cinématique, parce que
le monde était immobile et le corps parfaitement obéissant.

- **Proies mobiles** *(construit, spec mesurée)* — elles **vaquent** et **ne fuient pas** : une proie
  qui fuit converge vers une trajectoire radiale, contre laquelle poursuite et interception coïncident
  *par construction* (gain nul, mesuré). Vitesse ≥ 0,9× celle de l'agent, sinon aucun gain.
- **Terrain qui ralentit** *(hypothèse)* — sous-bois dense contre sentier dégagé. C'est le **fix direct**
  du R² 0,985 : le déplacement dépendrait enfin de **où l'on est**, pas seulement de ce qu'on commande.
- **Occlusion** — les troncs cachent. Crée le hors-vue, donc rend la mémoire *load-bearing* au lieu de
  décorative. (Chantier mémoire gelé faute d'occlusion réelle : c'est ici qu'il se débloque.)

### 2.4 Perception active — le regard

**La seule mécanique identifiée dont la valeur d'action est PUREMENT INFORMATIONNELLE.** Regarder ne
rapproche de rien, ne rapporte aucun repas : ça réduit une incertitude. Aucune formule de distance ne
peut même l'*exprimer* — alors qu'elle capturait les quatre leviers tués aujourd'hui.

- **Tête mobile** : le regard se commande indépendamment du déplacement. Indispensable à l'affût
  (avancer vers un couvert *en surveillant* la proie est géométriquement impossible si regard = cap).
- ⚠️ **On donne la CAPACITÉ, jamais le COMPORTEMENT.** Un balayage automatique codé-main serait le
  raccourci que l'owner a justement redouté. L'agent doit apprendre *quand* balayer et *quoi* fixer.
- **Coût** : ajoute une dimension à la proprioception ⇒ **exige un retrain du WM**, et ajoute une
  dimension à la recherche du planner (vx, ω, regard). ⇒ **à décider AVANT la collecte.**

### 2.5 Arbitraire — la seule condition prouvée où apprendre ≠ calculer

- **Types de proies à valeur arbitraire** *(construit)* — rien dans la physique perceptible ne prédit
  la valeur ; il faut avoir goûté. Mesuré : formule 49,5 % contre appris 69,7 % de la marge oracle.
- ⚠️ **BLOCAGE MESURÉ** : l'encodeur actuel **ne perçoit pas** ces types (29,5 % contre 44,2 % de
  majorité). Trois explications testées et réfutées : le canal (teinte ET luminosité détruites), la
  taille (baie 3,44 rayons vs buisson 3,17), la difficulté de la sonde (même la moyenne agrégée est
  illisible, R² −0,659, quand la même mesure sur le buisson donne +0,650).
  **Cause retenue** : la couleur de la nourriture était CONSTANTE pendant l'entraînement du WM.
  ⇒ **C'est ce qui justifie le retrain**, et c'est le premier que la mesure fonde vraiment.
- **Vulnérabilité individuelle** *(hypothèse)* — une proie jeune/blessée est plus lente ou moins
  vigilante. Lien apparence→attrapabilité arbitraire, donc apprenable seulement par l'expérience.

### 2.6 Interactions — inspirées de l'éthologie réelle

**L'affût et le couvert** — le résultat le plus intéressant de la recherche : *« les loups choisissent
leurs sites d'embuscade pour contrer les capacités sensorielles de leurs proies »*
([Behavioral Ecology](https://academic.oup.com/beheco/article/32/2/339/6125068)). La valeur d'une
position dépend de la ligne de vue **depuis le point de vue de la proie**. Aucune formule sur la
distance ne le capture : il faut raisonner sur l'occlusion *et* sur ce que l'autre perçoit.
C'est le meilleur candidat identifié, et il combine occlusion + regard + proie mobile.

**Une proie qui détecte et fuit** — l'approche devient une approche *sous détection*. Combinée au
couvert, elle produit la traque : s'approcher lentement, à couvert, plutôt que foncer.
⚠️ Attention au piège mesuré : la fuite doit être **déclenchée par la détection**, pas permanente,
sinon on retombe sur la trajectoire radiale dégénérée.

### 2.7 Interactions issues de la recherche — candidates classées par le filtre §1

**(a) Distance de fuite (*flight initiation distance*)** — concept MESURÉ en éthologie : la distance à
laquelle une proie s'enfuit varie avec le couvert, la vitesse d'approche, son état. Donne une base
empirique à l'affût au lieu d'un réglage arbitraire. La valeur d'une position dépend de ce que
**l'autre** perçoit ⇒ non-dérivable d'une distance. *(hypothèse à gater)*

**(b) Effet « many-eyes » / taille de groupe** — un groupe plus grand détecte le prédateur PLUS TÔT
mais offre PLUS de nourriture. La littérature est explicitement **contradictoire** (dilution contre
vigilance collective, méta-analyses divergentes) : un arbitrage que les écologues ne réduisent pas à
une formule est un bon candidat pour nous. *(hypothèse à gater)*

**(c) ⭐ QUALITÉ D'UN SITE INCONNUE — la meilleure trouvaille.** Le théorème de la valeur marginale
(Charnov) donne quand quitter un site : c'est une FORMULE, donc mort-né selon notre filtre. **MAIS**
la littérature 2024 est explicite : *« le MVT n'est valide que dans des environnements déterministes
dont les statistiques sont CONNUES du fourrageur ; les environnements naturels remplissent rarement
ces conditions »*. Sous incertitude il faut **estimer la qualité depuis sa propre expérience récente**
(mise à jour bayésienne).
Coche les quatre cases : perceptible, **non-dérivable** (aucune formule — il faut un estimé construit
sur le vécu), change l'issue (partir trop tôt/tard coûte des repas). **Et exige de la MÉMOIRE** ⇒
débloquerait le chantier mémoire, gelé faute de raison mesurée d'exister.
C'est une saveur d'arbitraire DIFFÉRENTE des types : intégrer dans le TEMPS, pas reconnaître une
apparence. *(hypothèse, la plus prometteuse)*

**(d) Le mouvement de la proie comme INDICE** — en écologie du mouvement, un animal qui trouve un bon
site ralentit et tourne davantage (*area-restricted search*). Le comportement de la proie **révèle
une information cachée sur le monde** : une proie qui s'attarde signale un site riche. C'est de la
perception-par-conséquence appliquée à un AUTRE AGENT, et ça rend le mouvement des proies informatif
au lieu d'être un bruit à intercepter. *(hypothèse à gater)*

⇒ **Les deux plus prometteuses : (c) qualité inconnue** — elle amène la mémoire — **et (a) distance de
fuite avec couvert** — elle amène l'affût et la perception active. Complémentaires : l'une pousse à
intégrer dans le temps, l'autre à raisonner sur ce que l'autre voit.

---

## 2bis. POURQUOI L'AGENT CONFOND UN TRONC BRUN ET UNE BAIE (question owner, 2026-07-24)

Ce n'est PAS une limite de l'œil : la rétine reçoit bien des RGB distincts. **C'est le détecteur qui
est grossier, et il est codé à la main.** Le slot normalise la couleur du rayon, calcule son cosinus
avec la requête `(1,0,0)` et déclenche au-dessus de **0,55**. Or un brun `(0.36,0.25,0.15)` normalisé
donne un cosinus de **0,776** — largement au-dessus. Le tronc EST de la nourriture, pour le slot.

**Et pourquoi il n'apprend pas de sa déception ?** Parce qu'il n'existe **aucun chemin d'apprentissage
entre l'expérience et la perception** : sur la config servie, le slot a **zéro paramètre appris** (le
scoreur, 2498 paramètres, est calculé puis intégralement écrasé par la branche géométrique). L'agent
peut mordre mille troncs, ça ne changera jamais ce qu'il considère comme de la nourriture. Sa
perception est **gelée par construction**.

Deux verrous INDÉPENDANTS, donc :
- **A2** — détecteur codé-main à seuil trop large (le tronc brun) ;
- **A1** — encodeur aveugle aux variations d'apparence de la nourriture (29,5 % contre 44,2 %).

⇒ **Conséquence pour le chantier** : le WM « typé » (`wm_objcentric_kin_typed`) a des requêtes
APPRISES — mesurées `[0.876, 0.349, 0.333]` au lieu des primaires exactes — mais **n'est pas promu**.
Si on ré-entraîne de toute façon, il faut régler les DEUX ensemble, sinon le problème du tronc brun
**survivra au retrain**.

### 2.8 Faire varier les couleurs dans la COLLECTE — et le piège qui peut le faire rater

C'est le levier qui fait tomber le verrou **A1**. Mais « mettre plusieurs couleurs » ne suffit pas :

⚠️ **JEPA jette délibérément ce qui est IMPRÉVISIBLE.** Si la couleur d'une baie est retirée au hasard
à chaque repousse, elle est du bruit du point de vue du prédicteur, et l'encodeur a *raison* de
l'écraser. On reproduirait l'échec en croyant l'avoir corrigé.

**Règle de conception qui en découle** : une couleur doit être **STABLE pour un objet donné**
(prévisible dans le temps — l'objet garde son apparence) et **VARIER ENTRE objets et entre épisodes**
(donc informative). C'est cette combinaison — constante par objet, variable dans la population — qui
force l'encodeur à allouer de la capacité à la dimension « apparence ».

**Et il faut faire varier TOUT, pas seulement la nourriture** : troncs, buissons, rochers, sol. Sinon
on répare la cécité pour une classe d'objets et on la recrée pour la suivante. C'est la lecture
correcte du §3 : enrichir le substrat **une fois**.

⚠️ **Le verrou A2 ne tombera PAS tout seul.** Le retrain corrige l'encodeur ; il ne touche pas au
détecteur codé-main (seuil cosinus 0,55) qui fait qu'un tronc brun EST de la nourriture. Il faut
promouvoir les **requêtes apprises** (`wm_objcentric_kin_typed`) dans le même mouvement, sinon le
problème du tronc brun survit au retrain.

### 2.9 Les animaux — codés à la main, et c'est honnête

Ils font partie du **MONDE**, pas de l'agent. Des règles simples suffisent, à condition de ne jamais
prétendre qu'ils sont intelligents.

- **Proies comestibles** *(construit)* — vaquent, ne fuient pas (la fuite est dégénérée, mesuré).
- **⭐ Animaux NON comestibles (distracteurs)** *(hypothèse, sous-estimée)* — oiseaux, écureuils : des
  choses qui BOUGENT et qu'on ne peut PAS manger. Sans eux, « ça bouge donc c'est de la nourriture »
  est un raccourci gratuit, et l'agent n'a jamais à discriminer. Avec eux, le lien apparence→comestible
  doit être **appris** — c'est la même famille que les types arbitraires, qui est la seule à avoir
  passé le filtre. Coût faible, valeur élevée.
- **Un concurrent** *(hypothèse)* — un autre prédateur qui mange les mêmes proies : la valeur d'une
  ressource dépend alors du comportement d'un autre, donc de quelque chose qu'il faut modéliser.
- **Une menace** *(existe déjà)* — `hazard_manager.gd` est en place ; à raccorder plutôt qu'à réécrire.

### 2.10 Ce à quoi on n'avait pas pensé

- **⭐ Un abri / une tanière** *(hypothèse)* — un lieu FIXE qui restaure ou protège. Ça crée le
  *central place foraging*, concept écologique réel (et la littérature loup le mentionne
  explicitement) : jusqu'où s'éloigner de son point d'ancrage ? L'arbitrage n'a pas de forme close dès
  que la qualité des sites est incertaine, et il donne un **sens spatial à la mémoire**.
- **⭐ Jour / nuit FONCTIONNEL** *(le cycle existe déjà, mais VISUEL seulement)* — la nuit réduit la
  portée de vision. Couplage direct avec la mémoire : ce qu'on a vu de jour doit être retenu pour la
  nuit. Très bon rapport valeur/coût, puisque la moitié est déjà écrite.
- **Fatigue** *(hypothèse)* — courir vite coûte plus et impose de récupérer. Rend la VITESSE une
  décision au lieu d'un paramètre, et donne au prédicteur une dynamique interne à modéliser.
- **Météo** — écarté pour l'instant : coût élevé, gain non mesuré.

### 2.11 Le corps — quelles modifications ajoutent une DÉCISION

À distinguer nettement de l'hygiène technique.

**Qui ajoutent une décision :**
- **Le regard indépendant** (cf. §2.4) — la seule action à valeur purement informationnelle.
- **⭐ S'abaisser / se tapir** *(hypothèse, excellente)* — réduit la visibilité pour les proies (elles
  détectent plus tard) mais ralentit. C'est un arbitrage sans forme close, et surtout c'est une action
  qui **manipule ce que les autres perçoivent**. C'est rare et précieux : ça rend la distance de fuite
  (§2.7a) actionnable au lieu d'être subie.
- **Vitesse couplée à la fatigue** — voir §2.10.

**Hygiène technique (pas un levier d'apprentissage, mais nécessaire) :**
- **Hitbox** — le corps est un assemblage cinématique ; en forêt dense, la qualité de la collision
  décide de la navigabilité. À traiter comme de l'ingénierie, en le mesurant (pénétrations = 0), sans
  le compter comme un gain d'apprentissage.

### 2.12 L'EAU — état réel, et à quelle condition elle vaut le coup

**Constat honnête : on n'avait RIEN décidé.** Les presets bosquets n'ont **aucune clé** eau/soif, et
la soif est mesurée **constante à 0** dans le corpus. L'eau a disparu quand le projet est passé aux
bosquets. La machinerie existe pourtant et dort : slot-2 (requête bleue), coût multi-drive,
`wm_objcentric_s2` promu.

⚠️ **La remettre TELLE QUELLE serait « linéaire »** : l'arbitrage faim/soif est déjà tranché par un
coût analytique pondéré par l'urgence, et le **critique d'arbitrage appris a ÉCHOUÉ au G3** — exactement
comme les quatre leviers du 2026-07-24. Une deuxième pulsion n'ajoute pas de difficulté non-dérivable,
elle ajoute une dimension à une formule qui la gère déjà. (Historique : le multi-drive était « LE MUR »,
l'agent mourait de faim campé sur l'eau.)

**Décision (owner, 2026-07-24)** : remettre l'eau sous forme de **FLAQUES** — plusieurs points d'eau
dispersés, à **retenir** (donc utiles à la mémoire), et dont la disponibilité **varie**. C'est la
variabilité, pas la deuxième pulsion, qui porte la valeur d'apprentissage : on retombe alors sur la
famille « qualité inconnue » (§2.7c), la seule avec l'arbitraire à avoir passé le filtre.

### 2.12bis ⭐ RÈGLE GÉNÉRALE SUR L'INCERTITUDE (vaut pour TOUT ce qu'on ajoutera)

> **L'incertitude doit être OBSERVABLE et GRADUELLE, jamais instantanée et cachée.**

Preuve déjà payée : la **relocalisation aléatoire** des baies crée bien de la conséquence (33 %) mais
le WM **ne peut pas la représenter** — le transport du slot suppose l'objet immobile, et un WM
déterministe prédit une *espérance* là où il faudrait des futurs énumérables (MoP-JEPA, anomalie A4).
On a donc fabriqué une difficulté que le modèle est structurellement incapable d'anticiper.

- ✅ **BON format** : une flaque qui **rétrécit visiblement**, une baie qui **se ternit** en vieillissant.
  Le WM peut l'encoder (progressif, perceptible) et l'agent doit quand même se souvenir et anticiper.
- ❌ **MAUVAIS format** : un saut aléatoire, une disparition instantanée, une valeur retirée au hasard.

⚠️ Même piège pour les COULEURS (§2.8) : stable par objet, variable dans la population. Le principe
est le même — **prévisible localement, informatif globalement**.

### 2.13 VITESSE ET DURÉE DE VIE — la calibration (chiffres mesurés)

| grandeur | valeur MESURÉE |
|---|---|
| vitesse du corps | **0,011 m/tick = 0,79 km/h** |
| traverser l'arène (22 m) | 2000 ticks = **67 % d'une vie entière** |
| éventail de vitesse offert au planner | `vx_grid = (0.55, 0.65, 0.75)` → bande de **±15 %** |
| événements par vie | **1 à 2 repas** |
| loup réel, trot / sprint | ~8-10 / ~50-60 km/h → **12× / 70×** notre vitesse |

**Deux problèmes distincts sont visibles là-dedans.**

**(a) Le choix de vitesse est un choix de FAÇADE.** Une bande de ±15 % ne peut pas porter de décision.
**Décision (owner)** : ouvrir un **éventail LARGE** — marcher / trotter / sprinter, comme un vrai loup.
Couplé à un **coût énergétique croissant**, sprinter devient un **pari** : dépenser de l'énergie contre
une chance d'attraper. Quand attaquer dépend alors de la distance, de l'énergie restante, de la vitesse
de la proie et d'une **probabilité de réussite incertaine** ⇒ encore la famille « qualité inconnue ».
Bonus : la portée d'imagination devient VARIABLE — au sprint, les mêmes 80 ticks de rêve couvrent bien
plus de terrain, donc la prévoyance s'étend automatiquement quand l'agent va vite.

**(b) La bonne métrique n'est pas le temps, c'est le nombre d'ÉVÉNEMENTS par vie.** À 1-2 repas, il n'y
a aucune place pour « j'investis maintenant, ça rapporte plus tard » : ni la mémoire, ni le choix de
site, ni la tanière n'ont de sens à cette échelle. C'est aussi la cause directe de la famine de données
(25 repas sur 20 vies). **Cible : 10 à 30 événements par vie.**
Deux leviers, de coûts très différents : allonger les épisodes coûte du calcul **linéairement**
(3000 → 12000 ticks = 4× la collecte) ; élargir la vitesse est **gratuit** et règle en plus la portée
et la taille du monde. ⇒ **priorité à la vitesse**, allonger les épisodes seulement en complément.

**⚠️ Note sur « accélérer la collecte »** : ce n'est PAS la même question. Augmenter le pas de
simulation dégraderait la résolution physique et changerait ce que le WM apprend. La bonne façon
d'accélérer une collecte est le **parallélisme** (plusieurs Godot), que le projet sait déjà faire.

---

## 2ter. ⛔ À DÉCIDER AVANT LA COLLECTE (liste bloquante)

Tout ce qui change **ce que le WM apprend** doit entrer dans la MÊME collecte. Décider après = un
second retrain, c'est-à-dire l'architecture axée-ressource que le §3 interdit.

1. **Le regard indépendant** — ajoute une dimension de proprioception.
2. **L'éventail de vitesse** — change la dynamique du corps.
3. **Le tapi** — change la dynamique ET la signature perceptive de l'agent.
4. **Les couleurs variables** (tout, pas seulement la nourriture) — c'est le fix du verrou A1.
5. **Les objets mobiles** — pour que le prédicteur apprenne une dynamique externe.
6. **L'occlusion** — sinon aucune mémoire ne pourra s'y brancher plus tard.
7. **La taille de l'arène** — elle dépend de (2), donc à trancher ensemble.

---

## 3. LIMITES MESURÉES — non négociables

| contrainte | mesure | conséquence |
|---|---|---|
| **couleur des troncs** | brun → fuite **0,2271** sur la requête rouge ; vert foncé → **0,0000** | un tronc brun est perceptuellement *rougeâtre* : le réalisme naïf est ici une ERREUR |
| **densité** | erreur du slot 0,00 m (0 %) → 0,29 m (30 %) → **1,43 m (60 %)** | rayon de capture 1 m ⇒ forêt dense = approche **impossible**. Cible ≈ 30 % |
| **navigabilité** | 45 arbres = fenêtre navigable ; **54 → immobile 85 % du temps** | plafond dur de densité |
| **budget de déplacement** | 0,011 m/tick × 3000 = **~33 m par vie** | agrandir l'arène a une limite dure ; compenser par vitesse/durée/densité, **à mesurer** |
| **rétine** | 36 rayons, portée **10 m**, seuil couleur 0,55, requêtes rouge/bleu | ne pas y toucher ; 0,55 < 1/√3 rend certains critères insatisfiables |
| **encodeur** | ne représente que ce qui a **VARIÉ à l'entraînement** | la collecte doit contenir TOUTES les variations qu'on voudra un jour |
| **JEPA** | jette *délibérément* l'imprévisible | un type tiré au hasard pourrait être écrasé **malgré** la variation — réserve honnête |

---

## 4. CE QU'ON S'INTERDIT (et pourquoi)

- **Décorer sans percevoir** — tout élément dessiné doit être perceptible. Sinon on refabrique le
  mensonge visuel actuel.
- **Coder un comportement plutôt qu'une capacité** — pas de balayage automatique, pas de « fuit
  toujours », pas de règle d'affût écrite à la main.
- **Ré-entraîner le WM pour une ressource** (§3) — le retrain est légitime *parce qu'il répare une
  cécité générale* (l'encodeur n'encode pas l'apparence), pas parce que « la bouffe compte ».
- **Ajouter une mécanique non testée par le filtre §1** — quatre idées séduisantes sont mortes
  aujourd'hui pour zéro run ; c'est la discipline qui a le mieux payé.
- **Différé, pas rejeté** : chasse en meute (plusieurs agents), marquage olfactif (nouveau **sens** :
  retrain légitime mais change les dimensions d'observation, à synchroniser dans plusieurs fichiers),
  saisons, cycle jour/nuit fonctionnel (il existe déjà, mais visuel seulement).

---

## 5. ORDRE DE CONSTRUCTION

1. **Forêt structurée** — porte l'occlusion, la variation d'apparence et le couvert : trois besoins
   mesurés d'un coup. Vérifier Clark-Evans **avant** d'aller plus loin.
2. **Décider le regard** — parce que ça change la proprioception, donc la collecte.
3. **Calibrer la taille** — analytiquement d'abord (budget 33 m), puis mesurer la navigabilité.
4. **Terrain + proies mobiles + types** — toutes les variations dans la MÊME collecte.
5. **UNE collecte, UN retrain.**
6. **Puis seulement** : l'affût, le critique, la mémoire.

---

## 6. GATES PRÉ-ENREGISTRÉS (aucun n'est passé)

| gate | critère | statut |
|---|---|---|
| arrangement | Clark-Evans **< 1** en peuplements, **≈ 1** en uniforme | à mesurer |
| navigabilité | survie non effondrée à densité cible | à mesurer |
| perception du type | encodeur **~30 % → > 70 %** après retrain | à mesurer |
| dynamique | le prédicteur bat « l'objet reste immobile » | sonde à écrire |
| non-régression WM | éval open-loop (position, yaw, eff_rank) ≥ actuel | à mesurer |
| difficulté non-plate | taux de conséquence nettement > 33 % | à mesurer |
| difficulté non-linéaire | appris > meilleure formule ajustée | à mesurer |

Les deux derniers sont **les gates du problème de l'owner**. Tout le reste n'est qu'un moyen.

---

## 6bis. CONTRAT D'INSTRUMENTATION — sondes et logs partout

Demande explicite de l'owner, et le projet en a déjà payé le prix : *« trois fois un réglage a semblé
appliqué sans l'être »*. On formalise donc une règle plutôt que de faire au cas par cas.

**Règle : chaque module de monde loggue, une fois par épisode, ce qu'il a RÉELLEMENT servi — mesuré,
jamais demandé.** Le modèle existe déjà : `[patch]` rapporte l'espacement MESURÉ et non celui
demandé ; `[prey]` rapporte la vitesse MESURÉE (0,00990 pour 0,00990 demandé) ; `[forest]` rapportera
le Clark-Evans mesuré.

**Ce que chaque objet du monde doit rapporter :**
| module | à logger (mesuré) |
|---|---|
| forêt | arbres placés / demandés, espacement mini réel, Clark-Evans, clairières, % d'occupation rétine |
| nourriture | nombre vivant, types servis + leur valeur, âge moyen, distance médiane à l'agent |
| proies / animaux | vitesse réelle, distance parcourue, fraction du temps visible, captures |
| corps | vitesse réelle m/tick, pénétrations de collision (doit rester 0), angle de regard |
| ressources | consommations par type, énergie rendue par consommation |
| perception | fraction de rayons occupés par classe, distance médiane du slot, hors-portée |

**Trois règles complémentaires :**
1. **Ne jamais dégrader en silence.** Si un réglage est infaisable (espacement trop grand, densité
   irréalisable), le DIRE bruyamment — `food_manager` le fait déjà (`push_warning` quand aucun centre
   n'est plaçable). Un monde qui se rabat sans le dire fabrique de faux négatifs.
2. **Une ligne de synthèse `[world]`** par épisode, listant les mécaniques ACTIVES et leurs valeurs
   mesurées. On doit pouvoir lire un log et savoir exactement quel monde a été servi — plusieurs
   confusions de la journée viennent de là.
3. **Bannière d'échafaudages actifs** — `diagnostics/guards.py` existe déjà pour ça (constantes
   MESURÉES vs DÉCLARÉES) ; l'étendre aux nouveaux modules plutôt que d'en écrire un autre.

**Sondes gratuites à écrire en même temps que la mécanique**, jamais après :
- lecture du type depuis l'encodeur (existe : `diag_latent_carries_type.py`) ;
- le latent porte-t-il l'objet (existe : `diag_latent_carries_object.py`) ;
- le prédicteur bat-il « l'objet reste immobile » (à écrire) ;
- occlusion : fraction de temps où une ressource vue est perdue de vue (à écrire) ;
- Clark-Evans (écrit, non vérifié).

---

## 6ter. CE QUI MANQUE À L'ARCHITECTURE — et dans quel ordre le construire

**Question owner : faut-il un réseau plus gros ? NON, et c'est mesuré.** À chaque comparaison de la
journée, un modèle LINÉAIRE a fait aussi bien qu'un MLP :

| test | linéaire | MLP |
|---|---|---|
| arbitraire (types) | 69,2 % | 69,7 % |
| hétérogénéité | 63,5 % | 64,0 % |
| lire le type dans le latent | 31,5 % | 28,2 % |
| lire l'objet dans le latent | +0,076 | −0,130 |

Et le latent n'utilise que **27 % de sa capacité** (rang effectif 34/128) : il ne sature pas, il
n'utilise pas ce qu'il a. **Un réseau plus gros n'aurait changé aucun résultat.** Le problème n'a
jamais été la capacité à représenter — c'est qu'il n'y avait rien à représenter, ou que l'information
avait été jetée en amont.

**Ce qui manque vraiment — et aucune de ces briques n'est un écart au JEPA : ce sont les parties NON
BÂTIES du blueprint de LeCun.**

1. **Un chemin d'apprentissage entre la CONSÉQUENCE et la PERCEPTION.** Le slot a zéro paramètre
   appris : l'agent ne peut pas réviser ce qu'il considère comme de la nourriture, quoi qu'il vive.
   C'est le manque le plus profond, et le critère JEPA (représentation informative) l'exige.
   Les requêtes apprises existent déjà (`wm_objcentric_kin_typed`), non promues.
2. **La représentation de l'INCERTITUDE.** Le WM est déterministe. LeCun prescrit des variables
   latentes pour les futurs multiples ; un JEPA déterministe prédit une *moyenne invalide* là où la
   planification a besoin de futurs énumérables (anomalie A4). Avec des flaques variables et des
   proies mobiles, ça devient bloquant.
3. **L'ABSTRACTION TEMPORELLE (H-JEPA).** Monde plus grand + vies plus longues ⇒ planifier à une seule
   échelle sur 80 ticks ne passera pas. C'est la réponse de LeCun au long horizon, et *la* brique qui
   permet des « stratégies » au sens courant.
4. **Une MÉMOIRE qui porte du poids.** `MultiSlotMemory` existe mais n'a jamais eu de raison d'exister.
   Occlusion + flaques à retenir lui en donneraient une.

**ORDRE, du plus causal au plus dérivé :**
1. **Perception** — requêtes apprises + couleurs variables dans la collecte. Sans ça, tout le reste
   s'appuie sur un substrat aveugle.
2. **Incertitude** — variables latentes, parce que le monde dessiné est intrinsèquement incertain et
   qu'un WM déterministe y prédira des moyennes qui n'existent pas.
3. **Hiérarchie temporelle** — le plus gros chantier ; n'a de sens qu'une fois le substrat voyant et
   le monde stratégique.
4. **Le critique en DERNIER.** Il échouait parce qu'il n'avait rien à apprendre ; il échouerait demain
   parce qu'il lit un substrat aveugle. **Il n'est pas la cause, il est le révélateur.**

### ⚠️ LE VRAI RISQUE (à nommer, pas à minimiser)

Ce n'est PAS « trop compliqué à apprendre ». C'est qu'**on ne sache plus ce qui échoue**. On empile
beaucoup de nouveautés dans UNE collecte — et on n'a pas le choix, chacune exigeant le retrain. Si le
résultat est mauvais, on ne saura pas laquelle est en cause. C'est en tension directe avec le principe
« une étape solide avant la suivante ».

**Seule mitigation : chaque ajout doit avoir SA sonde gratuite**, écrite en même temps que lui
(cf. §6bis). L'encodeur lit-il la couleur ? le prédicteur bat-il « l'objet reste immobile » ?
l'occlusion produit-elle du vu-puis-perdu ? Avec ça on peut ATTRIBUER un échec. Sans ça, on aura un
gros monde et un gros doute.

---

## 6quater. ANGLES MORTS — relecture critique (2026-07-24)

Relecture de tout ce qui précède à la recherche de ce qu'on a oublié. Classés par gravité.

### ⛔ A. QU'EST-CE QUI PERSISTE D'UNE VIE À L'AUTRE ? (le plus grave, jamais posé)

On n'a **jamais** répondu à ça, et ça conditionne l'apprenabilité de presque tout ce qu'on a décidé :

- **La table type → valeur** : si elle est retirée à chaque épisode, elle est **structurellement
  inapprenable** — l'agent ne peut pas « avoir goûté » dans une vie précédente. Or c'est le SEUL levier
  qui a passé le filtre. ⇒ elle doit être **FIXE sur toute la campagne**, pas seulement dans une vie.
- **La disposition de la forêt** : nouvelle à chaque vie ⇒ la mémoire spatiale ne sert **qu'à
  l'intérieur d'une vie**. C'est peut-être suffisant (retrouver une flaque vue il y a 500 ticks), mais
  ça doit être un CHOIX, pas un défaut hérité.
- **La position des flaques**, idem.

⚠️ Le piège symétrique : si tout est identique à chaque vie, l'agent peut **mémoriser la carte** au
lieu d'apprendre à percevoir — on fabriquerait un sur-apprentissage qu'on prendrait pour de
l'intelligence. Le bon réglage est probablement : **table type→valeur FIXE** (c'est une loi du monde),
**géographie VARIABLE** (c'est un tirage). À trancher explicitement.

### ⛔ B. LE DANGER — mentionné, jamais conçu

`hazard_manager.gd` existe et §2.9 le cite en une ligne. Rien n'est spécifié. Trois formes possibles,
de valeur très différente selon notre filtre :

- **Zones dangereuses fixes** (ravin, marécage) — perceptibles, prévisibles ⇒ probablement
  **dérivables** : « évite ce qui est rouge-vif » est une formule. Faible valeur d'apprentissage,
  mais utile comme structure spatiale (ça contraint les trajets, comme les arbres).
- **⭐ Le RISQUE DE BLESSURE À LA CHASSE** — une grosse proie peut blesser (le pack Quaternius a
  littéralement des animations *Kicks*). Attaquer devient un **pari** : gain contre risque.
  ⚠️ **MAIS ATTENTION** : le projet a DÉJÀ échoué là-dessus. Le chantier P2-bis a mesuré qu'un risque
  « pricé en espérance sans ancre d'aversion » dégrade tout, et le G3 arbitrage a exporté ses dégâts
  (danger 5→13). Ne pas rouvrir ça sans lire `design_purete_hjepa.md` §P2.
- **Un prédateur qui chasse le loup** — coûteux (un second agent mobile), et redondant avec le risque
  de blessure pour ce qu'on cherche.

### ⚠️ C. CHANGER LA VITESSE INVALIDE LA CALIBRATION MÉTABOLIQUE

Si on ouvre l'éventail de vitesse (§2.13), **toutes les constantes calées sur l'ancienne vitesse
deviennent fausses** : drain d'énergie par tick, rayon de capture 1 m, portée rétine 10 m, durée
d'épisode, densité de ressources. Le monde a été calibré pour 0,011 m/tick.
⇒ **Re-calibrer explicitement**, et ne pas s'étonner si la survie s'effondre au premier essai. C'est
un travail de mesure à part entière, pas un effet de bord.

### ⚠️ D. LE COÛT DE CALCUL D'UN MONDE RICHE (risque pratique de tout faire capoter)

Grande forêt + proies mobiles + distracteurs + flaques = beaucoup plus d'objets, de raycasts et de
collisions **par tick**. Or on collecte des **millions** de ticks.
⇒ **Mesurer les ticks/seconde AVANT de s'engager**, sur le monde cible. Si une collecte passe de 25 min
à 6 h, tout le plan change. Aucun de nos gates ne couvre ça aujourd'hui.

### ⚠️ E. POUR QUE LE TAPI AIT UN SENS, LA PROIE DOIT AVOIR UNE PERCEPTION

§2.11 propose de se tapir pour être moins visible, et §2.7a la distance de fuite. Les deux supposent
que **la proie voit** — donc qu'on lui implémente un modèle de perception (ligne de vue, portée,
seuil de détection dépendant du couvert et de la posture). Ce n'est pas une ligne de code : c'est un
petit système à part entière, et il n'est spécifié nulle part.

### 📋 F. Plus petits, mais à ne pas oublier

- **Relief / hauteur** — les loups utilisent le terrain pour observer. Un point haut qui donne une
  meilleure vue coupe bien avec le regard (§2.4) et l'occlusion. Non évoqué jusqu'ici.
- **Bord du monde** — aujourd'hui une arène circulaire avec réflexion. Dans une grande forêt, qu'y
  a-t-il au bord ? Un mur invisible est un artefact que l'agent apprendra à exploiter.
- **Déterminisme** — la recette (seed + mono-thread + serveur frais) a été payée cher. Beaucoup plus
  d'objets = beaucoup plus de consommateurs de RNG ⇒ **re-vérifier le rejeu bit-identique** après
  construction, sinon tous les juges contrefactuels tombent.
- **Cycle de vie** — le north-star mentionne des têtes ré-entraînées « la nuit » sur le vécu. Rien
  dans ce document ne dit comment le monde s'y raccorde.

---

## 7. RÉFÉRENCES

**Écologie spatiale** : processus de Neyman-Scott/Thomas (semis groupés) ; indice de Clark-Evans
(distance au plus proche voisin vs Poisson).
**Éthologie** : [choix des sites d'embuscade selon les capacités sensorielles des proies](https://academic.oup.com/beheco/article/32/2/339/6125068) ·
[distance de fuite et taille de groupe (méta-analyse)](https://www.sciencedirect.com/science/article/abs/pii/S0003347224000277) ·
[many-eyes / vigilance et taille de groupe](https://academic.oup.com/beheco/article/32/5/919/6307443) ·
[fourrageage sous incertitude : MVT + mise à jour bayésienne](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10996644/) ·
[théorie normative des décisions de quitter un site](https://arxiv.org/pdf/2004.10671) ·
[tactiques de chasse en forêt](https://www.sciencedaily.com/releases/2021/02/210209151819.htm) ·
[modélisation d'écosystème par RL profond](https://www.sciencedirect.com/science/article/pii/S1574954126002256)
**Architecture** : [TD-MPC, valeur terminale + bootstrap](https://proceedings.mlr.press/v162/hansen22a/hansen22a.pdf) ·
[LeCun, A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) ·
MoP-JEPA (un WM déterministe ne peut pas représenter un branchement stochastique) · H-JEPA (abstraction temporelle).
**Interne** : `audit_conformite_jepa.md` (anomalies A1-A5), `prereg_levier_perissable.md` (les cinq
leviers), `design_foret_bases_et_limites.md` (version courte).
