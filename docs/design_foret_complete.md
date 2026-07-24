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

## 7. RÉFÉRENCES

**Écologie spatiale** : processus de Neyman-Scott/Thomas (semis groupés) ; indice de Clark-Evans
(distance au plus proche voisin vs Poisson).
**Éthologie** : [choix des sites d'embuscade selon les capacités sensorielles des proies](https://academic.oup.com/beheco/article/32/2/339/6125068) ·
[tactiques de chasse en forêt](https://www.sciencedaily.com/releases/2021/02/210209151819.htm) ·
[modélisation d'écosystème par RL profond](https://www.sciencedirect.com/science/article/pii/S1574954126002256)
**Architecture** : [TD-MPC, valeur terminale + bootstrap](https://proceedings.mlr.press/v162/hansen22a/hansen22a.pdf) ·
[LeCun, A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) ·
MoP-JEPA (un WM déterministe ne peut pas représenter un branchement stochastique) · H-JEPA (abstraction temporelle).
**Interne** : `audit_conformite_jepa.md` (anomalies A1-A5), `prereg_levier_perissable.md` (les cinq
leviers), `design_foret_bases_et_limites.md` (version courte).
