# Forêt — BASES et LIMITES (pré-inscription, 2026-07-24)

**But de ce document** : poser le cadre du chantier forêt AVANT toute collecte ou retrain. Rien ici
n'est un résultat de ce chantier : ce sont les **contraintes mesurées** qu'il devra respecter, et les
**questions ouvertes** qu'il devra trancher. Aucun run n'a été lancé sur le monde forêt ce soir.

---

## 1. POURQUOI ce chantier (les mesures qui l'imposent)

Cinq leviers ont été testés aujourd'hui pour rendre un critique appris nécessaire. Un seul a réussi.

| levier | résultat mesuré | verdict |
|--------|-----------------|---------|
| conséquence (baies périssables) | 33 % de décisions conséquentes | sans effet sur la décision |
| feature non-géométrique (maturité) | lisible dans le latent à R² 0,65 | ne change pas l'ISSUE |
| prédiction (proie mobile) | interception 67,5 % vs poursuite 56,2 % | **forme close** → une formule suffit |
| hétérogénéité (choix entre proies) | formule 64,3 % vs MLP 64,0 % | **redondant** |
| **arbitraire (types de valeur)** | **formule 49,5 % vs appris 69,7 %** | ✅ **le seul où apprendre ≠ calculer** |

Deux constats structurels en découlent, et ce sont eux que la forêt doit corriger :

**(a) Le monde n'a presque aucune dynamique à apprendre.** Le déplacement prédit par le WM est
reconstructible à **R² 0,985 depuis la commande SEULE** — une droite. Corps cinématique + arène vide
+ nourriture immobile ⇒ la seule dynamique apprenable est « commande → déplacement ».

**(b) L'encodeur ne représente que ce qui a VARIÉ pendant son entraînement.** Le type d'une proie est
lisible à 82,9 % dans la rétine brute et 29,5 % après l'encodeur (majorité 44,2 %). Cause : la couleur
de la nourriture était CONSTANTE à l'entraînement. Trois explications alternatives testées et
réfutées : le canal (teinte ET luminosité détruites), la taille (baie 3,44 rayons vs buisson 3,17), la
sonde (même la moyenne agrégée est illisible, R² −0,659, quand la même mesure sur le buisson donne
+0,650).

⇒ **Corollaire opérationnel** : la collecte de retrain doit contenir TOUTES les variations qu'on
voudra un jour percevoir. Ré-entraîner pour une seule (la couleur) reproduirait le piège à la
prochaine idée. C'est la lecture correcte du §3 : enrichir le WM **une fois**.

---

## 2. LIMITES MESURÉES — le chantier doit les respecter

Ce ne sont pas des préférences, ce sont des mesures déjà payées.

**Couleur des arbres.** Le brun « naturel » est le PIRE choix : fuite mesurée **0,2271** sur la requête
rouge — un tronc brun est perceptuellement *rougeâtre*, donc confondable avec de la nourriture. Le
vert foncé `(0.13, 0.35, 0.13)` fuit **0,0000**. (`diag_foret_g0.py`, déjà dans `forest_solid.gd`.)

**Densité.** L'erreur de position rapportée par le slot passe de 0,00 m (0 % d'arbres) à 0,29 m (30 %)
puis **1,43 m à 60 %**. Le rayon de capture étant de 1 m, une forêt dense rend l'approche
**impossible**. Cible ≈ 30 % d'occupation de rétine. Bornes de navigabilité connues : 45 arbres =
fenêtre navigable, **54 arbres → 85 % du temps immobile**.

**Budget de déplacement.** Vitesse mesurée 0,011 m/tick × 3000 ticks = **~33 m de trajet par vie**.
⇒ Agrandir l'arène a une limite dure : à rayon 30 m (60 m de diamètre) l'agent traverse à peine une
fois. Toute augmentation de taille exige de compenser par la vitesse, la durée de vie, ou la densité
de ressources — **à mesurer, pas à supposer**.

**Rétine.** 36 rayons, portée **10 m** (et non 12), seuil de couleur 0,55, requêtes rouge (bouffe) et
bleu (eau). Ne pas y toucher : le seuil 0,55 est sous 1/√3, donc certains critères de « fuite nulle »
sont insatisfiables par construction.

**Le visuel ne doit pas mentir.** ⚠️ `forest_manager.gd` est aujourd'hui explicitement
« VISUAL-ONLY [...] NO collision, NO physics » et n'existe **qu'en mode visuel** : la jolie forêt est
invisible à l'agent et absente de l'entraînement. C'est une régression du type qu'on a passé la
journée à débusquer. **Tout élément dessiné doit avoir sa contrepartie perceptible** (couche rétine
bit 7 + `retina_color`) et, s'il bloque, sa collision (bit 2).

---

## 3. CE QUI EXISTE DÉJÀ (à réutiliser, pas à réécrire)

- `forest_solid.gd` — arbres **solides ET occultants**, opt-in `SYLVAN_FOREST_COUNT`, couleur choisie
  par mesure, contrat de couches déjà validé (G1, 0 pénétration). Câblé dans `main.gd`.
- `obstacle_manager.gd` — même contrat de couches, mur solide, prédicteur d'affordance associé.
- `ForestLowPolyAssets/` — 211 fichiers glTF (KayKit Forest Nature Pack, **CC0**) : arbres, rochers,
  buissons, herbes. Style cohérent avec le loup (Quaternius, CC0). Chargés au runtime.
- `forest_manager.gd` — décor visuel + cycle jour/nuit, **cosmétique uniquement** (cf. limite ci-dessus).

---

## 4. AJOUTÉ CE SOIR — codé, PAS ENCORE VÉRIFIÉ

Arrangement **écologique** dans `forest_solid.gd` (opt-in, `_stands <= 0` ⇒ comportement historique
bit-identique) :
- **peuplements** — processus de Neyman-Scott/Thomas : des centres, puis des arbres dispersés autour
  selon une gaussienne (`SYLVAN_FOREST_STANDS`, `SYLVAN_FOREST_STAND_SIGMA`) ;
- **clairières** — disques d'exclusion (`SYLVAN_FOREST_CLEARINGS`, `SYLVAN_FOREST_CLEARING_R`) ;
- **espacement minimal** conservé (concurrence entre arbres) ;
- **vérification** : indice de **Clark-Evans** loggé (distance moyenne au plus proche voisin rapportée
  à l'attendu sous Poisson ; < 1 = groupé, 1 = aléatoire, > 1 = régulier).

**Pourquoi ça sert, au-delà du réalisme** : une forêt structurée produit une occlusion NON UNIFORME —
couloirs, écrans, ouvertures. C'est ce qui rend une position meilleure qu'une autre, donc l'affût
signifiant. Un semis uniforme n'est qu'un brouillard homogène.

🚧 **STATUT : NON VÉRIFIÉ.** Le parse GDScript passe, mais la comparaison uniforme-vs-écologique n'a
pas été menée à terme. **Ne rien conclure ni promouvoir avant** d'avoir lu un Clark-Evans mesuré < 1
sur le semis écologique et ≈ 1 sur l'uniforme.

---

## 5. QUESTIONS OUVERTES — à trancher AVANT la collecte

1. **Taille de l'arène.** Combien peut-on agrandir sans rendre le monde intraversable (budget 33 m/vie) ?
   Compenser par la vitesse, la durée de vie, ou la densité ? → mesure analytique d'abord.
2. **Tête mobile / regard indépendant.** C'est la seule mécanique identifiée dont la valeur d'action
   serait **purement informationnelle** (regarder ne rapproche de rien). Elle ajoute une dimension à la
   proprioception ⇒ **exige un retrain**. Donc la décision doit être prise **avant** la collecte, pas
   après. Donner la CAPACITÉ, jamais le comportement de balayage (sinon c'est le raccourci codé-main).
3. **Quelles variations mettre dans la collecte ?** Au minimum : apparences (couleur, luminosité,
   taille), objets MOBILES (pour que le prédicteur apprenne une dynamique externe), occlusion. Toute
   variation absente de la collecte sera invisible au WM pour toujours.
4. **Affût / couvert** — la mécanique la mieux fondée (les loups choisissent leurs sites d'embuscade
   en fonction de ce que la proie peut percevoir). Non spécifiée à ce stade.

---

## 6. GATES pré-enregistrés (aucun n'est encore passé)

- **arrangement** : Clark-Evans < 1 en mode peuplements, ≈ 1 en uniforme.
- **navigabilité** : survie non effondrée vs monde sans arbres, à densité cible.
- **perception** : après retrain, lecture du type au niveau de l'ENCODEUR de ~30 % → **> 70 %**.
- **dynamique** : le prédicteur doit battre l'hypothèse « l'objet reste immobile » sur une proie mobile.
- **non-régression WM** : éval open-loop existante (position, yaw, eff_rank) au moins au niveau actuel.

⚠️ Réserve honnête déjà notée : JEPA jette *délibérément* l'imprévisible. Un type tiré au hasard à
chaque repousse est partiellement imprévisible ; il est possible que l'encodeur continue de l'écraser
malgré la variation. Le gate « > 70 % » est là pour le dire sans déplacer les poteaux.
