# Recherche — pourquoi le critique appris perd l'argmax (et ce que fait la littérature)

> Session de RECHERCHE du 2026-07-14, branche `feat/critic-clean-foundations`. Zéro entraînement,
> zéro épisode Godot : tout ce qui suit est mesuré par une sonde gratuite (2 min) ou cité.

## Mission

Décider quoi faire du critique appris, qui divise le forage par 2 (16 contre 34.5 pour la formule
codée-main) alors qu'il est excellent en régression (R² 0.972, corrélation 0.82 avec l'oracle).

## À lire d'abord

- `python/sylvan/control/planning/command_planner.py:725-726` — les deux lignes en cause.
- `diagnostics/diag_critic_aggregation.py` — la sonde écrite ici ; elle rejoue plusieurs façons de
  consommer la valeur sur les MÊMES rêves, sans rien ré-entraîner.
- `python/scripts/train_survival_critic.py:214` — la perte MSE sur retours Monte-Carlo.

---

## 1. Le fait mesuré : ce n'est pas un problème de valeur, c'est un problème d'ÉCART

Le planner note 33 candidats et exécute le meilleur. La question n'est donc pas « la valeur est-elle
juste ? » mais « **l'écart entre le meilleur et le DEUXIÈME est-il plus grand que l'erreur du réseau ?** »
Personne ne l'avait mesuré. C'est le seul chiffre qui décide.

Sur 76 états de replan réels × 33 candidats (`diag_critic_aggregation.py`) :

| agrégation | écart meilleur-pire | **écart d'action** (1ᵉʳ − 2ᵉ) | erreur du critique | **bruit / écart** |
|---|---|---|---|---|
| moyenne sur l'horizon *(le vivant)* | 0.0020 | **0.00001** | 0.00024 | **31×** |
| moyenne du dernier quart | 0.0035 | 0.00002 | 0.00045 | 23× |
| valeur terminale seule | 0.0040 | 0.00003 | 0.00053 | 19× |
| somme escomptée | 0.0377 | 0.00010 | 0.00472 | 47× |

**L'erreur du critique est 19 à 47 fois plus grande que l'écart qu'on lui demande de trancher.** Elle
l'est dans les 4 agrégations, et elle le reste quand on allonge le rêve (horizon 80 → 160 → 240 : l'écart
d'action monte ×5, mais le bruit monte autant — le ratio ne descend jamais sous 11).

Le corollaire est brutal et il explique TOUT l'historique : **l'oracle analytique gagne (36.5) non pas
parce qu'il est meilleur, mais parce que son erreur est EXACTEMENT ZÉRO.** Un écart de 1e-5 se tranche
parfaitement quand on ne se trompe jamais. Aucune approximation neuronale — quelle que soit sa qualité,
ses données ou sa capacité — ne descendra jamais son erreur à 1e-5. **La fente actuelle est fermée à
tout critique appris, par arithmétique.** Ce n'est pas un défaut du réseau : c'est la question qu'on lui pose.

### Deux conséquences qui invalident des conclusions passées

**(a) La sonde ORACLE ne pouvait PAS innocenter la moyenne-sur-l'horizon** (conclusion du 2026-07-13,
« mean-aggregation is NOT the bottleneck »). Une agrégation qui *dilue* l'écart préserve l'ORDRE tant que
la valeur est parfaite — elle ne nuit que s'il y a du bruit. Tester la moyenne avec un oracle sans erreur
ne pouvait donc rien révéler. *(Ceci dit : la sonde ci-dessus tue quand même l'hypothèse « changer
l'agrégation suffit » — c'était MON hypothèse d'entrée, et elle est fausse. Négatif informatif.)*

**(b) ~98 % de l'erreur du critique est un DÉCALAGE COMMUN à tous les candidats**, qui s'annule dans la
comparaison : l'erreur brute est 0.0126, mais après alignement affine sur l'oracle elle tombe à 0.00024.
Le réseau dépense donc sa capacité sur les 98 % qui ne servent à rien pour décider. **C'est pourquoi
chaque correction principielle a fait BAISSER le forage : elles amélioraient le mauvais 98 %.**

## 2. Le trou béant, non soupçonné : le critique tranche avec assurance quand il ne voit RIEN

Sur les **37 % de replans où la ressource urgente est hors-vue** (44 des 120 états ; le token dit
« connu=0 ») :

| | dispersion du score sur les 33 candidats |
|---|---|
| ORACLE (l'information réellement disponible) | 0.0016 |
| **CRITIQUE APPRIS** | **0.0091 — 5.6× plus « contrasté »** |

Le critique **fabrique une préférence là où il n'a aucune information**, et il pilote ainsi plus d'un tiers
des décisions en déploiement. L'ancien gate (`has_food AND has_water`) masquait exactement ce cas :
en le levant (`SYLVAN_CRITIC_ALWAYS=1`, commit `7293e3e`), on a livré ces 37 % d'hallucinations au corps.
**Une partie de la division-par-2 du forage vient très probablement de là, et pas de l'argmax.**

## 3. Ce que fait la littérature (vérifié, sourcé)

**Personne ne note une trajectoire par `moyenne de V`. Personne ne prend un argmax dur.** C'est unanime.

| | score d'une trajectoire imaginée | sélection de l'action |
|---|---|---|
| **TD-MPC / TD-MPC2** (arXiv 2203.04955 éq. 3 ; 2310.16828 éq. 6) | `Σ γᵗ R(z_t,a_t) + γᴴ Q(z_H,·)` — H récompenses + **1 seul** bootstrap terminal | MPPI : moyenne pondérée `exp(τ·G)` sur les 64 élites de 512, puis **tirage** (Gumbel), même en éval |
| **MuZero** (1911.08265, ann. B éq. 3-4) | `Σ γᵗ r + γˡ v(leaf)` | **comptes de visites**, pas argmax de valeur |
| **Gumbel MuZero** (ICLR 2022) | idem | `argmax(g + logits + σ(q̂))` — **preuve** (éq. 6-7) : jamais pire que le prior, à n'importe quel budget |
| **PlaNet** (1811.04551) | somme des récompenses prédites | **moyenne des 100 meilleurs** sur 1000 |
| **Dreamer v1-v3** | λ-retour = récompenses **+** valeur en queue | pas de recherche : **acteur amorti** |
| **Sylvan (nous)** | `moyenne_t V(z_t)` — **aucune récompense, aucun escompte** | **argmax dur sur 33 candidats** |

Deux résultats théoriques cadrent exactement notre mesure :

- **Écart d'action** (Farahmand, NIPS 2011) : l'action gloutonne est correcte tant que `erreur < écart/2`.
  Nous sommes à `erreur ≈ 30 × écart`. Le théorème ne nous protège de rien — il *nomme* notre échec.
  **Tallec, Blier, Ollivier (ICML 2019, arXiv 1901.09732)** est encore plus proche : ils *prouvent* que
  l'écart d'action **s'effondre quand le pas de temps devient petit devant l'échelle de la tâche** —
  notre rêve fait 1.6 m contre des ressources à 2-8 m. C'est notre régime exactement.
- **Malédiction de l'optimiseur** (Smith & Winkler, 2006) : en prenant l'argmax de N estimations non-biaisées,
  on sélectionne **le bruit le plus chanceux**. Pour N = 33, le gagnant est surévalué de **≈ 2.1 σ** par la
  seule sélection. Même résultat côté RL : van Hasselt (Double-Q, 2016) th. 1 ; Thrun & Schwartz (1993).

## 4. LeCun — le projet s'en réclame, et le contredit sur le point central

Lecture du blueprint (*A Path Towards Autonomous Machine Intelligence*, v0.9.2), verbatim :

> « Le module de coût […] est composé de deux sous-modules, **le coût intrinsèque, qui est immuable (non
> entraînable)** […] **et le critique, un module entraînable qui prédit les valeurs futures du coût
> intrinsèque.** » (légende fig. 2)
> « **C(s) = IC(s) + TC(s)** » (§3.2, éq. 1) — et l'énergie totale optimisée est **`F(x) = Σₜ C(s[t])`** (§3.1.2).
> « **Pour éviter un effondrement comportemental […] l'IC doit être immuable et non sujet à l'apprentissage.** » (§3.2)
> Rôle du critique : « **anticiper les résultats à long terme en utilisant le moins possible le coûteux
> world-model** » (§3.2).

**Le critique n'a JAMAIS été censé remplacer le coût — il s'AJOUTE à lui.** Chez LeCun, la discrimination
entre candidats est portée par `IC`, exact et dense ; `TC` ne fait qu'anticiper l'au-delà de l'horizon.
Notre `-min_dist` / coût survie **EST un `IC(s)`** au sens du blueprint : une fonction différentiable et
connue de l'état prédit, pas une récompense opaque de l'environnement (§8.3.2, « *plus proche du contrôle
optimal que de l'apprentissage par renforcement* »).

Et le `CLAUDE.md` du projet dit déjà la même chose (PRINCIPE N°3) : *« DRIVE = propriété du CORPS, définie
une fois à la conception, comme l'évolution câble les pulsions »*. **L'objectif « pureté = supprimer la
formule codée-main » n'est ni dans LeCun, ni dans nos propres principes. C'est cet objectif qui est faux,
pas le critique.**

---

## 5. Quatre changements, classés par (gain ÷ coût)

### ① Ne pas laisser le critique décider là où il ne voit rien — *~3 lignes*
`command_planner.py:604` gate le critique sur `has_food AND has_water`. La levée (`CRITIC_ALWAYS=1`) lui a
donné les 37 % de replans aveugles, où il hallucine (§2). Le bon gate n'est ni l'un ni l'autre : **le critique
note quand la ressource URGENTE est connue** (≈ 63 % des replans) ; sinon le problème n'est pas d'ÉVALUER
mais de **CHERCHER** — une capacité que ni le critique ni la formule ne possèdent (dette déjà notée).
- **Test gratuit** : déjà fait (§2). **A/B ensuite** : forage gate-visibilité vs `CRITIC_ALWAYS`. *Succès* = ≥ 28 (mi-chemin vers 34.5). *Kill* = ≤ 20.
- ⚠️ Honnêteté §2 : c'est un **aiguillage**, pas une guérison. Il ne rend pas le critique capable ; il l'empêche de nuire.

### ② Supprimer l'argmax dur — *~5 lignes*
`command_planner.py:726`. Aucune méthode établie ne fait ça (§3), et Smith & Winkler chiffrent la perte à
≈ 2.1 σ pour N = 33. Remplacer par la moyenne pondérée MPPI sur les k meilleurs (TD-MPC éq. 4) : la commande
exécutée devient la moyenne des candidats quasi-ex-aequo au lieu du plus chanceux.
- **Test gratuit** : sur les 76 états, comparer la valeur-oracle de la commande MPPI vs celle de l'argmax.
  *Valide* si valeur-perdue ↓ ET « tourne du bon côté » ↑. *Kill* si ≤ argmax.
- Bénéfice de bord : bénéficie AUSSI au coût codé-main.

### ③ Remettre le coût intrinsèque DANS le score : `score = Σₜ [IC(sₜ) + TC(sₜ)]` — *~10 lignes*
**C'est le vrai changement de trajectoire.** C'est la configuration prescrite par LeCun (§4) et la seule
qu'on n'ait JAMAIS essayée : on a toujours opposé `designed` OU `critic`, jamais `designed + critic`.
Elle dissout le mur du §1 par construction : l'écart d'action est alors porté par un terme **exact**, et le
critique n'a plus à trancher des ex-aequo à 1e-5 — il ajoute l'anticipation long-terme, ce pour quoi il est bon.
- **Test gratuit** : sur les mêmes 76 états, mesurer `bruit / écart` et la valeur-perdue pour
  `score = IC + λ·TC`, λ ∈ {0, 0.1, 0.3, 1}. *Valide* si ratio < 1 et si λ > 0 ne dégrade pas le classement de λ=0.
  *Kill* si le critique dégrade le classement à tout λ > 0 (⇒ il est activement nuisible, même en appoint).
- **Coût réel = renoncer à « la boucle 100 % apprise » comme but.** Décision d'owner, pas d'ingénieur.

### ④ Apprendre l'AVANTAGE, pas la valeur — *un ré-entraînement (minutes), + planner*
`train_survival_critic.py:214` régresse la valeur ABSOLUE au MSE. Or 98 % de cette valeur est un socle commun
qui s'annule dans la comparaison (§1b) : le réseau optimise le mauvais 98 %. Le dueling (Wang et al., ICML 2016)
énonce notre symptôme mot pour mot (« *les écarts de Q sont minuscules devant la magnitude de Q ; un peu de
bruit réordonne les actions* ») et prescrit de paramétrer `V(s) + A(s,a)` en n'utilisant que `A` pour décider.
- **Test gratuit d'abord** : le socle commun est-il bien la cause ? — **déjà mesuré : oui, 98 %** (§1b).
- **Gate avant tout run** : le critique-avantage doit atteindre `bruit / écart < 1` sur les 76 états. **S'il ne
  passe pas ce gate offline, ne PAS lancer de forage** (c'est le gate qu'on n'a jamais posé, et c'est pour ça
  qu'on a brûlé 4 runs).

---

## 6. Ce que la session a TUÉ (négatifs informatifs — ne pas y revenir)

- **« Il manque un terme de récompense / il faut bootstrapper la valeur terminale »** (soupçon n°1 du prompt) :
  testé sur 4 agrégations — le ratio bruit/écart ne s'améliore jamais. La forme canonique `Σγᵗr + γᴴV` ne peut
  pas nous sauver seule, parce que **rien ne se passe dans le rêve** (1.6 m contre des cibles à 2-8 m) : la
  récompense intra-horizon est quasi-constante entre candidats. *Ce n'est pas faux, c'est inopérant ici.*
- **« Allonger l'horizon »** : l'écart d'action monte ×5 (80 → 240 pas), mais le bruit monte autant. Ratio
  toujours ≥ 11. *(Rider utile quand même : à horizon 160-240, la valeur perdue par une erreur d'argmax tombe
  de 27 % à 2 % — le problème devient plus INDULGENT. À tester en appoint de ③, pas seul. Attention : Jiang
  et al. 2015 — allonger l'horizon échange de l'écart contre de l'erreur de modèle, ce n'est pas gratuit.)*
- **« Distiller depuis l'oracle »** (piste owner du 2026-07-08) : R² 0.972 et pourtant 23/36.5. Le §1 dit
  pourquoi c'était condamné d'avance : la distillation réduit l'erreur, elle ne la met pas à ZÉRO — et il
  faut zéro. **La question qu'elle devait trancher (« ne sait pas exprimer » vs « ne sait pas apprendre »)
  a maintenant une troisième réponse : ni l'un ni l'autre — la fente est trop étroite pour QUICONQUE.**

## 6bis. « L'entité peut-elle progresser de son vécu ? » — OUI, et c'est mesuré (2026-07-15)

Question de l'owner, après le §4 : si le critique n'a plus à remplacer le coût, reste-t-il quelque chose
à apprendre ? Un critique ne peut apporter que le **résidu** — l'écart entre ce que l'inné prédit et ce qui
arrive vraiment. Si ce résidu est du bruit, le vécu n'enseigne rien. Sonde `diag_experience_residual.py`
(gratuite, 57 vies mortes non-tronquées, split par ÉPISODE) :

| | |
|---|---|
| survie réellement vécue | médiane **930** pas |
| survie prédite par l'inné | médiane **1572** pas → **l'inné est optimiste de ~1,7×** |
| l'inné explique-t-il le vécu ? | R² **+0.52**, corrélation de rang **+0.88** (il ORDONNE bien, il CALIBRE mal) |
| **le résidu est-il apprenable ?** | **R² = +0.21 sur des vies JAMAIS VUES** (critère pré-enregistré : ≥ 0.15) |

**Le vécu contient une leçon, et elle est structurée** (ce n'est pas du bruit : un petit réseau la retrouve
sur des vies qu'il n'a jamais vues). Cette leçon a un contenu identifiable : l'inné suppose un trajet **droit,
à vitesse nominale, avec alternance parfaite** — la réalité (errance, hésitation, virages) coûte ~40 % de vie
en moins. **C'est exactement ce qu'un critique peut apprendre et que la géométrie ne saura jamais dire.**

→ Renforce ③ + ④ : `score = IC(inné, exact) + TC(résidu appris)`. Le résidu n'a **pas de socle commun** (il est
petit par construction) → le problème des 98 % du §1b disparaît de lui-même.
⚠️ Honnêteté §2 : corpus d'UNE politique déterministe → on mesure ce qui est apprenable **sous cette politique**
(boucle auto-confirmante déjà connue). R² 0.21 = réel mais **modeste**. Ne pas survendre.

## 7. Critère de succès (le BUT, pas le proxy)

Forage = **repas + boissons sur 12 vies** (jamais la survie médiane : le plafond épars est MÉTABOLIQUE,
`diag_metabolic_ceiling.py`). Référence à battre : **34.5** (formule codée-main). Plafond de la fente
actuelle : **36.5** (oracle). État du critique : **16**.
