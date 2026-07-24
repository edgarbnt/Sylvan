# OUTIL — matrice de survie de l'information

**Statut** : **CONSTRUITE ET VALIDÉE** le 2026-07-24 (§7). Ce qui suit est la proposition d'origine,
conservée telle quelle ; §7 dit ce qui a été bâti, ce qui a été reproduit, et les deux surprises de
mesure trouvées en le bâtissant. **Date** : 2026-07-24.
**Motivation** : le risque nommé dans `design_foret_complete.md` §6ter — on va empiler beaucoup de
nouveautés dans une seule collecte, et *« le risque n'est pas que ce soit trop compliqué à apprendre,
c'est qu'on ne sache plus ce qui échoue »*.

---

## 1. L'observation qui justifie l'outil

Les trois vraies trouvailles du 2026-07-24 ont **exactement la même forme** :

| trouvaille | mesure |
|---|---|
| le type de proie | **83 %** dans la rétine → **30 %** après l'encodeur |
| « bouffe en vue » | **0,556** à la profondeur 0 → **0,160** à la profondeur 79 |
| la position de l'objet | absente du latent, **présente** dans le slot |

À chaque fois, la découverte n'était pas « ça marche mal » mais **« l'information disparaît ENTRE tel
étage et tel étage »**. Et à chaque fois, il a fallu une heure de sondes ad-hoc pour l'établir.

Dans une architecture JEPA, c'est *toujours* la bonne question : tout le principe repose sur « la
représentation garde-t-elle ce qui compte ? ». Un logger classique dit ce qu'un module a **fait** ;
cet outil dit **où l'information MEURT**.

---

## 2. Ce que c'est

Une **matrice** : en lignes les propriétés du monde, en colonnes les étages du pipeline, dans chaque
case la part de l'information récupérable (R² pour un continu, précision pour une catégorie, avec la
baseline majoritaire à côté).

| propriété du monde | rétine | encodeur | latent d0 | latent dH | slot | ce que le planner note |
|---|---|---|---|---|---|---|
| position de la ressource | | | | | | |
| type / couleur | 83 % | **30 % ⚠️** | | | | |
| maturité | | | 0,65 | | | |
| présence en vue | | | 0,556 | **0,160 ⚠️** | | |
| vitesse d'une proie | | | | | | |
| occlusion (vu-puis-perdu) | | | | | | |

**Une commande, un tableau.** Toute chute entre deux colonnes est une piste, et on sait immédiatement
quel module accuser.

---

## 3. Propriétés qui la rendent utile

- **GRATUITE** — aucun entraînement : on lit un corpus déjà collecté et un WM gelé.
- **NON-RÉGRESSION** — on la relance après chaque retrain et on compare les colonnes. Une case qui
  baisse = une régression du substrat, visible avant de dépenser une seule heure d'A/B.
- **EXTENSIBLE** — une nouvelle mécanique = une ligne de plus. Ça matérialise la règle « une sonde
  écrite EN MÊME TEMPS que la mécanique, jamais après » (`design_foret_complete.md` §6bis).
- **ATTRIBUTIVE** — c'est sa vraie valeur : elle transforme « le retrain a raté » en « la couleur ne
  passe pas l'encodeur, le reste va bien ».

---

## 4. Deux compléments plus légers

**Vérificateur de contrat de monde** — compare ce qui a été DEMANDÉ (le preset) à ce qui a été SERVI
(les logs mesurés), et crie si ça diverge. Le projet a déjà perdu du temps trois fois sur un réglage
qui semblait appliqué sans l'être ; ça l'automatise.

**Tableau de bord des gates** — rejouer tous les gates pré-enregistrés et afficher passé/échoué avec
l'historique. Aujourd'hui les verdicts sont dispersés dans des messages de commit ; on ne peut pas
voir d'un coup d'œil où on en est.

---

## 5. À NE PAS dupliquer

Deux briques existent déjà et l'outil doit s'y **brancher**, pas ouvrir un troisième endroit où lire
l'état du projet :
- `tools/archi_hud/` — la carte vivante de l'architecture (`voir_archi.sh`) ;
- `diagnostics/guards.py` — constantes MESURÉES vs DÉCLARÉES, bannière d'échafaudages actifs.

Les sondes déjà écrites qui deviendraient des **lignes** de la matrice :
`diag_latent_carries_type.py`, `diag_latent_carries_object.py`, `diag_critic_beyond_geometry.py`.

---

## 6. Ordre suggéré

1. **La matrice seule** — la plus utile des trois, testable immédiatement sur les corpus existants.
2. Le vérificateur de contrat, quand le monde forêt aura beaucoup de réglages.
3. Le tableau de bord des gates, quand les gates se multiplieront.

---

# 7. CONSTRUITE — ce qui existe, ce qu'elle reproduit, et ce que la construire a appris

**Code** : `python/sylvan/info_matrix.py` (mécanique de mesure, extensible) +
`diagnostics/diag_info_matrix.py` (une commande, un tableau).

```
PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_info_matrix.py \
    --corpus data/replay_buffer/critic_bosq_ripe11 [--depths 0 20 79] [--rows type vue] [--json m.json]
```

Lignes = propriétés du monde (position, distance, type, bouffe-en-vue, maturité, rétine entière ;
une mécanique de plus = une `Property` de plus). Colonnes = rétine brute → encodeur → latent rêvé à
chaque profondeur → slot → token du planner. Chaque case : sonde LINÉAIRE **et** sonde MLP, held-out
**par épisode**, baseline affichée (majorité pour une catégorie, moyenne pour un continu). WM GELÉ.

## 7.1 Gate d'acceptation : reproduire les mesures faites à la main — PASSÉ

| mesure établie à la main | valeur historique | la matrice | convention |
|---|---|---|---|
| typ31 type — rétine / encodeur / latent / majorité | 82,9 / 29,5 / 27,3 / **44,2 %** | **84,0 / 27,0 / 27,6 / 44,3 %** | MLP, split positionnel |
| lum41 type — rétine / encodeur / latent / majorité | 67,0 / 35,6 / 31,2 / **32,3 %** | **66,3 / 32,9 / 30,9 / 32,3 %** | MLP, split positionnel |
| ripe11 « bouffe en vue » — profondeur 0 / 20 / 79 | 0,556 / 0,304 / 0,160 | **0,494 / 0,308 / 0,148** | linéaire, cible prédite |
| ripe11 rétine ENTIÈRE depuis le latent (prof. 0) | +0,798 | **+0,808** | linéaire |
| ripe11 position PRÉCISE du slot depuis le latent | +0,046 (distance −0,884) | **+0,048** (distance −0,911) | linéaire |
| ripe11+12 maturité depuis le latent | lin +0,476 / MLP +0,650 | lin **+0,428** / MLP **+0,649** | les deux sondes |

Auto-contrôles qui tombent exactement où ils doivent : rétine→rétine = **+1,000**, slot→position =
**+1,000**. Et deux runs successifs sont **bit-identiques** (test explicite).

## 7.2 Ce que la construire a appris — deux pièges de mesure, tous deux corrigés

**(a) Les chiffres historiques n'étaient pas mesurés avec le split honnête.** Les valeurs de type
viennent d'un split **positionnel** (les 70 % premières lignes en train), qui coupe AU MILIEU d'un
épisode : à stride 6 la queue de l'épisode coupé est quasi identique à sa tête, donc ça fuit. Sous le
split honnête **par épisode**, le type tombe à 31,5 % (typ31) et 24,2 % (lum41) au latent. L'outil
garde le split par épisode PAR DÉFAUT et offre `--split positional` pour rejouer l'ancienne
convention — comparer des chiffres à des chiffres sans se mentir sur ce qu'ils valent.
De plus, la matrice signale que **les classes de type sont DÉCALÉES entre train et held-out**
(distance totale 0,31 sur typ31, 0,46 sur lum41 — les types sont re-tirés à la repousse) : sur lum41
la classe majoritaire du train est **absente** du held-out. La précision du type est donc
structurellement bruitée ; la conclusion qualitative (« la rétine porte le type, l'encodeur le
détruit ») tient, le centième ne tient pas. La sonde d'origine, relancée aujourd'hui, rend elle-même
31,5 / 28,2 % là où l'audit avait noté 27,3 %.

**(b) Une case pouvait CHANGER sans que rien ne change.** `torch.linalg.lstsq` sur une colonne
rang-déficiente (le token du planner porte un canal constant `connu`) rendait **+0,584 puis −1,178
pour la même entrée** d'un run à l'autre : la révélation de rang de LAPACK tranche une quasi-égalité.
Fatal pour l'usage NON-RÉGRESSION, qui compare des colonnes d'un retrain à l'autre. La sonde linéaire
est désormais un **vrai ridge** (λ=1e-3 sur features standardisées) : déterministe, et il borne aussi
les extrapolations délirantes (la rétine brute donnait R² −10,4 en moindres carrés nus).

## 7.3 Deux distinctions que l'outil rend explicites (elles étaient implicites, donc confondables)

**Le pipeline n'est pas une chaîne.** Le slot est une **branche séparée** encodée par `slot_encoder`
directement sur la rétine — jamais par l'encodeur du WM ni par le RSSM. Lire les chutes de gauche à
droite attribuerait au latent une perte qui appartient au slot, c'est-à-dire **accuserait le mauvais
module** : exactement ce que l'outil existe pour empêcher. Les chutes sont donc lues le long des
arêtes réelles (`rétine→encodeur→latent d0→…`, et `rétine→slot→token`).

**« Prédire » et « se souvenir » sont deux questions.** `--target predit` (défaut) juge le latent rêvé
à la profondeur d sur ce qui est VRAI à t+d — la question JEPA, et la convention des mesures
historiques. `--target percu` juge toutes les colonnes sur ce qui était vrai à t — la question de la
mémoire. À la profondeur 0 les deux coïncident, ce qui rend les deux lectures comparables à la racine.

## 7.4 Branchements (spec §5 : se brancher, pas ouvrir un troisième endroit)

- `diagnostics/guards.py` : bannière d'échafaudages + `sanity()` sur chaque corpus, et l'outil REFUSE
  de rendre une matrice sur un corpus dégénéré. La palette du monde est **MESURÉE** sur le corpus et
  comparée aux palettes déclarées (écart médian, part hors-palette) — même esprit que
  `check_constants`. Si un monde ne sert qu'une seule apparence, la ligne « type » rend un `n/a`
  franc au lieu d'un 100 % qui ne mesurerait rien.
- `tools/archi_hud/architecture.json` : le checkpoint WM déclaré par la carte est lu et **comparé** à
  celui que la matrice sonde ; divergence = avertissement en tête de sortie. (Elle en signale une
  aujourd'hui : la carte ancre `world_model` sur `wm_objcentric_s1` alors que le substrat servi par le
  corps cinématique est `wm_objcentric_kin`.)
- La matrice n'est PAS un module d'architecture : elle ne figure donc pas dans la carte, au même titre
  que `guards.py` et `archi_hud` eux-mêmes.
