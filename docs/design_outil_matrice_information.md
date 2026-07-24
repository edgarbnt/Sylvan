# OUTIL — matrice de survie de l'information (à construire dans une autre session)

**Statut** : proposition, rien n'est construit. **Date** : 2026-07-24.
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
