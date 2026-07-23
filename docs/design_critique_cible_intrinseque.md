# Critique appris = cible-coût-intrinsèque (LeCun) — DIAG GRATUIT, KILL sur le MONDE (2026-07-23)

## Décision de méthode
Avant un 4ᵉ run de critique appris, recherche SOTA (workflow) + LeCun v0.9.2 lu en source primaire,
puis diagnostic GRATUIT sur corpus. Le diag a tué le chantier — pour la bonne raison, nommée.

## Ce que LeCun prescrit (source primaire, §3)
`C(s) = IC(s) + TC(s)`, somme simple. IC **immuable, hard-wired** (pulsions : faim, douleur) — il
INTERDIT de l'apprendre (« must be immutable and not subject to learning »). Le TC (critic) apprend
à prédire le **coût intrinsèque FUTUR** : `‖IC(s_{τ+δ}) − TC(s_τ)‖²`. Entraînable sur états vécus OU
imaginés par le WM. Ni δ ni γ donnés. ⇒ nos pulsions codées-main NE SONT PAS une impureté : elles
sont l'IC prescrit. Seul le coût de PLAN (21 boutons) est le TC à remplacer.

## Les 3 échecs bankés = UN SEUL bug, nommé par l'état de l'art
Le planner classe 117 candidats DANS LE MÊME ÉTAT → seule la variance **intra-état** compte. Nos
mesures (R², « inné +0,437 ») étaient **poolées** → dominées par la variance **inter-état** (où est
l'agent), sur laquelle le planner n'a aucun levier. Un modèle qui prédit la moyenne par état et
classe au hasard obtient déjà « inné +0,437 ». On mesurait la mauvaise variance.
Réf : Farebrother 2024 (offset constant dégrade la MSE sans changer la fonction) ; Fujimoto ICML'22
(faible erreur de régression ≠ qualité de décision).

## La bonne cible (jamais testée avant)
Résidu escompté ≈ N repas dans la fenêtre : `(faim(τ+δ) − faim(τ) + 0,05·δ)/40`. Mesuré, δ=600,
24001 ex : **BIMODAL** (0 repas 46 % / 1 repas 47 % / 2 repas 6 %) → une tête MSE prédirait 0,73,
valeur quasi jamais observée. Choix de cible VALIDÉ.

## 🚨 LE KILL, mesuré
Gate pré-inscrit par la synthèse : `Var_intra-état / Var_totale > 10 %`, sinon la cible est du bruit
de monde et **c'est le MONDE, pas le critique**.
**MESURE (état ≈ énergie/5 × position 1 m, 436 cellules ≥ 3 visites) : 8,2 %.** Et c'est une
SUR-estimation (les cellules regroupent des ticks proches, pas de vrais contrefactuels du même
instant). Sous la barre, sur une mesure optimiste. → **KILL.**

## Cause-racine, qui reboucle toute la session
Le corps est **trop RÉCUPÉRABLE** : il pivote sur place, obéit exactement à (vx,ω), et le MPC
replanifie chaque tick → la commande choisie MAINTENANT n'engage presque rien ; la faim à τ+δ est
fixée à ~92 % par la position + les 600 ticks de replan à venir. Une décision est toujours défaisable.
C'est le MÊME mur, 4ᵉ visage : valeur apprise inutile (rien à classer), prédiction inutile
(trajectoire analytique), mémoire payante seulement sous cône (il fallait FABRIQUER de
l'irréversibilité perceptive).

## Ce que le levier N'EST PAS, et ce qu'il EST
PAS la tête de valeur (elle n'a rien à classer). PAS un monde plus dur (ça déplace le seuil de
survie, pas la conséquence d'une commande). Le levier = **rendre les décisions CONSÉQUENTES** :
coût de rotation réel / momentum / action irréversible — pour qu'un candidat DIFFÈRE durablement
d'un autre. Chantier CORPS/MONDE, pas critique. À pré-inscrire séparément.

## Négatifs bankés (ne pas répéter)
- Ne pas juger un critique au R² POOLÉ : mesurer le rang INTRA-état (Kendall τ, regret@1), split
  GroupKFold par VIE.
- Un corpus n'observe qu'UN candidat exécuté par état : aucun gate de rang honnête n'est calculable
  sans étiqueter les 117 candidats (rêve WM, biaisé) ou re-collecter en branchant des alternatifs.
- symlog écrase l'étendue 48-98 → 0,70 (rejeté) ; HL-Gauss ne bat MSE qu'en dynamique déterministe,
  or respawn aléatoire (différé, pas prioritaire).

---

# ADDENDUM (2026-07-23) — le levier est le PLANNER, pas le corps

`diagnostics/diag_consequence_g0.py`, gratuit, selfcheck passé (dont un contrôle décisif : à
horizon trivial la variance intra est EXACTEMENT nulle, la sonde ne fabrique pas de signal).

Le KILL disait « le corps est trop récupérable ». La sonde SÉPARE les deux causes possibles en
déroulant les 117 candidats du planner depuis un MÊME état — de vrais contrefactuels, que le corpus
ne pouvait pas fournir (il n'observe qu'un candidat exécuté par tick).

| replan | ratio intra | faim moyenne | regret |
|---|---|---|---|
| **10 (actuel)** | **1,9 %** | 77,5 | 0,8 |
| 30 | 4,1 % | 78,1 | 1,7 |
| **60** | **12,7 %** | 78,6 | 3,6 |
| **120** | **38,9 %** | 78,6 | 7,9 |
| 300 | 71,8 % | **58,2** ⚠️ | 30,0 |

Axe corps (inertie `tau`) à replan=10 : 2,2 % / 2,1 % / 2,0 % / 4,2 % / **11,1 %** pour
tau = 0 / 5 / 15 / 30 / **60** ticks. Il faut donc un corps qui met ~60 ticks à répondre — pataud,
et qui imposerait une recollecte du WM.

⇒ **L'engagement du planner obtient le même effet à `replan-every` 60-120, qui est un PARAMÈTRE
EXISTANT** (aujourd'hui 10). Gratuit, réversible, zéro ré-entraînement.

**Contrôle anti-fausse-solution (§2) PASSÉ** : entre 60 et 120 la faim moyenne ne bouge pas
(77,5 → 78,6) pendant que le regret monte de 0,8 à 7,9 — on crée de quoi choisir SANS dégrader.
À 300 la faim s'effondre à 58,2 : là on fabriquerait de la variance en cassant l'agent, refusé.

⚠️ **Le chiffre d'hier (8,2 %) était OPTIMISTE** : il groupait des ticks proches faute de
contrefactuels. Avec de vrais candidats du même instant : **1,9 %**. Le KILL est plus net, et sa
cause est déplacée — ce n'est pas que le corps soit récupérable, c'est que **le planner efface sa
propre décision dix ticks plus tard**. Replanifier souvent rend l'agent robuste, et c'est
exactement ce qui le prive de décisions à apprendre.

⚠️ **CE QUE ÇA NE DIT PAS** : qu'il y aurait quelque chose à CLASSER, pas que l'entité ira mieux.
Allonger le replan la rendra moins réactive, et ceci est mesuré en simulation avec un relais
glouton, pas en vies. Condition NÉCESSAIRE. Le vrai test = A/B en vies sur `--replan-every`
(10 vs 60 vs 120), jugé sur les repas, AVANT de rouvrir le critique appris.
