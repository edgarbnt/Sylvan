# Prompt — session de RECHERCHE : comment le critique est-il réellement utilisé dans la littérature ?

> Copier-coller le bloc ci-dessous comme premier message d'une session fraîche.
> Écrit le 2026-07-08, à la fin d'une session qui a produit un diagnostic net mais aucune solution.

---

Sylvan — session de RECHERCHE, pas d'implémentation. Branche `feat/critic-clean-foundations`
(dernier commit `7293e3e`). Venv `env_pytorch_3.12`, CPU, `PYTHONPATH=python`, depuis la racine.

LIRE D'ABORD : `memory/MEMORY.md` + `memory/sylvan-mode1-build.md` (chercher « 2026-07-08 »),
`tools/archi_hud/architecture.json` (module `critique_appris`, état = `partiel`), et les commits
`7293e3e` + `f21624b`. Puis re-checker les orphelins (`pgrep -xc godot`), git log, disque.

## Le problème, et il est MESURÉ (ne pas le re-découvrir, partir de là)

Le planner imagine 33 séquences de commandes `(vx, ω)` dans le world-model, note chacune, et
exécute la meilleure. La note vient soit d'une formule codée-main, soit d'un CRITIQUE APPRIS
(`SurvivalCritic`, `python/scripts/train_survival_critic.py`). Le but du projet est de remplacer
la formule par le critique. **Ça ne marche pas, et on sait maintenant exactement pourquoi.**

Consommations (repas + boissons sur 12 vies, 2 graines) :

| Qui note les 33 candidats | graine 1 | graine 6 | moyenne |
|---|---|---|---|
| Formule codée-main (`designed`, le défaut livré) | 41 | 28 | 34.5 |
| **ORACLE** : la valeur analytique EXACTE, dans la fente du critique | 33 | 40 | **36.5** |
| Critique appris sur retours Monte-Carlo du vécu | 20 | 12 | 16 |
| Critique DISTILLÉ sur les notes de l'oracle (R² = 0.972) | 19 | 27 | 23 |

**Le mur, chiffré** (mesuré sur 60 vrais états de replan, 33 candidats chacun) :

- écart entre le meilleur et le pire candidat (**le signal à résoudre**) : **0.0148**
- erreur typique du critique distillé vs l'oracle (**le bruit**) : **0.0126**
- → **l'erreur fait 85 % du signal**
- corrélation critique ↔ oracle sur les notes : **+0.82** (il a bien appris la fonction)
- mais il choisit **le même candidat que l'oracle seulement 35 % du temps**

Autrement dit : **un critique excellent (R² 0.97, corrélation 0.82) se trompe de choix 2 fois sur 3**,
parce que l'argmax exige une précision qu'aucune approximation neuronale ne peut fournir. Ce n'est
pas un défaut du réseau — **c'est la question qu'on lui pose qui est mal posée.**

Éliminés définitivement par l'expérience : les DONNÉES (entraîner sur un bon forageur ou sur ses
propres erreurs donne le même échec), les NOTES (la distillation depuis un professeur parfait ne
donne que 23 sur un plafond de 36.5), et la CAPACITÉ du réseau (R² 0.972).

## Deux soupçons à confirmer ou démolir (formulés de mémoire, DONC À VÉRIFIER)

**1. Il manque un terme de RÉCOMPENSE.** Comparer les deux scores :

```
coût codé-main :  (pas survécus PENDANT le rêve)  +  (valeur analytique à l'arrivée)
                   └── terme de RÉCOMPENSE ────┘      └── bootstrap ──┘

notre critique :  moyenne de V le long du rêve
                  └── QUE de la valeur, AUCUNE récompense ──┘
```

Le terme de récompense différencie FORTEMENT les candidats (s'approcher rapporte tout de suite),
alors que les valeurs terminales se ressemblent. La forme canonique en planification avec valeur
apprise serait `Σ γ^t r(s_t) + γ^H V(s_H)` — un n-step return avec bootstrap. Notre `mean(V)` n'est
pas ça, et n'a d'ailleurs aucune justification théorique (héritage d'un autre chapitre du projet,
jamais ré-examiné : `command_planner.py`, `score = vmap.mean(dim=1)`).

**2. On prend un ARGMAX DUR** sur 33 candidats serrés. Les méthodes établies (TD-MPC, MuZero, MPPI,
CEM) ne le feraient probablement jamais — moyennes pondérées, softmax, comptages de visites, précisément
parce qu'un argmax dur est catastrophiquement fragile au bruit près des égalités. À vérifier.

## Ce qu'il faut chercher

1. **TD-MPC / TD-MPC2** — comment scorent-ils une trajectoire imaginée ? (récompenses accumulées +
   valeur terminale ? ensembles de critiques pour réduire l'erreur ? agrégation douce type MPPI ?)
2. **MuZero / MCTS avec value net** — comment évitent-ils la fragilité de l'argmax ?
3. **Dreamer (v1/v2/v3)** — pourquoi apprennent-ils un ACTEUR plutôt que de faire une recherche
   explicite dans l'imagination ? Est-ce précisément pour contourner ce problème ?
4. **LeCun, « A Path Towards Autonomous Machine Intelligence »** — que prescrit-il EXACTEMENT pour
   le module de coût, le critique entraînable, et la façon dont l'acteur les optimise ? Le projet
   s'en réclame (`docs/audit_lecun_2026-07-06.md`) — vérifier qu'on le respecte réellement.
5. **Y a-t-il un traitement EXPLICITE du problème « erreur de la valeur ≫ écart entre candidats » ?**
   Termes à essayer : value approximation error in MPC, argmax bias, action-gap, ranking losses for
   value functions, preference-based value learning, soft/entropy-regularized planning.

## Livrable attendu (et rien d'autre)

**Pas un essai.** Trois ou quatre **changements de design concrets et testables**, classés par
(gain attendu ÷ coût), chacun accompagné de :
- ce que la littérature dit précisément (avec la source),
- en quoi notre implémentation actuelle en diverge (avec le fichier:ligne),
- **le test GRATUIT qui le valide ou le tue AVANT tout run coûteux.**

Les sondes gratuites existent déjà et tournent en 2 minutes — les réutiliser, ne pas en réécrire :
- `diagnostics/diag_critic_landscape.py` — dispersion des notes, marge relative, regret vs l'oracle
- `diagnostics/diag_metabolic_ceiling.py` — le plafond épars est MÉTABOLIQUE (monde marginal :
  portée soutenable 4.0 m contre des spawns à 2-8 m → 2/3 des vies condamnées par l'arithmétique,
  quelle que soit la décision). **Ne JAMAIS juger le critique sur la survie médiane** — le juger sur
  le FORAGE (repas + boissons).
- l'oracle `SYLVAN_CRITIC_ORACLE=1` — plafond atteignable par une valeur parfaite dans la fente actuelle
- `SYLVAN_CRITIC_ALWAYS=1` — fait décider le critique 100 % du temps (défaut OFF : il ne note que ~3 %
  des replans en épars, le reste tombe sur la formule codée-main)

## Discipline (règles du projet, non négociables)

- Diagnostiquer GRATUITEMENT avant tout run cher ; critères de succès et de KILL écrits AVANT.
- Ne PAS déplacer les poteaux : un critère écrit d'avance qui échoue est un échec, on le dit.
- Ne pas masquer une lacune de capacité en relâchant une métrique.
- Collecte critique = SÉQUENTIELLE (deux runs Godot/planner en parallèle déconnectent le serveur).
- Ne JAMAIS stager `godot/scripts/main.gd` ni `godot/scripts/ui/` (chantier HUD de l'owner).
- Tenir `tools/archi_hud/architecture.json` à jour DANS LE MÊME COMMIT ; la carte ne ment jamais.
- Commits Conventional en anglais, scopés, sans attribution IA.

## Non commité à reprendre

`python/scripts/train_survival_critic.py` (mode `--labels analytic` = distillation, + métrique R²)
et `python/sylvan/control/planning/command_planner.py` (sonde oracle `SYLVAN_CRITIC_ORACLE`).
Les deux sont écrits, testés, et portent les résultats ci-dessus. À committer ou à jeter selon ce
que la recherche conclut.
