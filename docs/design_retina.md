# Design — RÉTINE (perception apprise) : décision technique + scope à faire

> **STATUT (2026-06-18 nuit-2) : ✅ LIVRÉ ET VALIDÉ.** La rétine est implémentée (étages 0→1→2), la perception
> est 100 % apprise, l'oracle radar est mort. Détail d'exécution + résultats : `docs/scope_retina.md`,
> `memory/sylvan-retina-decision.md`, schéma `docs/schema_jepa_sylvan.md`. Résumé : étage 0 raycast
> (`perception.gd::retina`), étage 1 tête apprise (`data/checkpoints/retina_head/`, foraging à parité), étage 2
> WM consomme la rétine (`wm_command_hex_retina_jepa_v2`, eff_rank 14, open-loop 0.22 m, foraging 980 ≥ oracle 965).
> Feu vert 🅑 (énergie food-aware, `diag_latent_foodaware.py`). Le doc ci-dessous = la DÉCISION d'origine (archive).

## Pourquoi la rétine (le cap)
Aujourd'hui la perception est un **ORACLE** : le radar 12-secteurs souffle directement à l'agent la **direction
et la distance de la bouffe**. Le WM ne prédit que des *features pré-mâchées*. Or le JEPA, son cœur même, c'est
**apprendre une représentation à partir de la perception BRUTE**. Tant que la perception est un oracle, « JEPA »
est un titre généreux, pas une réalité — c'est la plus grosse dette d'honnêteté du projet (cf CLAUDE.md §2).
La rétine fait **émerger** « rouge = nourriture, bleu = eau » de l'apprentissage au lieu de le coder.

## Technique CHOISIE : rétine RAYCAST profondeur + couleur (1D)
- **N rayons** lancés depuis la tête de l'agent ; chaque rayon renvoie **(profondeur normalisée + couleur RGB)** de
  ce qu'il touche. Point de départ proposé : **~36 rayons sur 360° × 4 (depth, R, G, B) ≈ 144 dims** (même ordre de
  grandeur que l'obs actuelle → rentre dans l'encodeur MLP/RSSM, CPU-friendly, MPC reste temps réel).
- **Pourquoi pas les pixels (image 64×64 + CNN)** : pas tué par le MPC (l'encodeur ne tourne qu'1× par replan, le
  rollout 120×horizon est en latent), MAIS un CNN est plus lourd, plus data-hungry (~200 épisodes seulement), et
  inutile à ce stade. Les rayons couleur forcent déjà un vrai apprentissage de représentation (couleur→sens,
  géométrie→position) à coût et risque maîtrisés. L'infra raycast existe déjà (le radar en dérive) → risque réduit.
- **Ce qui rend ça une VRAIE rétine et pas un nouvel oracle** : on ne donne plus « bouffe = secteur 7 ». On donne du
  brut (« rayon 12 → distance 3.2, couleur 0.9/0.3/0.2 »). L'agent doit APPRENDRE la couleur→ressource ET localiser.

## Sous-décision (le vrai défi) : comment le planner sait où aller ?
Aujourd'hui le planner appelle `food_xz_from_radar(oracle)`. Avec des rayons bruts :
- **🅐 Minimal viable (COMMENCER PAR LÀ)** : une **tête de perception APPRISE** : `rayons → position estimée
  bouffe/eau`, qui **remplace l'oracle géométrique**. La perception devient réellement apprise, mais on **garde la
  structure du planner**. Un seul gros changement isolé → testable, falsifiable.
- **🅑 JEPA-pur (plus tard)** : planifier dans le **latent** ; le coût est sur l'état latent (« énergie future
  prédite »), plus de coordonnées explicites. Le plus pur, mais gros changement → après que 🅐 ait prouvé que la
  perception apprise marche.

## Ce qui change (à détailler dans le scope)
- `godot/scripts/agent/perception.gd` : nouvelle fonction rétine raycast (depth+RGB par rayon) ; le radar-oracle
  food/water n'alimente PLUS l'obs du WM (peut rester comme *sonde de debug* seulement).
- **Dims** : obs WM 145 → (132 proprio + 144 rétine + 1 énergie) ≈ 277 (à figer). Resync OBLIGATOIRE :
  `constants.py`, `config.py`, `observation_builder.gd`, `command_wm.py`, `wm_dataset.py`, `symmetry.py`,
  `serve_planner_command.py`. (Voir CLAUDE.md « Ne JAMAIS changer ces dims sans synchroniser ».)
- WM/encodeur : ré-appris sur la rétine (la représentation visuelle doit s'apprendre). **Re-collecte de données
  AVEC la rétine** (régime propre hexapode) + retrain. Garder la discipline JEPA (cosine + VICReg) acquise.
- Planner (`command_planner.py`) : option 🅐 = brancher la tête de perception apprise à la place de
  `food_xz_from_radar`. NE PAS toucher au MPC brute-force (120 itérations) — voir ci-dessous.

## CE QU'ON NE TOUCHE PAS MAINTENANT
- **Le MPC brute-force (120 candidats × horizon, DÉJÀ en latent)** reste tel quel. Il sera remplacé par **Mode-1**
  (politique apprise en une passe, + recherche gardée en fallback pour les cas durs = System 1/2) — mais **EN
  DERNIER**, une fois le système mûr (sinon on distille un système qui change deux fois). Étape intermédiaire
  possible : recherche guidée (CEM/MPPI ou gradient à travers le WM différentiable) pour moins de rollouts.
- **ORDRE GLOBAL** : RÉTINE → décision foresighted (critique appris / horizon long, le vrai fix de la robustesse
  multi-pulsions myope) → Mode-1 (amortit le tout). Rien n'est refait si on respecte cet ordre.

## JALON FALSIFIABLE (rétine)
> **« L'agent navigue jusqu'à la bouffe en utilisant UNIQUEMENT les rayons couleur bruts, sans qu'on lui dise jamais
> où elle est. »** Si ça marche → perception réellement apprise, oracle mort, vrai JEPA. Si ça cale → on saura
> exactement où (perception ? localisation apprise ? WM ?).

## DISCIPLINE (rappel CLAUDE.md §1 + §2)
Scope GRATUIT d'abord (zéro entraînement) : version minimale viable, dims exactes, fichiers à changer, plan de
re-collecte, critères SUCCÈS/KILL pré-écrits. Gater le cher (collecte+retrain WM) derrière le pas-cher. Garder le
critère HONNÊTE (pas d'oracle déguisé, pas de tolérance gonflée pour masquer un échec).
