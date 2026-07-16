# Prompt — chantier « petit H-JEPA » : étage waypoint à 2 niveaux

> Copier-coller le bloc ci-dessous comme premier message d'une session fraîche.
> Écrit le 2026-07-16, à la fin de la session qui a produit le verdict d'élimination + la recherche.

---

Sylvan — chantier d'IMPLÉMENTATION : l'étage waypoint (« petit H-JEPA », 2 niveaux). Branche
`feat/critic-clean-foundations` (dernier commit de référence : recherche + spec commitées le 2026-07-16).
Venv `env_pytorch_3.12`, CPU, `PYTHONPATH=python`, depuis la racine.

LIRE D'ABORD (dans cet ordre, puis re-checker orphelins `pgrep -xc godot`, git log, disque ~4-5 G) :
1. **`docs/recherche_hjepa_waypoint.md`** — LE document : pourquoi cet étage, ce que licencie LeCun, les
   design-choices éprouvés (tableau trans-systèmes), le croisement avec nos faits mesurés, et la **spec §6**.
2. `docs/etat_critique.md` — l'état complet du chantier danger (verdict d'élimination §, gates, échafaudages).
3. `memory/MEMORY.md` + `memory/sylvan-mode1-build.md` (entrées 2026-07-15/16).

## Le problème, MESURÉ (ne pas re-découvrir)

Le monde-danger est validé : zone verte létale+perceptible+localisable (WM 3-slots
`data/checkpoints/wm_objcentric_kin_haz`, zéro retrain). Une entité aveugle y meurt 7-8/12 et le danger
garde la bouffe (15 repas → ~5). **Six juges codés-main ont été éliminés** (répulsion ×3, rêve ×2, détour
binaire, détour gradué, centre corrigé, mur-vert ×2) : AUCUN ne casse le troc évitement↔repas, parce que
**la fente MPC (33 arcs myopes à commande constante, ~0.8 m, replan glouton) ne peut pas COMPOSER
« contourne le gardien puis mange », quel que soit le score** (trace 88 replans : gradients 6-37 pas vs
spreads 300-500 ; minima locaux canoniques Koren-Borenstein). Et l'écart d'action à cet étage (1e-5) a déjà
tué le critique appris. Les DEUX murs se réparent au même endroit : UN ÉTAGE AU-DESSUS.

## Le design à implémenter (spec complète : recherche §6 — la respecter, elle est sourcée)

**Étage haut v0 (serveur, échafaudage DÉCLARÉ)** — à chaque décision (spawn / waypoint atteint / timeout /
changement de cible) : candidats = cible directe + anneau de 6-8 waypoints (R≈2-3 m, positions au sol) ;
score = ligne entité→wp dégagée de vert-proche (rétine brute 36 rayons, style mur-vert) + ligne wp→cible
dégagée + longueur totale → queue de survie analytique ; **COMMIT** du gagnant comme cible du niveau bas
(mécanisme override existant : cf `food_override`/`_apply_search` dans `serve_planner_command.py`) jusqu'à
atteinte (~0.9 m) ou timeout (~150-200 pas), avec hystérésis. **PAS de reconstruction du danger**
(pas de centre/rayon estimés — leçon des 6 échecs) : seulement « cette direction est-elle verte-proche ? ».
**Étage bas : le planner actuel INTOUCHÉ** (`command_planner.py` ne change pas ; il reçoit juste une cible).

Pourquoi ces choix (tout est sourcé dans la recherche) : bas-niveau GELÉ = SoRB/Puppeteer/PRM-RL (et tue la
non-stationnarité HIRO) ; waypoints spatiaux bruts = HIRO/PRM-RL (navigation) ; cadence grossière + commit =
K/k 8-50 partout, plus fin NUIT (HIQL) ; mode discret = TangentBug (l'échappée d'un minimum local est un
CHANGEMENT DE MODE, pas un blend) ; peu d'options très différentes = là où un critique PEUT discriminer
(HIQL fig.8 — c'est l'étage où le critique renaîtra).

## Gates PRÉ-ENREGISTRÉS (écrits avant — on ne déplace pas les poteaux)

- **G0 non-régression monde plat** : sans danger (SYLVAN_HAZARD_COUNT=0), forage ≥ baseline (le candidat
  « direct » doit gagner trivialement). Si G0 casse → bug de l'étage, pas de tuning.
- **G1 monde-danger** : **repas > 10 ET morts-danger ≤ 2** sur 12 vies (réf sans danger : 15 repas / 16
  boissons ; meilleur juge plat : 9 repas / 6 morts ; aveugle : 5 repas / 6-8 morts). Mesure :
  `diagnostics/diag_hazard_gate.py` (survie + repas/boissons déjà parsés). Harnais :
  `scripts/collect_critic_corpus_kin.sh` (env : WM_CKPT=wm 3-slots, SYLVAN_HAZARD_COUNT=1, HORIZON exposé).
- Diagnostic AVANT tuning si G1 échoue : trace (le pattern SYLVAN_HAZARD_DEBUG existe dans
  `command_planner.py` — en faire l'équivalent à l'étage waypoint : quel wp choisi, pourquoi, atteint ou
  timeout) — LOCALISER, ne pas deviner (leçon de la journée d'élimination).

## Après G1 (ne PAS commencer par ça)

L'étape apprise : le **critique note les waypoints** (gros écarts → apprenable — c'est LA réhabilitation du
critique) ; corpus contrasté GRATUIT en variant les waypoints (l'exploration au bon étage — leçon Director :
bonus au manager, jamais au worker) ; gate résidu `train_survival_critic --labels residual` (+0.10 R², CV
4 plis) déjà écrit. À terme : remplacer proposeur/scoreur main par l'appris (aspiration LeCun §4.7).

## Discipline (règles du projet, non négociables)

- Diagnostiquer GRATUITEMENT avant tout run cher ; critères de succès/KILL écrits AVANT ; un juge/design qui
  échoue = négatif informatif à COMMITER, pas à maquiller.
- Mesurer LE BUT (repas+boissons, morts-danger), jamais la survie médiane seule (plafond métabolique) ni la
  cause de mort seule (leçon : « éviter » peut juste déplacer la cause).
- Collecte = SÉQUENTIELLE (2 serveurs en parallèle se déconnectent). Runs en background + `/sylvan-kill` +
  vérif 0 orphelin. ⚠️ zsh ne word-split PAS `$VAR` (piège vécu : paires `"a:b"` + `${CFG%%:*}`).
- Ne JAMAIS stager `godot/scripts/main.gd` ni `godot/scripts/ui/` (chantier HUD owner). ⚠️ Les 4 hooks
  hazard vivent dans main.gd NON COMMITÉS — vérifier `grep -c hazard_manager godot/scripts/main.gd` ≥ 7
  avant tout run ; s'ils ont disparu (checkout/stash), les reposer (cf `docs/etat_critique.md`).
- Tenir `tools/archi_hud/architecture.json` à jour DANS LE MÊME COMMIT (module `monde_hazard` +, si créé,
  un module étage-waypoint ; états pur/partiel/échafaudage/manquant ; validateur avant commit).
- Commits Conventional en anglais, scopés, sans attribution IA. L'échafaudage se DÉCLARE (état + env opt-in
  défaut OFF), ne se déguise pas.

## Fichiers probablement touchés

`python/scripts/serve_planner_command.py` (l'étage waypoint vit AU SERVEUR : décision, commit, override de
cible — réutiliser `_apply_search`/`food_override` ; NE PAS toucher `command_planner.py` sauf debug-print) ;
peut-être un petit module `python/sylvan/control/waypoint_layer.py` ; `diagnostics/diag_hazard_gate.py`
(réutiliser tel quel) ; `tools/archi_hud/architecture.json`.
