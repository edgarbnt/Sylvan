# Task 2 — Rapport (reconstruit par le CONTRÔLEUR après interruption de session)

> ⚠️ Le rapport in-process du subagent implémenteur a été perdu (la session a été interrompue
> pendant Task 2). Le fichier qui était ici avant était STALE (un rapport d'un build antérieur,
> « mémoire spatiale » / egomotion_head — sans rapport avec Mode-1). Ce rapport-ci ne contient que
> des faits VÉRIFIÉS par le contrôleur (git + inspection directe des données). Le mécanisme exact du
> code est à juger depuis le diff par le reviewer.

## Commit
`ffb7278` — "Mode-1 Task 2 : collecte BC planner multi-pulsions (serve_planner_command + Godot)"

Fichiers changés (172 insertions) :
- `collect_mode1_bc.sh` (nouveau, 95) — lanceur de collecte.
- `python/scripts/serve_planner_command.py` (+47) — logger BC (attendu : gaté sur `SYLVAN_BC_LOG`).
- `godot/scripts/control/policy_player.gd` (+25) — déviation : Godot touché (attendu : gaté / additif).
- `godot/scripts/main.gd` (+5) — déviation : Godot touché.

NB : l'approche recommandée était « server-side uniquement, 0 changement Godot ». L'implémenteur a
touché Godot. La contrainte « 0 changement Godot » est une contrainte du RUNTIME Mode-1 (déploiement) ;
toucher Godot pour la COLLECTE est tolérable **uniquement si totalement gaté/non-régressant** (les runs
normaux planner/foraging/survie ne doivent PAS changer quand `SYLVAN_BC_LOG` est absent). À VÉRIFIER au review.

## Données VÉRIFIÉES par le contrôleur (inspection directe des buffers)
- `data/replay_buffer/mode1_bc_a` : 12 épisodes, 29 655 lignes.
- `data/replay_buffer/mode1_bc_b` : 12 épisodes, 26 086 lignes. (Total ~55.7k transitions ≥ cible 15k.)
- Forme par ligne : `obs.proprio`=132, `wm.retina0`=144, `obs.thirst` présent, `wm.cmd`=[vx,ω]
  (commandes RÉELLES du planner, ex. [0.6, -0.6], [0.75, 0.6]) → labels BC corrects.
- **Eau-dans-rétine CONFIRMÉE** (vérif Gate-0 différée) : blue-ray hits 206 175 (a) / 183 151 (b) ;
  red-ray hits 124 969 / 101 378. La rétine rend bien food (rouge) ET eau (bleu).

## Verdict contrôleur
La COLLECTE FONCTIONNE (données valides, volume suffisant, eau-dans-rétine OK). Le review doit se
concentrer sur la **NON-RÉGRESSION** (logger BC OFF par défaut → comportement byte-identique des runs
normaux, Y COMPRIS les changements Godot) et la propreté du code.

---

## Fix review — 2026-06-28

### Fix 1 — Gate `send_reset()` sur SYLVAN_BC_LOG (non-régression, `main.gd`)

**Diff (ligne 646-648, avant → après) :**
```
- if _cpg_planner and policy_player.is_server_ready():
+ if _cpg_planner and policy_player.is_server_ready() and OS.get_environment("SYLVAN_BC_LOG") != "":
      policy_player.send_reset()
```
Sans ce gate, `{"reset":true}` était envoyé au serveur planner à chaque épisode dans TOUS les runs
planner normaux (foraging, survie, A→B nav) — ce qui réinitialisait l'EMA du slot côté serveur à
chaque épisode, changeant le comportement des runs qui n'utilisent pas la collecte BC.

### Fix 2 — Env collecte alignée sur baseline (`collect_mode1_bc.sh`)

Deltas old → new :
| Variable | Avant (bugué) | Après (aligné baseline) |
|---|---|---|
| `SYLVAN_WATER_COUNT` | `8` | `5` |
| `set` mode | `set +e` (global) | `set -uo pipefail` |
| pkill / kill lines | pas de `|| true` | `|| true` ajouté |

`SYLVAN_FOOD_COUNT=5` était déjà présent. `SYLVAN_PLANNER_HEADING_W=2.0`, `SYLVAN_PLANNER_URGENCY_W=6.0`,
le bloc régime-propre, `SYLVAN_CPG_PLANNER=1`, `SYLVAN_RETINA_PLANNER=1`, `SYLVAN_WM_USE_RETINA=1`,
les drains et rayons — tous corrects et alignés avec `baseline_multidrive_slot.sh`.

### Fix 3 — Error handling (`collect_mode1_bc.sh`)

`set +e` global → `set -uo pipefail` : un crash Godot, checkpoint absent ou port non ouvert remonte
maintenant en erreur au lieu d'être silencieusement ignoré. `|| true` ajouté uniquement sur les 3
lignes pkill/kill (comportement légitimement non-fatal).

### Vérification syntaxe
`bash -n collect_mode1_bc.sh` → SYNTAX OK
