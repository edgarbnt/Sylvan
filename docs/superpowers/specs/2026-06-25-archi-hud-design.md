# Archi-HUD — carte vivante de l'architecture Sylvan

**Date :** 2026-06-25
**But :** un schéma temps-réel, dans le navigateur, qui montre en permanence les modules
de l'architecture (ce qu'ils sont + leur état de pureté JEPA), et qui s'anime quand un run
foraging tourne. Rendre l'archi (compliquée) lisible d'un coup d'œil et toujours à jour.

## Problème

L'architecture Sylvan est devenue conceptuellement dense (pipeline 3 couches + modules JEPA,
certains purs, un échafaudage à résorber, des trous = futurs chantiers). Aucune vue d'ensemble
ne dit, à un instant donné : **quels modules existent, ce qu'ils sont, lesquels sont solides /
béquilles / manquants, et sur quoi on bosse.** Une doc statique dériverait et mentirait (anti-
pattern central du projet : §2 « ne pas masquer le vrai problème »).

## Décisions actées (issues du brainstorming)

1. **Hybride** : carte conceptuelle permanente + overlay runtime qui anime les modules quand un
   run tourne.
2. **Surface** : page web locale (HTML/JS) ouverte dans le navigateur.
3. **Data-driven, PAS page libre** : le HTML/JS est un moteur de rendu bête et stable ; toute
   l'info d'archi vit dans des JSON. Raisons : le mode live exige des poignées stables (`id`) pour
   binder les valeurs ; une page réécrite à la main dériverait, casserait le live, et ne serait pas
   diffable. La liberté d'expression est préservée par un champ `detail` libre par module.
4. **Source de vérité** = `architecture.json` (maintenu par l'assistant, versionné, avec ancres
   code + `live_field`) ; les valeurs runtime vivent dans un `live.json` séparé écrit par le run.

## Architecture du HUD (séparation des rôles)

```
tools/archi_hud/
  index.html          # structure de la page
  app.js              # moteur de rendu (bête, stable — ne contient AUCUNE info d'archi)
  style.css           # styles + couleurs d'état
  architecture.json   # LA donnée conceptuelle (modules, états, détails, ancres, live_field)
data/hud/live.json    # écrit par le run, lu en boucle par la page (gitignore)
voir_archi.sh         # sert le dossier en local + ouvre le navigateur
```

- `index.html` + `app.js` + `style.css` : écrits une fois, quasi jamais retouchés. `app.js` fetch
  `architecture.json`, dessine les boîtes/flèches, gère le clic-détail, et poll `live.json`.
- `architecture.json` : **seule** chose éditée quand l'archi évolue (changer un `etat`, ajouter un
  module). Micro-édition diffable → l'historique git raconte l'évolution de l'archi.
- `live.json` : écrit par le serveur planner pendant un run ; absent/périmé → mode conceptuel.

## Modèle de données

### `architecture.json`
```json
{
  "focus": "transport_slot",
  "modules": [
    {
      "id": "transport_slot",
      "etat": "echafaudage",
      "titre": "Transport du slot",
      "couche": "planner",
      "quoi": "L'objet est transporté à la main par trigo dans le planner (devrait vivre dans le WM)",
      "detail": "Texte/HTML libre : preuve, citation de code, pourquoi c'est une béquille…",
      "code": "command_planner.py:204-218",
      "live_field": "min_dist",
      "depends_on": ["world_model", "perception_slot"]
    }
  ]
}
```

- `etat` ∈ `{ pur, partiel, echafaudage, manquant }` → pilote la couleur/bordure.
- `focus` = l'`id` du module « ◀ ON BOSSE ICI » (halo/badge).
- `code` = `fichier:ligne(s)` (relatif à `python/sylvan/...` ou racine) → cliquable + vérifiable.
- `live_field` = nom de la clé dans `live.json` qui anime ce nœud (absent = nœud non-animé).
- `depends_on` = ids amont → trace les flèches de flux.

### `live.json` (écrit par le run, ~toutes les 10 steps)
```json
{
  "ts": 0, "episode": 3, "step": 412,
  "fields": { "bearing": 32, "disp_ok": true, "done_prob": 0.02,
              "min_dist": 1.8, "command": [0.7, -0.3],
              "energy": 58, "thirst": 100, "fwd_v": 0.27, "yaw": -144 }
}
```
- `ts` = compteur monotone (PAS d'horloge murale — interdit côté scripts/déterminisme ; un simple
  entier incrémenté suffit pour détecter « périmé »). La page garde le dernier `ts` vu : inchangé
  pendant >N polls → run terminé → repasse en mode conceptuel.

## Liste des modules (carte v1)

Pipeline du corps vers le cerveau + modules JEPA manquants. États au 2026-06-25 :

| id | Module | État | Rôle JEPA | Ancre code |
|---|---|---|---|---|
| `corps_cpg` | Corps / CPG | pur | substrat moteur (by-construction) | `hexapod_v2/policy_best.pt` |
| `residu_ppo` | Résidu PPO | pur | actor bas-niveau (équilibre/perf) | `hexapod_v2/policy_best.pt` |
| `world_model` | World Model | partiel | World Model (recon droppée ✅ ; déterministe + sans incertitude) | `wm_rich_fidele_sym_jepa/wm_best.pt` |
| `perception_slot` | Perception / Slot | pur | Perception (slot_head, label-free) | `models/slot_head.py` |
| `transport_slot` | Transport du slot | echafaudage | (devrait être DANS le WM) trigo à la main | `command_planner.py:204-218` |
| `cout_planner` | Coût planner | pur | Cost intrinsèque (`-min_dist`) | `command_planner.py:226` |
| `planner_mpc` | Planner MPC | pur | actor haut-niveau (recherche command-space) | `command_planner.py:166` |
| `drives` | Drives faim/soif | pur | pulsions (propriété du corps) | `command_planner.py:249` |
| `memoire_spatiale` | Mémoire spatiale | manquant | Short-term memory (permanence d'objet) | — |
| `critique_appris` | Critique appris | manquant | Critic (V(slot)=gain futur) | — |
| `configurator` | Configurator | manquant | Configurator (arbitrage/attention) | — |

`focus` initial = `transport_slot` (chantier courant : internaliser le slot dans le WM).

## Rendu & interaction

- **Boîtes empilées** (le pipeline) reliées par des **flèches de flux** dérivées de `depends_on`
  (perception → WM → transport → coût → planner → commande → corps).
- **Couleur/bordure = état** : vert (pur), jaune (partiel), orange (échafaudage), rouge pointillé
  (manquant).
- Chaque boîte : **titre + ligne « quoi »**. Le module `focus` porte un **halo + badge ◀ ON BOSSE ICI**.
- **Clic → panneau latéral** : déplie `detail` + lien `code` (`fichier:ligne`).
- **Légende** des états visible en permanence.

## Mode live

- La page poll `data/hud/live.json` toutes les ~0,5 s.
- Pour chaque module ayant un `live_field` présent dans `live.fields`, le nœud **pulse** et affiche
  la valeur → on *voit* le flux traverser l'archi pendant le run.
- `live.json` absent ou `ts` figé sur >N polls → **mode conceptuel** (aucune valeur affichée). La
  carte ne ment jamais sur un run qui ne tourne pas.
- Écriture : un petit hook dans `serve_planner_command` (il a déjà bearing/min_dist/command/…),
  complété par les champs côté Godot (energy/thirst/fwd_v/yaw) si simple ; sinon v1 = sous-ensemble
  disponible côté planner, le reste ajouté plus tard.

## Hors-scope (YAGNI v1)

- Pas de zoom/pan, pas d'historique temporel, pas de multi-run simultané.
- Pas d'auto-dérivation de l'état depuis le code (le « pur vs échafaudage » est sémantique → resterait
  à la main de toute façon).
- Pas de framework JS (vanilla — le rendu est simple, zéro build).

## Critères de succès

1. `bash voir_archi.sh` ouvre une page montrant les 11 modules avec le bon état (couleur) et le
   badge focus sur `transport_slot`.
2. Cliquer un module ouvre son `detail` + lien code.
3. Changer un `etat` dans `architecture.json` + recharger → la couleur change (zéro édition HTML/JS).
4. Pendant un run foraging, au moins les nœuds câblés côté planner (perception, transport, coût,
   planner) pulsent avec des valeurs plausibles ; à l'arrêt du run, retour automatique au mode
   conceptuel.

## Risques / notes

- **Servir en local** : `voir_archi.sh` lance un petit serveur statique (python `http.server`) car
  `fetch()` d'un `file://` est bloqué par le navigateur. Pas d'ouverture `file://` directe.
- **Pas d'horloge** dans les scripts (règle projet) → `live.json.ts` = entier monotone, pas un
  timestamp.
- Le HUD est un **méta-outil** : il ne touche jamais au pipeline ALife, seulement il le lit/le décrit.
