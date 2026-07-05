# Recette : ajouter une pulsion (quoi toucher, quoi ne JAMAIS toucher)

_2026-07-03. Consolide ce qui était éparpillé (principe de travail n°3, `design_mode1.md`,
`design_wm_factorise.md`, mémoire). C'est le **contrat de scalabilité** de l'archi : si ajouter
une pulsion exige plus que ce qui est listé ici, c'est un signal d'alerte §3 (mauvais étage)._

## 1. Mission
Qu'ajouter une pulsion (ex. une 3ᵉ : abri, sel, chaleur…) soit un geste **local et borné** —
jamais une refonte. Le test falsifiable de ce contrat = **Gate-S** (`design_mode1.md`) :
politique/planner GELÉS + pulsion jamais vue → l'entité la gère sans retrain des étages généraux.

## 2. À lire d'abord
- `les règles du projet` §3 (substrat lent vs pulsions rapides — le principe).
- `python/sylvan/control/mode1/obs.py:20` (`build_tokens` : le token-pulsion concret).
- `python/sylvan/control/planning/command_planner.py` (`_survival_extension` + branche multi-ressource).

## 3. Ce qu'on TOUCHE (coût borné, par pièce)

| Pièce | Geste | Coût |
|---|---|---|
| **Corps (Godot)** | déclarer le drive (drain passif + refill au contact + rayon) et l'objet perceptible (collider + meta `retina_color`, couche 8) | config ; `perception.gd:20` dit déjà « futur objet = poser collider+meta, rien à re-coder » |
| **Token de pulsion** (politique/valeur Mode-1) | 1 token `[niveau, valence, 36 profondeurs couleur-gatées]` de plus dans `build_tokens` — encodeur partagé drive-symétrique | quelques lignes ; **zéro retrain** visé (à prouver par Gate-S) |
| **Tête de lecture** (slot-encoder) | 1 petite tête label-free « où est l'objet de cette couleur » sur le **latent GELÉ** (chemin prouvé : `build_slot_channel.py`, `CommandWorldModel(slot_resources=N)`) | minutes d'entraînement, ne touche NI le WM NI les têtes existantes |
| **Coût du planner** | la ressource entre dans le rollout de survie (drain/refill = propriétés du corps) | config + généralisation N-ressources de l'alternance (voir §6) |

Cible de pureté (au-delà de la recette) : une tête de lecture **paramétrée par la requête-couleur**
(analogue des tokens drive-symétriques) → ressource nouvelle = *zéro* entraînement, juste une
requête ; et le lien « telle perception soulage tel drive » **appris la nuit** sur le drive vécu.

## 4. Ce qu'on ne touche JAMAIS (et pourquoi)

| Pièce | Pourquoi intouchable |
|---|---|
| **WM (substrat)** | GELÉ tant qu'on n'ajoute pas un *sens* nouveau. Le latent porte déjà toute la perception rétinienne (couleurs comprises) — prouvé pour l'eau par gate G1 (`diag_wm_water_latent.py`, R² 0.53 sans retrain). 🚨 Un `--w-<ressource>` sur le WM = raccourci interdit (§3). |
| **Transport géométrique** | Propriété de l'ESPACE, pas de l'objet : « quand je bouge, où passe l'objet dans mon référentiel » vaut pour bouffe, eau, n'importe quoi (`slot_calib=(1,-1,-1)`, leçon 2026-06-25 : c'est une géométrie, PAS un truc à fitter). |
| **CPG + résidu moteur** | Agnostiques à la ressource par construction (BLUEPRINT §14). |
| **Encodeur de la politique drive-symétrique** | Partagé entre tokens — c'est précisément ce qui rend le token branchable. |

## 5. État ACTUEL vs cible (honnêteté §2 — ce qui reste impur aujourd'hui)

| Pièce | Cible (recette) | Aujourd'hui |
|---|---|---|
| Lecture bouffe | slot appris dans le WM | ✅ `wm_objcentric_s1`, `out["slot"]` |
| Lecture eau | slot-2 bleu appris | ❌ radar-oracle EMA (`serve_planner_command.py`, `vision_water`) — **échafaudage à résorber** (build = miroir du slot rouge) |
| Coût survie | pas-vécus simulés | ✅ implémenté (`SYLVAN_PLANNER_COST=survival`, gate B0) mais **pas promu** (A/B en cours) ; dynamique drives analytique **flaggée** — pureté finale = tête drive-dynamics APPRISE (3ᵉ verrou, `design_mode1_pivot_mode2.md` §5b) |
| Token pulsion | N tokens drive-symétriques | ✅ faim+soif (`build_tokens`) ; Gate-S (3ᵉ pulsion sans retrain) **jamais passé** — le contrat n'est pas encore prouvé |

## 6. Limite connue de la recette (ne pas la survendre)
- `_survival_extension` est écrite pour **2 ressources** (alternance A↔B, 2 ordres simulés). À N
  ressources, l'ordre de visite devient combinatoire → généralisation à écrire (heuristique
  gloutonne « la plus urgente d'abord » ou petite recherche) — c'est un geste sur le COÛT, pas sur
  le substrat, donc conforme à la recette, mais ce n'est pas « gratuit ».
- Le no-retrain est prouvé dans la famille « approcher-consommer » (valence + ; l'évitement =
  valence −, extension plausible non prouvée). Une pulsion de nature radicalement autre
  (température ambiante, sommeil) = extension du contrat, pas couverte.

## 7. Critère de succès = le BUT
**Gate-S** : geler politique/planner, introduire une 3ᵉ pulsion + son objet (couleur nouvelle),
appliquer UNIQUEMENT les gestes du §3 → l'entité survit multi-3-pulsions ≥ baseline 2-pulsions
(pas-vécus, multi-seed). Tout retrain d'une pièce du §4 pour y arriver = ÉCHEC du contrat.
