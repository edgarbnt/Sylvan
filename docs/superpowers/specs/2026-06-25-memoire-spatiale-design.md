# Mémoire spatiale — design (échafaudage-first)

> Statut : design validé par l'owner (2026-06-25, « go pour tout ça »). Route choisie = **échafaudage-first**.
> Chantier focus de la carte vivante (`tools/archi_hud/architecture.json`, focus = `memoire_spatiale`).
> Suite de la résolution « slot internalisé dans le WM » (forager vivant = `wm_objcentric_s1`,
> `run_forage_wmslot.sh`). Mémoire auto : `sylvan-objectcentric-pur.md`, `sylvan-wm-factorise-design.md`.

## 0. But (north-star de l'incrément)

Que l'entité **se souvienne où est un objet quand elle ne le voit plus** : continuer à connaître sa position
ego par **dead-reckoning** (transport par l'ego-motion réelle) au lieu de la ré-encoder depuis une perception
vide, puis **re-grounder** quand l'objet est re-perçu. C'est le SUBSTRAT qui rendra légitime, plus tard, le
passage de la vision 360° à un **cône avant** (« ne pas voir derrière » devient le rôle de la mémoire, pas un bug).

Préséance des principes CLAUDE.md : §1 (gate gratuit décisif AVANT tout build/retrain), §2 (mesurer le BUT
honnête, jamais un proxy offline), §4 (chaque étape solide avant la suivante ; promouvoir seulement si
≥ baseline). Leçon « offline ≠ but » (le slot avait +0.68 offline et régressait closed-loop) → **le seul juge
final est le closed-loop** ; le test offline ne fait que GATER la dépense, il ne promeut rien.

## 1. Où vit la persistance (tranché)

- **L'ÉTAT (le belief) vit dans le SERVEUR, entre replans.** La mémoire est inter-replans par nature (elle doit
  survivre quand l'objet n'est pas perçu). Or `command_wm.rollout_open_loop` **ré-encode** le slot depuis la
  perception *courante* (`encode_slot(obs0)`) à chaque replan, et ne le dead-reckone qu'à l'intérieur de
  l'horizon *imaginé*. Le rollout ne peut donc PAS porter une mémoire cross-temps. Le serveur
  (`serve_planner_command.py`) est déjà stateful (`_ema_pos`, `_cmd`, `replan_every`) → c'est là que vit le belief.
- **L'OPÉRATEUR de transport vit déjà dans le WM** : `CommandWorldModel.transport_slot(slot, disp_real)`, calib
  GÉOMÉTRIQUE FIXE `(1,-1,-1)`. On le réutilise tel quel → apples-to-apples avec le forager vivant.
- Le belief devient simplement le **slot t0** fourni au planner quand l'objet n'est pas vu (override de
  `encode_slot(obs0)`). Le rollout continue de dead-reckoner en avant DANS l'imagination comme aujourd'hui.
- C'est **F-pure-3 appliqué au temps RÉEL** au lieu du temps imaginé (F-pure-3 : un slot dead-reckoné tient
  l'objet à +0.95 à travers une réorientation >60°).

Route = **échafaudage-first** (mirroir exact de l'échelle qui a marché pour le slot : codé-main → validé →
internalisé). L'internalisation dans le WM (mémoire émergente dans le hidden RSSM) est DIFFÉRÉE, après que la
mémoire-échafaudage est validée et que le cône avant existe.

## 2. Test gratuit DÉCISIF — `diag_slot_memory_drift.py` (AVANT tout build)

Pur offline, sur buffers existants. **Zéro collecte, zéro retrain.** C'est le gate §1 : il décide si on build.

**Données (confirmées présentes)** : `data/replay_buffer/retina_wm_a` + `retina_wm_b` (régime foraging, virages
réels). Champ `wm` par pas : `retina0` [144] (perception pour `encode_slot`), `food_rel0` [3] (vérité-terrain
ego de l'objet : x_right, z_fwd, flag), `torso0` [3] (pose torse → ego-motion inter-frame `torso0[i]→[i+1]`,
exactement ce que F2 a mesuré). Tenir `retina_head_a` en réserve.

**Mécanisme** : à un pas `k` « dernier-vu » (balayé sur plusieurs k par trajectoire), couper la perception.
Pour `t > k` : `belief(t) = transport_slot(belief(t-1), egomotion_réelle(t-1→t))` avec la calib FIXE `(1,-1,-1)`
= l'opérateur EXACT du WM vivant. `egomotion_réelle` = delta de pose torse du log (F2 : proprio→ego-motion
+0.98 → on ISOLE la persistance, pas l'estimation d'ego-motion).

**Deux variantes de `belief(k)`** (diagnostic fin) :
- (a) `belief(k) = encode_slot(retina_k)` — réaliste, inclut l'erreur de perception à t0.
- (b) `belief(k) = food_rel0(k)` — vérité-terrain, isole la dérive PURE du dead-reckon (borne haute).

**Baselines** :
- (i) belief **GELÉ** (pas de dead-reckon, slot collé à `belief(k)`) = « mémoire statique ».
- (ii) re-encode **aveugle** = ce que le live renvoie sans objet visible (bruit/zéro) = « pas de mémoire ».

**Métriques** vs N = pas-depuis-vu ∈ {5, 10, 20, 40} : **bearing MAE (°)** et **position MAE (m)**, agrégées
multi-trajectoires / multi-k / multi-seed (au moins les 2 buffers).

**SUCCÈS (pré-enregistré)** : le belief dead-reckoné garde **bearing MAE < ~20° ET position MAE < ~0.5 m
jusqu'à N ≈ 30-40**, ET **bat nettement le gelé** (le dead-reckoning APPORTE). Ancre : F-pure-3 +0.95 à >60°.

**KILL (pré-enregistré)** : **position MAE > ~1 m en < 15 pas aveugles**, OU dead-reckon ≈ gelé (n'apporte rien)
→ l'erreur d'ego-motion compound trop vite → mémoire infaisable avec cet opérateur → **STOP + escalade** (PAS
d'enchaînement de tweaks ; un négatif ici est INFORMATIF — il dit que le levier est l'ego-motion/calib, pas la mémoire).

Interprétation du diagnostic : si (b) dérive aussi → problème opérateur/calib/ego-motion ; si seul (a) dérive →
problème de perception à t0 (bord de champ), pas du transport.

## 3. Build (SEULEMENT si le test §2 passe) — `SlotMemory` côté serveur

Échafaudage propre dans `serve_planner_command.py` (comme le slot l'a d'abord été). À chaque tick réel :

1. **Dead-reckon** : `belief ← transport_slot(belief, egomotion_réelle_du_pas)`. Source d'ego-motion = le
   proprio (F2 : proprio→ego-motion +0.98). Choix d'implémentation tranché AU BUILD, sans run long : soit une
   petite tête `proprio→egomotion` (F2 dit trivialement apprenable), soit un calcul analytique depuis les
   vitesses linéaires/angulaires du proprio × dt. Pas d'oracle de pose monde.
2. **Perception + gate de saillance** : `encode_slot` + le gate de saillance du `slot_head`. Si **saillant**
   (objet vu) → **RE-GROUND** : la perception est précise (4.9°) → remplacement direct du belief (blend léger
   seulement si jitter mesuré problématique). Sinon → **garder** le belief dead-reckoné.
3. **Sortie** : le belief sert de slot t0 au planner (override de `encode_slot(obs0)` dans la branche WM-slot
   quand l'objet n'est pas vu). Le rollout dead-reckone en avant comme aujourd'hui.

**Non-régression gratuite** : sans occlusion (vision 360° actuelle), l'objet est re-grounded chaque tick → le
belief = la perception courante → comportement **byte-proche** du forager vivant. Filet intact.

## 4. Gate closed-loop (occlusion artificielle, vision encore 360°)

On ne peut pas encore occulter naturellement → on **masque** la perception : zéro les rayons rétine d'un objet
une fois qu'il a été vu (ou dans une zone d'angle paramétrable). Mesures :

- **Avec occlusion** : mémoire ≥ no-mémoire — atteindre un objet vu-puis-occulté là où le live actuel le perd.
  Réutiliser `diag_nav_ab_wmslot.sh` (engagement/approche min) + foraging A/B (`run_forage_wmslot.sh`), avec le
  masque d'occlusion activé.
- **Sans occlusion** : non-régression byte-proche du forager vivant (15/16 engagement, foraging méd ~860+).

**Promotion** seulement si **PUR (au sens échafaudage-honnête) ET ≥ baseline** (§4). Mettre à jour
`architecture.json` dans LE MÊME commit (état `memoire_spatiale`, role/apporte/preuves/limites/code).

⚠️ **Ops** : le gate closed-loop a besoin de godot lancé en **NATIF** (l'in-session plafonne ~120 s ; planner
WM-slot ~7 pas/s ; background-launch de ces scripts échoue). Le test §2 est pur python rapide. Tuer proprement :
`pkill -9 -f serve_planner_command ; pkill -9 -f 'godot --path godot'` + vérifier 0 restant.

## 5. Fichiers touchés

- **Nouveau** : `python/scripts/diag_slot_memory_drift.py` (test §2, gratuit) + un wrapper `diag_slot_memory.sh`.
- **Build** : `python/scripts/serve_planner_command.py` (classe/état `SlotMemory`, override slot t0) ;
  éventuellement `command_planner.py` (accepter un `slot_belief` override dans la branche WM-slot) ;
  l'ego-motion source (petite tête ou analytique) — module à créer au build.
- **Gate occlusion** : masque rétine paramétrable (env `SYLVAN_OCCLUDE_*`) côté serveur ou godot ; variantes
  des scripts `diag_nav_ab_wmslot.sh` / `run_forage_wmslot.sh`.
- **Carte** : `tools/archi_hud/architecture.json` (à chaque décision qui change le module).

## 6. Critères de promotion (récap falsifiable)

| Étape | Gate | Critère | Si échec |
|---|---|---|---|
| §2 test gratuit | offline drift | bearing<20° & pos<0.5 m à N≈30-40, bat gelé | STOP + escalade (≠ tweaks) |
| §4 closed-loop occlusion | `diag_nav_ab` + foraging A/B (natif) | avec occlusion ≥ no-mémoire ; sans occlusion = non-régression | diagnostiquer GRATUITEMENT avant tout retrain |
| Promotion | pur ET ≥ baseline | + `architecture.json` à jour même commit | ne pas promouvoir |

## 7. Hors-scope (différé, explicite)

- **Cône de vision avant** (`perception.gd:17` 360° → cône) : changement ULTÉRIEUR, seulement quand la mémoire
  est validée (sinon on casse sans filet ; + le WM est entraîné sur 360° → cône = recollecte + retrain WM).
- **Internalisation de la mémoire dans le WM** (hidden RSSM porteur de l'objet, mémoire émergente) : différée
  après l'échafaudage validé, mirroir de l'échelle du slot.
- **Multi-objet / multi-type** (slots eau/prédateur persistants) : un slot food d'abord.
- **Re-grounding sophistiqué** (Kalman / incertitude) : remplacement simple d'abord ; uncertainty-ready plus tard.
