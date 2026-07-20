# Design — MÉMOIRE SPATIALE (permanence hors-champ) : la brique que DEUX murs réclament — pré-inscrit 2026-07-20

## Mission
Donner à l'entité un monde qui **persiste hors du champ de vision** : se souvenir de la position
ego d'une ressource **vue puis perdue de vue**, et la **re-cibler** quand le besoin l'exige. But
falsifiable : la mémoire, branchée, **convertit en vies** une part mesurable des morts que deux
chantiers indépendants viennent d'imputer à l'absence de permanence — **jugée en vies, ≥ la config
vivante, zéro comportement codé** (« retourner où j'ai vu » doit ÉMERGER de mémoire + coût existant,
pas d'un `if`).

## ⚠️ RÉOUVERTURE D'UN NÉGATIF FERMÉ — l'hypothèse NOUVELLE (anti-boucle §1, obligatoire)
`memoire_spatiale` porte DÉJÀ un négatif : gate closed-loop occlusion « **AUCUN GAIN → NON PROMU** »
(carte, 2026-07-05). **Ne PAS le relancer en espérant.** Ce qui a changé (hypothèse falsifiable
justifiée par preuve) :
1. **Le négatif testait la MAUVAISE question.** Ses échecs sous occlusion = objets **JAMAIS VUS**
   (la mémoire est impuissante par construction) sous un **masque d'occlusion OOD sur un WM 360°**
   (pas un vrai cône) → un test biaisé contre la seule chose que la mémoire sait faire.
2. **Deux murs INDÉPENDANTS surfacent maintenant le SWEET SPOT « vu-puis-perdu »** :
   - **Arbitrage** (`design_critique_arbitrage.md`, VERDICT G3 + investigation post-juge) : la place
     G0 (13-16 morts-par-arbitrage/24 vies) est verrouillée par la **bascule PRÉCOCE**, qui exige de
     viser une ressource **HORS-VUE** ; campée, l'entité a PERDU la nécessiteuse de vue → le choix
     multi ne peut même pas la considérer. Une meilleure tête de choix est RÉFUTÉE (4 sondes) : le
     verrou est la permanence, pas la valeur.
   - **Obstacle** (`design_obstacle_affordance.md`, §GEL, réouverture (b)) : contourner un mur
     OCCLUANT fait perdre la cible → l'agent glouton erre. Finding structurel identique.
   **Deux chemins totalement disjoints pointent la même brique = signal de priorisation fort.**
3. **Le test décisif n'a JAMAIS été payé** : `MultiSlotMemory` (par-ressource, invalidate-à-la-
   consommation) existe mais n'a été jugée QUE sur l'occlusion mono-slot. La bascule d'arbitrage
   multi-drive et le contournement non-occluant sont des juges NEUFS.
Si le G0 ci-dessous montre que « vu-puis-perdu » est RARE (place ≤ bruit) → le négatif TIENT,
STOP, on ne re-paie pas l'échafaudage. La réouverture n'est légitime QUE si G0 chiffre la place.

## À lire d'abord
- `design_critique_arbitrage.md` (§ investigation post-juge : le mur = mémoire, pas la valeur) +
  `design_obstacle_affordance.md` (§GEL, réouverture (b)) — les deux murs qui fondent le chantier.
- `python/sylvan/control/slot_memory.py` — **`MultiSlotMemory`** (le composant à REBRANCHER, PAS à
  réécrire) : re-ground si `visibility_k > eps`, dead-reckon `EgomotionHead` sinon, ÂGE par
  ressource expirant à `max_age_steps=500` (plafond géométrique : dérive ~0.2 m/100 pas vs rayon
  1.0 m), `invalidate(k)` à la consommation (intéroception). `SlotMemory` = variante mono-slot.
- Carte `architecture.json` module `memoire_spatiale` (état PARTIEL, limites déclarées : cône OOD,
  perception bord-de-champ, seuil saillance 0.05 non calibré — les 3 dettes à traiter honnêtement).

## Le principe (honnête §2/§3)
- **La mémoire = SUBSTRAT GÉNÉRAL** (une permanence ego-centrée), pas un dispositif par-ressource ni
  par-tâche. Elle vit AU-DESSUS du slot/WM gelés, elle ne les ré-entraîne pas (§3). Le planner LIT
  le belief comme un override du slot quand l'objet est hors-vue — `SlotMemory` le fait déjà.
- **Le comportement doit ÉMERGER** : « retourner vers l'eau mémorisée quand j'ai soif » = mémoire
  (position hors-vue disponible) + coût survie EXISTANT (l'urgence tire vers elle). JAMAIS un
  `if soif then va_vers_dernière_eau` codé.
- **Pureté (le test « ça survit à un changement de monde ? »)** : dead-reckon = ego-motion APPRISE
  du proprio (EgomotionHead, R² 0.95-0.99) ; re-ground = saillance APPRISE du slot ; expiration =
  géométrie (dérive vs rayon), pas un seuil tuné pour passer. Zéro oracle dans la boucle.
- **Honnêteté anti-survente** : la mémoire ne peut aider QUE « vu-puis-perdu ». Elle n'invente rien
  sur un objet jamais perçu. Si la place vient surtout d'objets jamais-vus → c'est un mur de
  PERCEPTION/EXPLORATION, pas de mémoire (le dire, ne pas maquiller).

## Gates PRÉ-ENREGISTRÉS (falsifiables, cheaper-first)

### G0 — LA PLACE « vu-puis-perdu », chiffrée GRATUITEMENT (0 run, 0 Godot, 0 train). GATE TOUT.
Sur les corpus vécus EXISTANTS (arbj_{ref}_s1/s2 + les 12 runs G0/arb ; BC_LOG = drives + rétine +
plan par tick), reconstruire par vie et TRANCHER, pour les **morts-par-arbitrage** (parseur
`diag_arbitrage_g0`, la place déjà chiffrée) :
1. **Part « vu-puis-perdu »** : sur les morts-par-arbitrage, quelle fraction avait la ressource
   nécessiteuse **PERÇUE dans la rétine** à un moment, puis **SORTIE du champ** ≥ K pas avant la
   fenêtre critique ? (mémoire = convertible) vs **jamais perçue** (perception/exploration, hors
   mémoire) vs **perçue en continu** (déjà un problème de choix/portée, déjà réfuté).
2. **Faisabilité géométrique** : au moment critique, le belief dead-reckoné de la nécessiteuse
   (rejeu `transport_geom` + `EgomotionHead` sur le proprio loggé) serait-il resté **dans le rayon
   de capture soutenable** (dérive < seuil ET distance ≤ portée métabolique) ? = la place NETTE.
3. **Contrôle bord-de-champ (la dette #2 de la carte)** : la dernière perception « avant de perdre »
   était-elle en BORD de rétine (belief semé bruité) ? Quantifier le risque de fantôme.
**VERDICT G0** :
- Part « vu-puis-perdu » convertible **> bruit (±5 équiv./24 vies)** ET faisabilité géométrique OK
  → chantier LICENCIÉ, G1.
- Part ≤ bruit, OU dominée par jamais-vue → **STOP, le négatif occlusion TIENT** (le vrai verrou
  est perception/exploration, pré-inscrire ça comme chantier distinct), négatif commité.
- Faisabilité KO (dérive > rayon systématiquement) → mémoire courte insuffisante → STOP + noter.

### G1 — rebranchement + qualité de belief offline (payé si G0 licencie)
- Rebrancher `MultiSlotMemory` dans `serve_planner_command` (flags existants), **opt-in défaut OFF
  bit-identique** ; smoke ε=OFF byte-identique quand mémoire OFF.
- MESURER offline sur trace : le belief re-cible-t-il la bonne ressource au bon moment ? erreur de
  belief vs re-perception ultérieure (ground-truth de retrouvailles) ; taux de fantôme (belief
  jamais re-confirmé). Calibrer le seuil saillance / `max_age` sur la GÉOMÉTRIE mesurée, pas pour
  passer un gate (§2).

### G2 — juges closed-loop (payés si G1 ; DEUX juges déjà bâtis)
- **Juge ARBITRAGE** : `scripts/judge_arb_critic.sh` (harnais commité 2026-07-20) — bras
  mémoire-ON vs config vivante, 2×24 vies seeds 1+2, réf re-mesurée, **PASS pré-chiffré AVANT le
  run** : morts-par-arbitrage ↓ au-delà du bruit ET conso ≥ réf−5 ET morts totales ≤ réf+2 (la
  mémoire ne doit pas exporter l'échec, leçon G3 arbitrage).
- **Juge OBSTACLE non-occluant** (réouverture (a) du §GEL obstacle) : monde mur-décalé
  protubérant, contournement testable ; mémoire-ON doit contourner + forage préservé.
- Un PASS sur AU MOINS un juge (avec non-régression sur l'autre monde) = la mémoire porte une
  valeur. Budget dur : pas d'entraînement (la mémoire est sans-paramètre-appris nouveau — tout est
  déjà appris : egomotion, saillance) → le coût est la calibration + les juges, pas un run PPO/WM.

## ⭐⭐ VERDICT G0 (2026-07-20, `diagnostics/diag_memory_g0.py`, gratuit 0 run/0 train) : STOP — le mur d'arbitrage est la PORTÉE, pas la mémoire ; le négatif occlusion TIENT
288 vies, 155-195 morts-par-arbitrage. Décomposition (rétine RED/BLUE = ce que l'œil a vu ;
dead-reckoning rejoué par `EgomotionHead` R² 0.95-0.99 + `transport_geom`, exactement
`MultiSlotMemory`) :
- **`never_seen` = 0** partout → AUCUN trou de perception : chaque mort-par-arbitrage avait vu sa
  ressource. La mémoire est *en principe* pertinente ; mais elle ne suffit pas — voir ci-dessous.
- **`seen_in_critical` DOMINE = 114/155 (73 %) à 154/195 (79 %)** : la ressource nécessiteuse était
  **ENCORE EN VUE au moment critique** → ce n'est PAS un problème de mémoire (rien à re-cibler de
  mémoire, elle est sous les yeux). C'est le problème de CHOIX (arbitrage G3, réfuté) et surtout de
  PORTÉE (investigation post-juge : basculer vers la nécessiteuse ne réussit que 37.7 % — elle est
  LOIN).
- **`seen_then_lost` = 41 bruts (3.4/24)** MAIS **FAISABLES = 1.9/24 (optimiste), 0.0/24
  (conservateur)** — LARGEMENT sous la barre 5. Et **bord-de-champ = 41/41** : la ressource perdue
  était TOUJOURS en train de reculer sur le côté/derrière (l'agent fonce vers l'autre cible). En
  rétine 360°, elle n'est « perdue » qu'une fois **au-delà de 10 m** → hors portée métabolique de
  toute façon (d'où faisable 0/41 conservateur). La mémoire re-ciblerait un souvenir **inatteignable**.
**➜ STOP per pré-enregistrement (verdict « ≤ bruit »).** Le négatif occlusion TIENT, re-dérivé
d'un angle indépendant. **Diagnostic unifié des DEUX G0** : le mur multi-drive épars est un
**PLAFOND DE PORTÉE/MÉTABOLIQUE** (la nécessiteuse est trop LOIN quand elle devient urgente),
que ni une meilleure tête de choix (arbitrage G3) ni la mémoire (ici) ne cassent — la ressource
est *vue-mais-inatteignable* ou *souvenue-mais-inatteignable*, jamais *atteignable-mais-oubliée*.

**Où vit ENCORE une place mémoire (NON testable sur ces corpus, à ne PAS deviner)** :
1. **Occlusion PHYSIQUE** (monde obstacle : un mur solide cache la ligne de vue même à &lt;10 m) — la
   seule vraie perte « atteignable-mais-invisible ». Mais ces corpus n'ont pas d'obstacle → couplé
   au G3 obstacle gelé (il faudrait collecter le monde-obstacle), pas un gain autonome.
2. **Cône de vision RÉEL** (dette #1 de la carte : WM ré-entraîné sur perception en cône, pas un
   masque OOD sur WM-360°) — alors « derrière moi » devient vraiment invisible. Coûteux (cycle WM),
   non gaté par une place mesurée ici.
**Recommandation (owner) : ne PAS bâtir la mémoire sur la justification arbitrage — elle est
réfutée.** Le vrai levier que les deux G0 pointent = la PORTÉE (corps plus rapide `kin_speed`, dette
notée du pivot cinématique ; OU monde moins épars) — décision de SUBSTRAT, pas un étage à empiler.
Chantier mémoire re-scopé : ne le rouvrir que couplé à l'occlusion PHYSIQUE (monde obstacle), avec
son G0 propre sur un corpus-obstacle.

## ⭐⭐ VERDICT G0 MONDE-OBSTACLE (2026-07-21, `diagnostics/diag_obstacle_memory_g0.py`, gratuit) : place RÉELLE mais MODESTE, range-entangled → A/B en-vies gaté (pas de retrain)
Mesuré sur les corpus obstacle EXISTANTS (`obstacle_g1nav`/`g2nav`, food-only + mur solide), vs
sparse 360° :
- **STRUCTUREL — l'occlusion est réelle et grande** : bouffe visible **22 %** (obstacle) vs **63 %**
  (sparse). Monde food-only → l'occlusion est la SEULE cause (pas d'arbitrage).
- **COÛT réel** : forage ~divisé par 2 (15 repas vs 40 sparse à ticks comparables).
- **BEHAVIORAL (le finding G3-gelé quantifié)** : la bouffe disparaît à **6,9 m** (méd) et réapparaît
  à **6,8 m** → l'agent naïf **ne progresse PAS** vers la bouffe cachée pendant qu'il est aveugle
  (même distance avant/après) → sans mémoire, il n'avance pas vers ce qu'il ne voit plus. C'est LE
  travail de la mémoire.
- **CAVEAT honnête (§2)** : l'occlusion arrive surtout à ~7 m (range-borderline), PAS de près —
  seulement ~10 % des transitions concernent une bouffe <3,5 m. Donc la place est RÉELLE mais MODESTE
  et **entangled avec le plafond de portée** (7 m = bord de l'enveloppe métabolique). Le raffinement
  « belief dead-reckon <3,5 m » a donné 0 % = ARTEFACT DE SEUIL (la bouffe est à 7 m, pas 3,5 m), jeté
  honnêtement (comme la sous-mesure tautologique de l'arbitrage).
**➜ VERDICT : place mémoire RÉELLE (la plus claire de toutes les directions essayées : occlusion +
coût + agent naïf qui échoue, sans confond d'arbitrage) MAIS modeste et range-entangled — PAS un
slam-dunk.** Pas un STOP (le signal structurel est net) ni un PASS franc. **PROCHAIN = A/B EN VIES
CHEAP** (mémoire `MultiSlotMemory` ON vs OFF dans le monde-mur, ZÉRO retrain — juste brancher le
module ; harnais forage obstacle) = le vrai arbitre, avec attentes TEMPÉRÉES (le plafond de portée à
~7 m cape le gain possible). Si l'A/B est plat → la mémoire ne récupère pas le coût d'occlusion sur ce
substrat, négatif commité, et le plafond de portée est confirmé comme limiteur PERVASIF (il revient à
CHAQUE direction). Note owner : ce pattern — toute direction bute sur la portée ~7 m — est lui-même le
signal fort que le levier réel est la PORTÉE/vitesse (substrat), pas un étage de plus.

## Ce qu'on ne touche JAMAIS
Le WM (gelé) ; le slot_encoder / la saillance / l'EgomotionHead (déjà appris, GELÉS) ; les drives ;
le sprint-critic et les lentilles waypoint (interface gelée) ; le choix de cible designé (l'arbitrage
appris est CLOS en négatif — la mémoire agit AVANT lui en rendant une cible hors-vue disponible, pas
en re-scorant). `main.gd` jamais stagé. Carte à jour dans le commit de chaque verdict.

## Critère de succès = le BUT
En vies : l'entité **revient vers une ressource qu'elle a mémorisée hors-vue** au bon moment (bascule
précoce en arbitrage OU contournement d'obstacle), **forage préservé, zéro comportement codé**. Si
PASS : premier modèle du monde hors-champ load-bearing → débloque simultanément les deux murs gelés
(arbitrage + obstacle), et pose le substrat de l'intelligence « plus profonde » (l'audit
`formes_intelligence.md`). Si échec : négatif commité, cause diagnostiquée sur trace ; le G0 gratuit
d'abord garantit qu'on ne re-paie l'échafaudage QUE si la place « vu-puis-perdu » existe vraiment.
