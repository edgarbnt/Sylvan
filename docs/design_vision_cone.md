# Design — VISION EN CÔNE (perception frontale réaliste) : le substrat qui rend l'intelligence NÉCESSAIRE — pré-inscrit 2026-07-21

## Décision de direction (owner, 2026-07-21) — pourquoi ce chantier existe
L'arène **ouverte + plate + vision 360° + 2 ressources éparses** est ÉPUISÉE : tout l'arc récent
(obstacle gelé, critique-arbitrage négatif, mémoire STOP, commitment négatif, cône-vs-360 réfuté,
métrique-juste) a montré que ses morts restantes sont soit des **artefacts** (la 360°), soit du
**substrat** (portée métabolique) — **jamais** des lacunes de décision *apprenable*. Un monde qui ne
*réclame* pas d'intelligence n'en fera pas émerger : à 360° on voit tout → la mémoire est inutile, la
planification triviale, la perception passive, la curiosité vaine. **La simplification-racine à
corriger est la vision 360°.** Passer à un CÔNE frontal réaliste est le seul changement qui soit à la
fois PUR (un sens réaliste, §3) et FONDATEUR (il rend mémoire, perception active et planification
enfin nécessaires — d'un coup). Ce doc pré-inscrit le chantier ; **rien n'est payé (surtout pas le
retrain WM) tant que le G0 GRATUIT n'a pas prouvé que le cône crée la structure revendiquée.**

## Mission
Donner à l'entité une **perception frontale (cône)** : elle ne voit que devant, doit **tourner pour
regarder** (perception ACTIVE), et le « hors-vue » devient réel. But falsifiable : l'entité **acquiert
ses cibles en s'orientant** (comportement ÉMERGENT, pas codé), le forage est préservé, et la mémoire
spatiale devient enfin LOAD-BEARING (elle a un vrai « vu-puis-perdu » à exploiter) — le tout SANS
aucun comportement de balayage codé-main.

## Principe (honnête §2/§3)
- **Le cône = un SENS** (perception), pas une pulsion ni une tâche. Ré-entraîner le WM pour un
  NOUVEAU SENS est explicitement légitime (§3 : « WM entraîné rarement et SEULEMENT si nouvelle
  perception/un nouveau sens »). Ce n'est PAS le raccourci interdit (`--w-<ressource>`).
- **Le comportement doit ÉMERGER.** « Tourner pour scanner », « s'orienter vers un but mémorisé » =
  perception active + coût de survie existant + mémoire. JAMAIS un `if rien_en_vue then tourne` codé.
- **Critère de pureté** (« ça survit à un changement de monde ? ») : l'angle du cône est une propriété
  déclarée du CORPS (comme un champ visuel biologique), datée ici ; la RÉACTION (scanner, se souvenir)
  est apprise/émergente.
- **Honnêteté anti-survente** : le cône rend le monde PLUS DUR. Le risque réel = la perception active
  est difficile → le forage peut s'effondrer. On ne déclare rien avant le juge closed-loop.

## État à la reprise (ce qui existe — rien à réinventer)
- **Masque FOV DÉJÀ écrit** : `occlude_retina(retina, fov_deg)` + `SYLVAN_OCCLUDE_FOV_DEG`
  (`serve_planner_command.py:46,424`) — met à 1.0/0 les rayons hors du cône frontal ±fov/2 ;
  `>=360` = identité (non-régression). **MAIS aujourd'hui appliqué OOD** sur un WM entraîné en 360°
  (leçon mémoire : le masque sur WM-360° est hors-distribution → non fiable). Le vrai cône EXIGE un
  WM ré-entraîné avec le masque ON.
- **Rétine Godot** : `perception.gd` (36 rayons, ray 0 = avant, 10°/rayon, MAX_RANGE 10). Le cône se
  fait par masque des rayons hors-cône (rayons arrière → depth 1.0, couleur 0), pas besoin de changer
  Godot — les rayons arrière deviennent constamment vides, le WM apprend « rien derrière ».
- **Cycle de retrain WM** (pivot cinématique, éprouvé) : `collect_wm_kinematic.sh` (régime propre) →
  `scripts.train_wm_command` (warm-start) → `build_slot_channel.py` → `wm_objcentric_kin`. À rejouer
  avec le masque ON à la COLLECTE.
- **Mémoire** : `MultiSlotMemory` (`slot_memory.py`, dead-reckon EgomotionHead + re-ground saillance)
  existe, opt-in — c'est ELLE qui devient load-bearing sous cône (G0 mémoire STOP notait : place =
  occlusion, non testable en 360°).

## Gates PRÉ-ENREGISTRÉS (falsifiables, cheaper-first — le cher gaté derrière le gratuit)

### G0 — LA STRUCTURE créée par le cône, chiffrée GRATUITEMENT (0 run/Godot/train). GATE TOUT.
Sur les corpus 360° EXISTANTS, appliquer le masque cône OFFLINE (`occlude_retina` à ±θ) et mesurer si
le cône crée bien la structure qu'on revendique — AVANT de payer le retrain :
1. **Place MÉMOIRE** : re-jouer l'analyse « vu-puis-perdu » du G0 mémoire, MAIS avec le masque cône.
   En 360° elle était ≈ 0 (STOP). Sous cône, la ressource poursuivie sort-elle FRÉQUEMMENT du champ
   (→ vu-puis-perdu ATTEIGNABLE) ? Seuil : part vu-puis-perdu-faisable **> bruit (5/24 vies)**.
2. **Besoin de perception ACTIVE** : quelle fraction des replans la cible désignée est-elle HORS du
   cône (→ il faut tourner pour la ré-acquérir) ? En 360° = 0 (tout visible). Sous cône, une fraction
   substantielle = le monde force l'orientation active.
3. **Contrôle de faisabilité** : la cible sort-elle du cône par des angles ATTEIGNABLES par rotation
   (kin_turn) dans le temps métabolique ? (sinon on crée un monde impossible, pas dur).
**VERDICT G0** : (1) place mémoire > bruit ET (2) perception-active requise substantielle ET (3)
faisable → chantier LICENCIÉ, G1. Sinon (le cône ne change presque rien, ou rend le monde
impossible) → **STOP, négatif commité**, la 360° reste (on ne paie pas le retrain pour rien).
⚠️ Choix de θ DÉCLARÉ ici avant le G0 (candidat ±120° = cône large biologique) ; balayer θ ∈
{±90, ±120, ±150} au G0 pour choisir le plus petit θ qui crée la place SANS rendre infaisable (§2 :
θ n'est pas ajusté après coup pour faire passer un gate).

### G1 — retrain WM sous cône (le cher, gaté par G0) — feasibility de la DYNAMIQUE
- Collecte régime propre avec `SYLVAN_OCCLUDE_FOV_DEG=θ` ON (`collect_wm_kinematic.sh` adapté) →
  `train_wm_command` (warm-start `wm_objcentric_kin`) → `build_slot_channel`.
- **MESURER open-loop** (`eval_wm_command`) : le WM apprend-il la dynamique CÔNE — objets qui
  APPARAISSENT/DISPARAISSENT à la rotation (le cœur du sens nouveau) — SANS régresser
  displacement/pos/eff_rank vs le WM 360° ? Un WM qui ne prédit pas l'apparition-par-rotation ne
  porte pas le sens → STOP + diagnostic (budget dur : 1 retrain + 1 re-collecte diagnostiquée).

### G2 — juge closed-loop (gaté par G1) — le comportement ÉMERGE-t-il ?
- Forage survie multi-drive, cône ON, vs la config 360° vivante (réf mesurée AVANT) :
  **PASS** = forage ≥ réf − bruit (l'entité s'ORIENTE pour acquérir ses cibles = perception active
  émergente ; pas d'effondrement) ; **et** la mémoire (`MultiSlotMemory` ON) apporte enfin un gain
  MESURABLE sous cône (re-juge mémoire, cette fois avec une place réelle). Zéro balayage codé.
- **KILL précoce** : forage s'effondre (l'entité tourne en rond sans acquérir) → le sens actif est
  hors de portée du substrat actuel, négatif commité (piste = corps/tête d'orientation, hors scope).

## ⭐⭐ VERDICT G0 (2026-07-21, `diagnostics/diag_cone_g0.py`, gratuit 0 run/train) : PARTAGÉ → retrain NON licencié (§1), redirection vers l'obstacle
288 vies, cône appliqué OFFLINE (1er ordre : orientations vécues en 360°).
- **(A) perception active = PASSE modestement** : ressource urgente HORS-cône en portée = ±90° 15,6 %
  / ±120° 9,4 % / ±150° 3,5 % (360° = 0). Un cône crée bien du « tourner pour voir » — réel mais
  modeste, et 1er ordre (un VRAI cône verrait l'entité s'orienter vers sa cible → moins).
- **(B) place mémoire = ÉCHOUE la barre** : « vu-puis-perdu FAISABLE » = ±90° **1,0/24 vies** (barre
  5), ±120° 0,5, ±150°/360° 0. `seen_in_critical` DOMINE (151-176) : même sous cône, au moment
  critique la ressource est ENCORE devant → peu de « perdu quand j'en ai besoin ».
- **Nuance honnête (§2, ne pas maquiller mais ne pas sur-lire)** : (B) est aussi structurellement
  SOUS-MESURÉ — le comportement qui CRÉE le vu-puis-perdu (tourner le dos à une ressource puis en
  avoir besoin) n'existe QUE sous un vrai cône ; les corpus 360° ne le contiennent pas. Donc (B)=1/24
  est un PLANCHER, pas une preuve d'absence. Mais il ne PASSE pas le gate pré-enregistré.
**➜ VERDICT per pré-enregistrement (« (1) mémoire > bruit ET (2) active ET (3) faisable → licencié ;
sinon STOP ») : la condition (1) échoue → le RETRAIN WM n'est PAS licencié.** Le G0 gratuit a fait
son travail (§1) : il a évité de payer un cycle WM cher sur un pari dont le payoff (B) n'est même pas
mesurable offline. Le cône n'est pas RÉFUTÉ (son payoff exige l'investissement pour être testé) — il
est « non cheaply justifiable ». **REDIRECTION (evidence-driven)** : le but « hors-vue → mémoire »
est atteint PLUS CHER par le cône (retrain + gate inconclusif) que par l'OBSTACLE — l'occlusion
PHYSIQUE (un mur cache une ressource ATTEIGNABLE = le vrai « atteignable-mais-invisible ») est
concrète, géométrique, DÉJÀ à moitié bâtie (monde_obstacle, gelé au G2), et ne demande PAS de retrain.
Reco : obstacle-first pour la mémoire ; le cône (réalisme + perception active) reste un pari de
substrat PLUS TARD, une fois la mémoire prouvée utile en occlusion physique. Décision owner.

## Ce qu'on ne touche JAMAIS
Les drives (pulsions du corps) ; le moteur (locomotion = prérequis DONNÉ) ; le readout géométrique du
slot ; le transport. Le WM EST ré-entraîné — c'est le but (nouveau SENS, §3) — mais UNE fois, pas par
ressource. Aucun comportement de balayage/scan codé-main (§2). `main.gd` jamais stagé. Carte
`architecture.json` à jour dans le commit de chaque verdict.

## Critère de succès = le BUT
En vies, sous cône : l'entité **s'oriente pour percevoir et acquérir** ses ressources (comportement
émergent), **forage préservé**, et la **mémoire devient load-bearing** (elle exploite un vrai
« vu-puis-perdu »). Si PASS : le substrat de la PARTIAL OBSERVABILITY existe → mémoire, perception
active, planification-avec-occlusion et (plus tard) curiosité deviennent enfin nécessaires et
buildables, une tête apprise à la fois — le patron qui marche déjà pour la perception. Si échec :
négatif commité, cause diagnostiquée ; le G0 gratuit d'abord garantit qu'on ne paie le retrain QUE si
le cône crée vraiment la structure.

## Liens
- `docs/design_memoire_spatiale.md` (G0 STOP : place mémoire = occlusion, non testable en 360°) —
  le cône est l'autre route vers le « hors-vue », testable ici.
- `docs/design_obstacle_affordance.md` (§GEL) — obstacles = occlusion PHYSIQUE ; complémentaire du
  cône (occlusion par le champ). Les deux créent la topologie/partial-observability.
- `memory/sylvan-foraging-economy.md` (métrique juste : l'artefact-derrière de la 360°) —
  la motivation empirique directe.
- `memory/sylvan-keystone-3b-geometric-wall.md` (véhicule forward-only) — sous cône, l'engagement de
  cible devient une compétence à apprendre, plus un artefact.

---

# RÉOUVERTURE (2026-07-21) — la prémisse du G0 a changé

**Pourquoi rouvrir un chantier non licencié.** Le G0 cône avait rendu un verdict partagé et le
retrain WM n'avait **pas** été licencié : (A) la perception active passait modestement, (B) la place
mémoire échouait la barre offline. Deux faits NOUVEAUX changent la prémisse :

1. **Le monde a une forêt** (`forest_solid.gd`, gate passé : atteinte lointaine −16,5 pts, capacité
   de base préservée). Le G0 cône avait été joué en arène **vide**.
2. **Le « hors-vue » mesuré était du SCINTILLEMENT, pas de l'occlusion** — sans aucun arbre, 3309
   éclipses de durée médiane **5 ticks**, parce que 36 rayons à 10° ne touchent une ressource
   lointaine que par intermittence. Les massifs n'y changent presque rien (5 → 6 ticks).
   ⇒ *Aucune mémoire ne peut exploiter une ressource qui clignote toutes les 5 ticks.*

**L'idée (owner) : faire d'une pierre deux coups.** Un cône frontal crée un hors-vue **structurel et
durable** — ce qui est derrière reste invisible tant qu'on ne se retourne pas — au lieu du
scintillement. Il rend aussi la perception réaliste. C'est le seul changement qui attaque la *cause*
mesurée plutôt que d'empiler des occulteurs.

**Coût : nul en première approche.** Le masquage existe déjà (`SYLVAN_OCCLUDE_FOV_DEG`,
`serve_planner_command.py:49`) et met les rayons hors-cône à « rien en vue » (depth 1.0, RGB 0) —
état **in-distribution** pour le WM, qui voit déjà des rayons vides. Donc **aucun ré-entraînement**
pour ce premier test ; la question du retrain ne se posera que si le cône est retenu.

## Critères PRÉ-INSCRITS (avant lancement)

Protocole : monde forestier (18 massifs), `SYLVAN_OCCLUDE_FOV_DEG=180` (±90°, la valeur du G0),
2 seeds, témoin = les corpus forestiers 360° déjà collectés.

1. **Le cône remplace-t-il le scintillement par du vrai hors-vue ?**
   → durée **médiane** des éclipses ≥ **50 ticks** (contre 5-6 aujourd'hui).
   *C'est la condition PRIMAIRE : sans elle, le reste n'a pas de sens.*
2. **La mémoire a-t-elle enfin une place ?**
   → `seen-then-lost` FAISABLES > **5/24 vies** — **la barre historique, non déplacée** (§2).
3. **Garde de capacité** : atteinte `[0,2)` ≥ **85 %**. En dessous, le cône casse la capacité de base
   (l'entité ne retrouve plus ce qu'elle a devant) → réduire l'angle avant toute conclusion.
4. **Validité** : `guards.sanity()` sur les 4 corpus, sinon verdict **NUL**.

**Attendu honnête** : l'atteinte VA se dégrader — c'est normal, on retire de l'information. Ce qui
est jugé ici n'est pas « l'entité fait-elle mieux » mais « le monde produit-il enfin la situation où
une mémoire SERAIT utile ». Un cône qui dégrade tout SANS créer de hors-vue durable serait un échec.
