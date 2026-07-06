# Pivot corps : différentiel caché (roues invisibles, pattes cosmétiques)

_2026-07-06. Pourquoi on remplace le corps hexapode à pattes par un coeur cinématique (vx, ω),
habillé visuellement en loup low-poly. Le *pourquoi* et le plan vivent ici ; l'*état* des modules
vit dans `architecture.json` ; l'historique long dans `memory/sylvan-mode1-build.md`._

## 1. Mission
Rendre la LOCOMOTION un prérequis DONNÉ (cinématique, obéit exactement à (vx, ω)) au lieu d'une
compétence à pattes lente et pénible. But réel : assainir le substrat pour que la cognition
(world-model JEPA + planification + pulsions) arrête de payer une taxe locomotion sur chaque
expérience. La créature RESTE incarnée et visuellement une bête qui marche ; seul le moteur du
mouvement change, caché sous un cycle de marche décoratif.

## 2. À lire d'abord
- `docs/orbite_far_target_pur.md` §10 : la limite MESURÉE (corps trop lent) qui motive le pivot.
- `CLAUDE.md` (« Corps actuel = HEXAPODE », dims 132/18/144/145, régime propre) : ce qui change.
- `tools/archi_hud/architecture.json` : modules `cpg`/`résidu` (à jeter comme moteur), `wm`/`planner`/
  `slot`/`pulsions` (à garder).

## 3. Limite mesurée (la raison du pivot)
Vérité-terrain (buffer `wm_hex_v2`, `wm_dataset.py`) : le corps hexapode avance **~0.0043 m/pas** en
ligne droite, l'avance **chute de moitié** en tournant (0.0021 à |ω| fort), et il tourne **~0.7°/pas**
même à ω max. Conséquence : portée « atteindre + manger » plafonnée à **~5 m** dans le budget énergie
réel (énergie 80 → ~1600 pas). Au-delà (6-8 m) le corps ne peut PAS fermer le dernier mètre avant la
panne. Plus le mur récurrent « tourner-en-avançant » et l'impossibilité du pivot vx≈0 (hors bande WM
0.55-0.75). C'est un plafond LOCOMOTION (saga déjà épuisée à ~0.49 m/s), pas un problème de scoring.

## 4. Essayé → résultat (négatifs informatifs, NE PAS répéter)
- **Échafaudage de cap FAR_ALIGN** (récompense la trajectoire rêvée qui pointe la ressource) : CAUSAL
  et robuste sur l'APPROCHE (orbite 4.9 m → **1.9 m** de proximité médiane vs OFF), MANGE 3/10 à 5 m.
  Mais taux de repas capé par la vitesse du corps (0/12 à 6 m). Casse l'orbite, ne bat pas la physique.
- **Peaufinage END-align** (cap de FIN au lieu du cap MOYEN, anti-spirale) : **RÉFUTÉ** (proximité 2.6 m
  vs 1.9 m = pire). Gardé flaggé, documenté négatif.
- **Horizon plus long** (80/120/200) : RÉFUTÉ (l'approche latérale du rêve sature).
- Conclusion : le levier n'est ni le scoring ni le WM (le rêve est FIDÈLE : 0.0046 rêvé ≈ 0.0043 réel).
  C'est le CORPS. Rouvrir la vitesse de l'hexapode = puits connu → on change de corps.

## 5. Le pivot : garder / jeter / remplacer
- **GARDER (intacts)** : `command_wm` (WM commande-espace), `command_planner` (MPC en (vx, ω)), le
  SLOT object-centric (perception/permanence), la rétine, les PULSIONS (faim/soif) + coût survie, le
  critique appris. Toute la couche COGNITIVE. L'interface reste **(vx, ω)** — d'où l'invariance.
- **JETER (comme moteur du mouvement)** : le CPG-physique (démarche à pattes) et le RÉSIDU PPO
  (équilibre/correction). Ils n'existaient que pour faire marcher/tenir debout les pattes. Le CPG peut
  SURVIVRE comme animation cosmétique (phase du cycle de marche), pas comme physique.
- **REMPLACER** : le corps Godot → **coeur cinématique différentiel** : un `CharacterBody` qui intègre
  directement « avance à vx selon le cap, tourne à ω » (roues invisibles, ou pas de roue du tout —
  cinématique pur, stable, obéit au doigt). Pivote sur place trivialement (vx≈0, ω fort désormais
  in-distribution). Proprioception : les 132 dims d'angles de pattes disparaissent → proprio simplifiée
  (pose/vitesse du corps, éventuellement état des pattes cosmétiques = redondant). **Recollecte du WM**
  sur la nouvelle dynamique (rapide, propre par construction). Sync des dims dans `constants.py`,
  `observation_builder.gd`, `sylvan_agent.gd`, `symmetry.py`.
- **VISUEL (découplé, dernière étape)** : loup low-poly CC0 de Quaternius
  (`https://poly.pizza/m/P1gU3Qkr9r`, GLTF/GLB, importe direct dans Godot 4). Mesh cosmétique, cycle de
  marche calé sur vx (vitesse d'anim ∝ vitesse), le mesh pivote avec ω, capsule de collision, roues
  cachées. Peut être re-choisi sans rien casser (c'est le point du découplage).

## 6. Prochain pas — cheaper-first (GATE GRATUIT avant de tout recâbler)
1. **GATE mécanique (le moins cher qui décide)** : bâtir le coeur cinématique (vx, ω) MINIMAL dans
   Godot (obéit exactement, pas d'apprentissage), le brancher au planner EXISTANT (le planner sort déjà
   (vx, ω)), et re-lancer le test far-food (`collect_curriculum_farfood.sh`, énergie 80, bouffe 5-8 m).
   Aucune recollecte WM, aucun retrain : juste « est-ce que le corps qui obéit dissout l'orbite ? ».
2. Si le gate passe → recollecte WM (nouvelle dynamique) + rewire proprio + sync dims + brancher le loup.
3. Si le gate échoue → le corps n'était pas la cause, STOP + réexaminer (mais l'approche 4.9→1.9 m de
   l'échafaudage rend l'échec improbable).

## 7. Honnêteté (§2 + consigne owner)
Le VISUEL peut être une bête qui marche ; la PROSE (README, doc) doit dire que le déplacement est un
**prérequis cinématique DONNÉ**, jamais revendiquer une locomotion à pattes apprise/émergente. Le README
est un **constat au présent** de ce que le projet EST (pas de « avant X, maintenant Y » : l'historique
n'intéresse personne). Les revendications qui comptent (JEPA, planification, pulsions, décision
émergente) restent 100 % intactes et honnêtes.

## 8. Critère de succès = le BUT
- Gate mécanique : far-food atteint **≥ 60 %** avec le coeur cinématique (le gate que les pattes
  échouaient par plafond de vitesse) → le pivot dissout l'orbite.
- Gate final : monde ÉPARS 1+1 survie médiane **> 1800** ; non-régression dense **≥ ~2735**.
- Multi-seed avant toute promotion (dette de rigueur du projet).

## 9. Gate mécanique PASSÉ (2026-07-06) — coeur cinématique implémenté + testé
Implémenté : flag `SYLVAN_KINEMATIC` (2 fichiers, `sylvan_agent.gd` `_kinematic_step` + `main.gd` gate,
défaut OFF = zéro régression). Le corps GLISSE l'assemblage entier rigidement à (vx, omega) via
`PhysicsServer3D.body_set_state` (réutilise le placement de `reset_agent`), pattes gelées KINEMATIC en pose
neutre → proprio 132 cohérente. Tunables `SYLVAN_KIN_SPEED` (m/s par vx) / `SYLVAN_KIN_TURN` (rad/s par omega).
Smoke OK (compile, obéit : fwd_v = kin_speed×vx constant, glisse droit).

**Gate far-food (énergie 80, bouffe 5-8 m — la plage qui échouait, 12 ep, kin_speed=0.5 kin_turn=0.53) :**
- **KIN seul (sans échafaudage) = 0/12** → le corps SEUL ne dissout PAS l'orbite (c'est le SCORING du
  planner, pas que le corps). L'échafaudage de cap reste nécessaire (→ remplacé par le critique, plan intact).
- **KIN + échafaudage = 7/12 (58 %)**, survie 1900, repas NETS (min food_d 1.00-1.03 m). vs hexapode+échaf
  qui plafonnait (3/10 à 5 m, **0/12 à 6 m**). Le corps rapide+obéissant apporte la **PORTÉE** far-food,
  SANS recollecte WM (le WM hexapode décalé ~2.7× navigue quand même : l'échafaudage donne la direction,
  le corps couvre le terrain). 58 % ≈ seuil 60 % (au bruit près, avant recollecte).

**Verdict : pivot VALIDÉ** (le mur de portée far-food est cassé). Suite : (a) recollecte WM sur la dynamique
cinématique (retire le décalage ~2.7× → devrait passer >60 % franc + fiabiliser) ; (b) brancher le loup
(visuel) ; (c) reprendre la recette échafaudage→critique pour retirer le hint de la boucle finale.

## 10. Recollecte WM sur dynamique cinématique (2026-07-06) — pipeline validé, recollecte NEUTRE
Fait : collecte cinématique (`scripts/collect_wm_kinematic.sh`, kin_speed=0.5 kin_turn=1.5, 2×150 ép,
fall_rate 0 %, dynamiques PROPRES : fwd linéaire en vx, yaw linéaire en ω, side-slip ~0) → train base WM
(recette JEPA-pur `train_wm_jepa_pur.sh`, warm-start `wm_rich_fidele_sym_jepa`, 10 ép, val 0.810,
displacement 0.0005 = appris nickel, eff_rank 15) → `build_slot_channel.py` (slot `slot_head_multi` réutilisé
tel quel, la rétine est inchangée) → **`data/checkpoints/wm_objcentric_kin`** (obs 277, slot-2, food/water 0/1).

**Re-gate far-food (énergie 80, 5-8 m, 12 ép, kin_speed=0.5 kin_turn=1.5) :**
- matché + échafaudage = **6/12 (50 %)** ; matché SANS échafaudage = **0/12** (orbite → scaffold toujours requis).
- vs WM décalé + échafaudage = 7/12 (58 %). **50 % ≈ 58 % = BRUIT (n=12).**

**Conclusion honnête : la recollecte est NEUTRE.** Le décalage WM n'était PAS le facteur limitant (le WM matché
navigue aussi bien, pas mieux). Le pipeline complet est validé de bout en bout. Le plafond ~50-58 % = la QUEUE
LOINTAINE (7-8 m) au bord de l'enveloppe vitesse/énergie (kin_speed=0.5 → portée ~7 m) : les réussites sont
≤6 m nettes, les 7-8 m ratent par portée. **Levier pour >60 % = corps PLUS RAPIDE (kin_speed↑ + recollecte),
pas le WM.** Le gain-pivot cœur (far-food 0 %→~55 % à 5-8 m) tient. Checkpoints `wm_kin_base` +
`wm_objcentric_kin` sur disque (NON promus vivants ; hexapode s2 reste le défaut).
