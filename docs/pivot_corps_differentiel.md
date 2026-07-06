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
