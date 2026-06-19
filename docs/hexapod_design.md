# Hexapode trépied — design & plan de build (2026-06-16)

DÉCISION (user) : changer la NATURE de l'entité — abandonner le quadrupède sprawlé (marche lente,
saccadée, sur-place : le mur = équilibre précaire sur 4 pattes → l'énergie va à ne pas tomber, pas à
avancer ; cranker la vitesse → bascule). Voir [[sylvan-phase-a-progress]] pour la saga complète.

NOUVELLE NATURE : **hexapode terrestre, démarche en TRÉPIED** (cafard). Principe : 6 pattes →
stabilité statique GRATUITE (ne tombe jamais) → l'énergie va à la PROPULSION + on peut cranker la
cadence/foulée SANS chute. Preuve vivante : cafard ~50 longueurs-corps/s, stable. Stable = MOTEUR de
vitesse, pas son opposé. Le cerveau JEPA (espace de commande) se rebranche (agnostique au corps).

## Corps
- 1 tronc allongé bas : box ~0.50 (z) × 0.18 (x) × 0.09 (y), centre à y~0.18 (bas, large base). Rigide
  (pas de colonne pour le 1er hexapode — simplicité ; on pourra ajouter une flex plus tard).
- 6 pattes en 3 paires le long du tronc : z = {+0.18 (avant), 0 (milieu), −0.18 (arrière)}, x = ±0.09.
- Chaque patte = upper (capsule) + lower (capsule), comme l'actuel. Posture SPRAWL (insecte) : pattes
  écartées sur les côtés → base large, très stable. Le sprawl est BON ici (≠ quadrupède : 6 appuis).
- DOF par patte : hip_x (balancement sagittal = propulsion), hip_z (abduction), knee (flexion). 3×6=18.
- Pieds = les 6 "lower". Friction grippante (7.0, acquis de la saga).

## CPG — démarche TRÉPIED
- 2 trépieds antiphase : A = {avant-G, milieu-D, arrière-G}, B = {avant-D, milieu-G, arrière-D}.
  phase_offset = 0.0 pour A, 0.5 pour B. (3 pattes au sol pendant que 3 balancent → toujours stable.)
- Par patte : hip_x = stride·sin(2π·(phase+offset)) ; knee lift = lift·max(0,cos(...)) (lever en swing).
- Stride scale avec vx ; **cadence couplée à vx** (cpg_speed_cadence_k, acquis : foulée+fréquence
  montent ensemble = le chemin de vitesse stable). Virage : différentiel G/D du stride (skid) — pas de
  tank/pivot. Pas de colonne → le virage = skid + éventuellement abduction hip_z.
- Pas de S-wave (pas de colonne). La propulsion vient des 6 pattes.

## Dims / contrat (à propager)
- action_dim = 18 (6 pattes × 3). Pas de spine.
- BODY_NAMES = [trunk, L1u,L1l, R1u,R1l, L2u,L2l, R2u,R2l, L3u,L3l, R3u,R3l] = 13 corps.
- proprio = 7 (tronc lin+ang vel) + 13×6 (orient) + 6 (contacts) + 3 (COM) + 18 (angles) + 18 (vit) +
  2 (clock) = 132. obs = 132 + 12 (vision/commande) = 144.
- Fichiers à toucher : sylvan_agent.gd (corps+dof_config+CPG+proprio), constants.py (PROPRIO_DIM=132,
  ACTION_DIM=18), observation_builder.gd (PROPRIO_DIM), action_adapter.gd (action_dim=18),
  ppo/symmetry.py (NOUVELLE carte miroir hexapode : L↔R = avant-G↔avant-D etc., hip_z négué).
- Symétrie sagittale : swap gauche↔droite des 3 paires ; signes latéraux (hip_z, lin vel x, ang vel y/z,
  COM x, orient x-comp) négués comme avant. Reconstruire _build_proprio_maps + _ACT_PERM/SIGN pour 18.

## Plan d'exécution (portes de validation)
1. **Corps + CPG trépied** dans sylvan_agent.gd (ou une variante). PORTE 1 : run CPG pur open-loop —
   le trépied AVANCE-t-il vite (disp) ET reste stable (ne tombe pas, hauteur OK) ? Tester vx 0.4→1.0 +
   cadence couplée. Si oui → continuer ; si non → diagnostiquer le gait avant d'investir.
2. Propager les dims (constants, obs_builder, action_adapter) → le sim tourne sans erreur.
3. Reconstruire la carte de symétrie hexapode + self_check.
4. Entraîner DE ZÉRO (CPG bootstrap + résidu) avec reward omni (vx/yaw tracking) + curriculum + grip +
   cadence couplée + symétrie. Viser une vitesse nettement > 0.2 m/s, fluide.
5. Eval + check visuel.

NB acquis réutilisables : cadence-couplée (cpg_speed_cadence_k), grip (foot_friction 7), reward
locomotion_omni_v1 (balanced, coupling vitesse↔virage), curriculum commandes, symétrie+mirror-augment,
ops (kill par PID + verify). Le cerveau JEPA inchangé (espace commande).
