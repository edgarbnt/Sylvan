# Design — Monde v2 « la bouffe au cœur du danger » (2026-07-16, direction owner)

## Mission
Créer un monde où la géométrie codée-main de l'étage waypoint devient MESURABLEMENT sous-optimale,
pour donner à l'appris une vraie marge à gagner (« enrichir le monde avant le cerveau », 2ᵉ fois).
Cause : dans le monde-danger v1, l'analytique fait 14 repas sur un plafond de 15 → tout apprenant a
~1 repas de marge PAR CONSTRUCTION (les 3 négatifs closed-loop du critique-waypoint, 2026-07-16).

## À lire d'abord
- Plan complet : `~/.claude/plans/distributed-sleeping-oasis.md` (contexte, phases, risques).
- `godot/scripts/world/hazard_manager.gd` (engulf) ; `godot/scripts/agent/homeostasis.gd` (regen).
- Verdict des 3 négatifs : `docs/design_critique_waypoint.md` (fin).

## Le monde v2 (décisions owner)
1. **Bouffe parfois AU CŒUR de la zone** (`SYLVAN_HAZARD_ENGULF_P`, défaut 0 = OFF) : au spawn ET à
   chaque respawn (la zone SUIT la bouffe avec prob p — le dilemme se rejoue). Plus de détour
   possible : manger = SPRINT douloureux. La bonne décision dépend des DRIVES + de la SANTÉ, que la
   géométrie ne lit pas → c'est la place pour l'appris. Garde anti-injustice : jamais re-centrer une
   zone sur l'agent (< r+0.5).
2. **Régénération lente de santé** (`SYLVAN_HEALTH_REGEN`, défaut 0 = OFF ; valeur monde v2 : 0.05/pas)
   → la santé devient une ÉCONOMIE cyclique (encaisser, récupérer, recommencer), pas un budget à sens
   unique qui dégénère en « dépense ton capital puis évite à jamais ».

## Économie chiffrée (à vérifier au smoke)
`eat_radius=1.0` < zone `r=1.3` → pénétration ~0.3 m ≈ ~55 ticks dedans ≈ **~27 dégâts par repas
englouti** (à 0.5/pas). Regen 0.05 → récupéré en ~540 pas. Un sprint = décision rentable si affamé
et en bonne santé, ruineuse si repu ou blessé.

## Nouveau canal : la santé atteint le serveur
`health` ajouté au payload (policy_player.gd, additif) + record BC (`obs.health`) — hook local
main.gd non stagé (pattern hazard). Le futur apprenant (arbitrage du sprint) et les diagnostics en
ont besoin ; personne ne le CONSOMME encore côté décision (slack conservé pour l'instant).

## Gates PRÉ-ENREGISTRÉS (avant tout run — cf plan, Phase 2)
Monde v2 de référence : `SYLVAN_HAZARD_COUNT=1, ENGULF_P=0.5, HEALTH_REGEN=0.05` ; 12 vies seed 1.
- **(a)** analytique actuel vs **(b)** oracle-sonde « sprint calibré » (santé > 60 ET énergie < 50 →
  autoriser le direct malgré le vert ; échafaudage DÉCLARÉ, sonde jetable).
- **G-place** : (b) ≥ (a) + **4 repas** ET morts-danger (b) ≤ (a) + 1 → la place existe → chantier
  apprenant licencié (forme IC+TC, session dédiée). Échec → ajuster le MONDE (P, DAMAGE, REGEN),
  JAMAIS les seuils.
- Pré-requis Phase 0 : médiane 3 seeds du baseline v1 garde repas > 10 ET morts ≤ 2.

## Ce qu'on ne touche pas
WM (zéro retrain), dims, command_planner.py, corps cinématique. Monde v1 bit-identique (tout
défaut OFF). main.gd jamais stagé.
