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

## ⭐ VERDICT DE SESSION (2026-07-16 nuit) : G-PLACE NON CONCLU — LA RÉSOLUTION DE L'INSTRUMENT EST LE VERROU

Runs (12 vies/bras, monde v2 P=0.5 regen=0.05) : géométrie 10/2 (s1) et 6/1 (s2) ; oracle-v1 14/2 (s1)
et 9/4 (s2 — 2 noyades sur vies SANS bouffe engouffrée → sonde affûtée v2 : sprint seulement si cible
<3 m) ; oracle-v2 **12/2 (s2 : +6 repas ✓✓)** mais **7/3 (s1 : −3 ✗)**. Incohérence expliquée par la
DÉCOUVERTE de la nuit : **variance run-à-run ±5 repas À SEED IDENTIQUE** (jumeaux seed-3 : 15 vs 10 —
timing TCP → trajectoires non déterministes). **Un critère à ±4 repas sur 12 vies vit DANS le bruit.**
Le monde v2 est construit, smoké, vivant ; le signal directionnel existe (3 comparaisons sur 4
favorables à la politique santé/drive-consciente, dégâts s1 : 331 vs 704) mais N=12 ne peut pas le
prouver. REPRISE (à trancher owner) : (i) gates à N≥24 vies/bras ou médianes de runs répétés pour
TOUT critère à ±4 repas — le coût de la preuve a doublé, c'est le prix de l'honnêteté ; (ii)
instrumenter les bras oracle (debug) — possible artefact de phase d'approche de la sonde v2 (cible
engouffrée >3 m : le sprint ne se déverrouille qu'en dessous) ; (iii) garde sans-cible : 5→3 morts
(sa classe soldée), attribution des 3 restantes (classe frôlements-direct) à sonder.

## GATES 24 VIES (pré-enregistrés AVANT lancement, 2026-07-17)
4 bras séquentiels, 24 vies chacun, monde v2 (P=0.5, regen=0.05), instrumentés (debug + WP_LOG) :
ordre s2(a), s2(b), s1(a), s1(b) — le paquet dur d'abord (kill précoce possible).
- **PASS global** : sur CHAQUE seed, (b) ≥ (a) + 8 repas /24 vies (échelle du +4/12) ET morts-danger
  (b) ≤ (a) + 2. → place prouvée, chantier apprenant licencié.
- **KILL** : un seed montre (b) < (a) en repas (contradiction directionnelle à N=24).
- Entre les deux → toujours non-conclu → décision owner (N=36 ou autre design).
Bruit attendu sur totaux de 24 : ~±5 (SD 12-vies ~3.5 ×√2) ; signal attendu si réel : +8-12.

## ⭐⭐ VERDICT G-PLACE 24 VIES (2026-07-17 matin) : DIRECTION CONFIRMÉE 4/4, BARRE STRICTE MANQUÉE SUR UN SEED
| seed | (a) géométrie | (b) oracle | Δ repas (gate ≥+8) | Δ morts (gate ≤+2) |
|---|---|---|---|---|
| 2 | 15 / 5 | 24 / 6 | **+9 ✓** | +1 ✓ |
| 1 | 19 / 6 | 23 / 3 | **+4 ✗** | **−3** ✓ |
Cumul 48 vies/bras : (a) 34 repas/11 morts vs (b) **47 repas/9 morts** → +13 repas ET −2 morts.
L'oracle **Pareto-domine** la géométrie sur les DEUX seeds (plus de repas ET, au seed 1, moitié
moins de morts — il a converti l'avantage en sécurité plutôt qu'en repas). Mais le critère global
pré-enregistré (≥+8 sur CHAQUE seed) n'est pas atteint → per pré-enregistrement : NON-CONCLU strict,
décision owner. Évidence accumulée : 4/4 comparaisons favorables (2 N, 2 seeds), signal poolé +13
(bruit poolé ~±10). Corpus instrumentés (decisions.jsonl ×4) sur disque pour le chantier apprenant.
