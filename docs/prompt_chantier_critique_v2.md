# Prompt — chantier « le critique appris du sprint » (monde v2, forme IC+TC)

> Copier-coller comme premier message d'une session fraîche. Écrit le 2026-07-17, à la clôture de la
> session monde-v2 (G-place licencié par l'owner, angle mort de la règle corrigé à découvert).

---

Sylvan — chantier d'IMPLÉMENTATION : le **critique appris du sprint** à l'étage waypoint, forme
`note = inné + correction` (IC+TC). Branche `feat/critic-clean-foundations`. Venv `env_pytorch_3.12`,
CPU, `PYTHONPATH=python`, racine. Re-checker : orphelins (`pgrep -xc godot`), git log, disque ≥ 2 G.

LIRE D'ABORD : 1) **`docs/design_monde_v2_risque.md`** (le doc du chantier : monde v2, économie du
sprint, G-place licencié, CRITÈRE PRÉ-ENREGISTRÉ du critique) ; 2) `docs/design_critique_waypoint.md`
(les 3 négatifs du critique-douleur : leurs leçons sont des CONTRAINTES) ; 3. `memory/MEMORY.md` +
fin de `memory/sylvan-mode1-build.md` (2026-07-16/17).

## Le problème, MESURÉ (ne pas re-découvrir)
Monde v2 (`SYLVAN_HAZARD_ENGULF_P=0.5`, `SYLVAN_HEALTH_REGEN=0.05`) : la bouffe naît 1 fois sur 2 AU
CŒUR du danger → manger = sprint douloureux (~27 dégâts) dont la rentabilité dépend des DRIVES + de
la SANTÉ, invisibles à la géométrie. Mesuré à 24 vies/bras ×2 seeds : géométrie **34 repas/11 morts**
poolés ; oracle-sprint (règle-triche 3 seuils : bloqué + santé>60 + énergie<50 + cible<3 m) **47/9**
= le plafond. La place existe (Pareto-dominance sur les 2 seeds). L'oracle est un bouche-trou DÉCLARÉ :
il meurt le jour où le critique passe son gate.

## Le design imposé par les 3 négatifs de la veille (contraintes, pas des options)
1. **IC+TC, jamais remplacement** : l'analytique complet (longueur + marges vertes) reste le socle —
   sa CONSISTANCE inter-replans est ce qui a tué les scoreurs appris (notes MC par état = choix
   flottants, 2× mesuré). La correction apprise s'ajoute PAR-DESSUS et n'apprend QUE l'arbitrage du
   sprint (quand outrepasser la géométrie).
2. **Entrées de la correction** : drives (énergie, soif), SANTÉ (désormais au payload/corpus), et la
   douleur prédite du candidat (checkpoint bankée `waypoint_pain_v3`, AUC 0.894 — le savoir existe,
   il n'a jamais eu le bon rôle). Symétrie miroir par canonicalisation (featurizer existant).
3. **Labels du vécu** : issue des décisions de sprint/refus (repas obtenu, dégâts payés, pas vécus).
   Corpus : 4×24 vies instrumentées sur disque (`critic_kin_g24*/decisions.jsonl` + logs Godot dans
   `data/gate_logs/`) — MAIS collectées SANS ε : compléter par une collecte exploratoire monde-v2
   (`SYLVAN_WP_EXPLORE_EPS≈0.15`, machinerie existante) pour les contrefactuels (leçon boucle
   auto-confirmante).
4. **Le juge : critère PRÉ-ENREGISTRÉ (angle mort corrigé — repas ET morts, POOLÉS)** :
   sur 2×24 vies (seeds 1+2), **apprenant ≥ géométrie +8 repas poolés ET morts ≤ +2 poolées**.
   Références : géométrie 34/11 ; plafond 47/9. Récupérer ≥ la moitié de l'écart = 41+.
5. **Bruit de l'instrument** : ±5 repas par 24-total à seed identique (non-déterminisme TCP mesuré).
   AUCUN verdict sur des écarts < ce plancher ; tout gate serré = poolé ou répété.

## Discipline (non négociable, cf CLAUDE.md)
Diagnostic GRATUIT avant tout run ; critères/KILL écrits avant ; négatif = commité ; collecte
SÉQUENTIELLE ; `/sylvan-kill` + 0 orphelin ; ne JAMAIS stager `godot/scripts/main.gd` ni `ui/`
(hooks hazard+v2 locaux : vérifier `grep -c hazard_manager godot/scripts/main.gd` ≥ 8) ; carte
`architecture.json` à jour DANS LE MÊME COMMIT ; commits Conventional anglais sans attribution IA.

## Fichiers probablement touchés
`python/sylvan/control/waypoint_layer.py` (hook de la correction dans decide(), à côté de la règle
oracle qu'elle remplacera) ; `python/scripts/train_waypoint_pain.py` (gabarit trainer/labels/CV) ;
nouveau `python/scripts/train_sprint_critic.py` ; `scripts/collect_critic_corpus_kin.sh` (harnais tel
quel, env monde-v2) ; `diagnostics/diag_hazard_gate.py` (parse tel quel) ; carte.
