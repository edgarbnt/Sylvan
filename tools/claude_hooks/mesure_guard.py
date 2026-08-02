#!/usr/bin/env python3
"""Hook PostToolUse(Bash) — rappelle les contrôles de dégénérescence après un diagnostic ou un A/B.

POURQUOI (2026-08-02). Deux verdicts vides ont été rendus le même jour, et les deux auraient été
attrapés par un contrôle de dix secondes :
  · une ABLATION dont les 16 lignes rendaient 100 % — table PLATE, donc test qui ne mesure rien
    (budget de simulation irréaliste) ; même chose pour la norme homéostatique, désaccord identique
    pour tous les exposants ;
  · un A/B dont le bras expérimental n'avait JAMAIS agi (échafaudage posé sur une sortie du planner
    atteinte 16,8 % du temps). La bannière au démarrage prouvait le CHARGEMENT, pas l'EXÉCUTION.

Ce hook n'interdit rien et ne bloque rien : il réinjecte les trois questions au moment exact où
elles servent — juste après la commande qui produit le chiffre, avant qu'il ne soit interprété.
"""

from __future__ import annotations

import json
import re
import sys

# Commandes qui PRODUISENT un chiffre destiné à être interprété.
TRIGGER = re.compile(r"(diag_|gate_|judge_|_ab\b|baseline_|/ab_|permutation)", re.IGNORECASE)

CHECKLIST = (
    "CONTRÔLES DE MESURE (ce sont ceux qui ont attrapé deux verdicts vides le 2026-08-02) :\n"
    "  1. DÉGÉNÉRESCENCE — le résultat bouge-t-il quand tu changes les paramètres ? "
    "Une colonne identique partout = un test qui ne mesure rien, pas une robustesse.\n"
    "  2. LE BRAS A-T-IL AGI ? — mesure son MÉCANISME (la valeur qu'il est censé changer), "
    "pas sa bannière de chargement. Un bras inerte ressemble à « aucun effet ».\n"
    "  3. LA MÉTRIQUE EST-ELLE BIMODALE ? — ici la survie l'est (soit ~350, soit 3000). "
    "Juger sur une métrique PAR TEMPS VÉCU, sinon une vie chanceuse porte tout le résultat.\n"
    "  4. TON ÉTIQUETTE EST-ELLE JUSTE ? — si la vérité-terrain vient d'un oracle partiel "
    "(une seule cible dans un monde qui en sert plusieurs), tu juges ton étiquette, pas l'agent."
)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)
    cmd = ((payload.get("tool_input") or {}).get("command") or "")
    if not TRIGGER.search(cmd):
        sys.exit(0)
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": CHECKLIST,
        }
    }))


if __name__ == "__main__":
    main()
