#!/usr/bin/env python3
"""Hook Stop — BLOQUE une réponse qui affirme une CAUSE sans dire d'où elle sort.

POURQUOI CE HOOK EXISTE (2026-08-02, demande owner). En une session j'ai annoncé cinq fois
« la cause est X » avec assurance, et cinq fois la mesure suivante m'a contredit :
  1. « marge de 5 cm entre la couronne de baies et la bouche »  → faux, les baies BOUGENT
  2. « le rayon de braquage est le mur »                        → réfuté (agilité x2 = rien, x4 = pire)
  3. « la perception ne coûte que 2 points »                    → faux d'un facteur 7 (biais persistant)
  4. « la sélection par valeur vaut x2,32 »                     → réfuté par la théorie du régime optimal
  5. « le mouvement des objets est le chantier suivant »        → bloqué par le bruit du slot
À chaque fois j'avais mesuré la MOITIÉ de ce qu'il fallait, et présenté le résultat comme établi.
Une consigne de plus dans CLAUDE.md n'y changerait rien : il y en a déjà quatre, violées le
même jour. Ce hook rend la règle EXÉCUTABLE — il refuse la fin de tour, il ne la conseille pas.

LA RÈGLE : toute phrase qui affirme une cause, un verdict ou une preuve doit porter une étiquette
d'origine sur la même ligne :
    [MESURÉ: <commande ou fichier>]   le chiffre vient d'une mesure faite, citable
    [INFÉRÉ]                          déduit d'autres mesures, non mesuré directement
    [HYPOTHÈSE]                       pas encore testé — dit comme tel

Sortie : {"decision": "block", "reason": ...} = le tour continue et je dois corriger.

GARDE ANTI-BOUCLE : on ne bloque qu'UNE fois par message (empreinte en cache). Si je renvoie le
même texte, ça passe — le hook force une relecture, il ne prend pas la réponse en otage.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

CACHE = Path.home() / ".claude" / ".verdict_gate_seen"

# Formules qui AFFIRMENT une cause / un verdict / une preuve.
CLAIM = re.compile(
    r"(la cause[- ]racine|la cause est|le coupable|le goulot est|c'est (?:ça|ce) qui (?:tue|bloque|explique)"
    r"|ça (?:prouve|démontre)|cela (?:prouve|démontre)|ce qui prouve|est (?:donc )?(?:prouvé|établi|démontré)"
    r"|voilà pourquoi|c'est (?:donc )?(?:la|le) (?:raison|cause)|il est établi"
    r"|le (?:vrai )?(?:blocage|verrou|mur) est)",
    re.IGNORECASE,
)
TAG = re.compile(r"\[(MESURÉ|MESURE|INFÉRÉ|INFERE|HYPOTHÈSE|HYPOTHESE)\b", re.IGNORECASE)
# Formulations déjà honnêtes : on ne les embête pas.
HEDGE = re.compile(
    r"(je ne sais pas|pas établi|non concluant|reste à (?:mesurer|vérifier|trouver)|à confirmer"
    r"|hypothèse|je soupçonne|il faudrait (?:mesurer|vérifier)|n'est pas prouvé|sous réserve)",
    re.IGNORECASE,
)


def last_assistant_text(transcript: Path) -> str:
    """Dernier message de l'assistant dans le transcript JSONL."""
    text = ""
    try:
        for line in transcript.read_text(errors="ignore").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant":
                continue
            content = (rec.get("message") or {}).get("content")
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text"]
                if any(p.strip() for p in parts):
                    text = "\n".join(parts)
            elif isinstance(content, str) and content.strip():
                text = content
    except OSError:
        return ""
    return text


def offenders(text: str) -> list[str]:
    """Lignes qui affirment une cause sans étiquette d'origine ni précaution.

    ⚠️ FAUX POSITIF CORRIGÉ (2026-08-02, dès le premier déclenchement réel) : la version initiale
    bloquait sur le TITRE « ## Ce qui est établi ». Un titre de section n'affirme rien — le contenu
    dessous portait ses chiffres et ses sources. Un garde qui crie au loup sur des titres finit
    ignoré, et un garde ignoré ne vaut rien. On exige donc une vraie PHRASE : pas un titre markdown,
    pas une cellule de tableau, et un verbe conjugué autour de l'affirmation.
    """
    bad = []
    for line in text.splitlines():
        s = line.strip()
        if not s or len(s) < 40:
            continue
        if s.startswith("#") or s.startswith("|") or s.startswith(">"):
            continue  # titre, tableau, citation : pas une affirmation
        # Une phrase, pas un fragment : il faut au moins un point ou une virgule.
        if not re.search(r"[.,;]", s):
            continue
        if CLAIM.search(s) and not TAG.search(s) and not HEDGE.search(s):
            bad.append(s[:160])
    return bad[:5]


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)
    tp = payload.get("transcript_path")
    if not tp:
        sys.exit(0)
    text = last_assistant_text(Path(tp))
    if not text:
        sys.exit(0)
    bad = offenders(text)
    if not bad:
        sys.exit(0)

    # Anti-boucle : un seul blocage par contenu de message.
    fp = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
    try:
        seen = set(CACHE.read_text().split()) if CACHE.exists() else set()
    except OSError:
        seen = set()
    if fp in seen:
        sys.exit(0)
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text("\n".join(list(seen)[-200:] + [fp]))
    except OSError:
        pass

    listing = "\n".join(f"  · {b}" for b in bad)
    print(json.dumps({
        "decision": "block",
        "reason": (
            "PORTE DES VERDICTS — tu affirmes une cause sans dire d'où elle sort :\n"
            f"{listing}\n\n"
            "Pour chacune, fais UN des trois, puis termine :\n"
            "  [MESURÉ: <commande/fichier>] — la mesure existe et tu peux la citer\n"
            "  [INFÉRÉ]                     — déduit d'autres mesures, pas mesuré directement\n"
            "  [HYPOTHÈSE]                  — pas testé ; dis-le, ne l'affirme pas\n\n"
            "Et avant de re-répondre, vérifie les 3 pièges qui t'ont eu aujourd'hui :\n"
            "  1. la mesure couvre-t-elle le cas qui la FALSIFIERAIT (pas seulement celui qui la confirme) ?\n"
            "  2. le bras/facteur testé a-t-il réellement AGI (pas juste été chargé) ?\n"
            "  3. le résultat bouge-t-il avec les paramètres — une table plate ne mesure rien.\n"
        ),
    }))


if __name__ == "__main__":
    main()
