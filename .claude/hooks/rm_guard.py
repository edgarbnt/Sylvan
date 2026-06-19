#!/usr/bin/env python3
"""PreToolUse(Bash) guard: block catastrophic deletions.

Triggered by the 2026-06-16 incident where `find . -path '*cr*' -prune -exec rm -rf`
deleted ALL source (scripts/critic contain "cr"). Blocks the clearly-dangerous shapes
while allowing normal scoped cleanups (e.g. `rm -rf data/replay_buffer/foo*`).

Hook contract: reads JSON on stdin, exit 0 = allow, exit 2 = BLOCK (stderr → Claude).
"""
import json
import re
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)  # can't parse → don't get in the way

cmd = (payload.get("tool_input") or {}).get("command", "")
if not cmd:
    sys.exit(0)

reasons = []

# 1) find ... (-path|-name '*X*') ... (-exec rm | -delete)  — the exact 2026-06-16 disaster shape.
if re.search(r"\bfind\b", cmd) and (re.search(r"-exec\s+rm\b", cmd) or re.search(r"\b-delete\b", cmd)):
    if re.search(r"-(path|name|wholename)\s+['\"]?\*[^'\" ]*\*", cmd):
        reasons.append("`find` + double-wildcard `-path/-name '*X*'` + rm/-delete "
                       "(c'est exactement le glob qui a supprimé tout le code le 2026-06-16).")

# 2) rm -r/-f with a dangerous target.
for m in re.finditer(r"\brm\b[^\n;|&]*", cmd):
    seg = m.group(0)
    if not re.search(r"-[a-z]*[rf]", seg):
        continue  # not recursive/forced → harmless
    # double-wildcard token like *cr* (matches anything containing X) — very broad.
    if re.search(r"(^|\s)\S*\*\S*\*\S*", seg):
        reasons.append("`rm -rf` avec un glob à double wildcard `*X*` (trop large — scoper le chemin).")
    # root / home / cwd / bare star / unexpanded var (could be empty → rm -rf /).
    if re.search(r"\s(/|~|\.|\.\.|\*|\$[A-Za-z_])(\s|/|$)", seg):
        reasons.append("`rm -rf` sur une cible dangereuse (/, ~, ., .., *, ou $VAR potentiellement vide).")

if reasons:
    print("⛔ Suppression BLOQUÉE par le garde-fou rm (.claude/hooks/rm_guard.py) :", file=sys.stderr)
    for r in dict.fromkeys(reasons):  # dedup, keep order
        print(f"  - {r}", file=sys.stderr)
    print("Si c'est intentionnel : scoper le chemin explicitement, ou lister d'abord "
          "ce qui matche (`ls`/`find ... -print`) avant de supprimer.", file=sys.stderr)
    sys.exit(2)

sys.exit(0)
