#!/usr/bin/env python3
"""Valide tools/archi_hud/architecture.json : schéma, états, références d'id, ancres code.
Idiome repo : script autonome, sort != 0 si invalide. Lancer : env_pytorch_3.12/bin/python tools/archi_hud/validate_architecture.py"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHI = os.path.join(ROOT, "tools", "archi_hud", "architecture.json")
ALLOWED_ETATS = {"pur", "partiel", "echafaudage", "manquant"}
REQUIRED_KEYS = {"id", "etat", "titre", "couche", "quoi", "role", "comment", "apporte",
                 "etat_detail", "limites", "preuves", "code", "live_field", "depends_on"}


def validate(path: str) -> list[str]:
    errs: list[str] = []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    modules = data.get("modules")
    if not isinstance(modules, list) or not modules:
        return ["'modules' doit être une liste non vide"]
    ids = {m.get("id") for m in modules}
    if data.get("focus") not in ids:
        errs.append(f"focus '{data.get('focus')}' n'est pas un id de module")
    seen = set()
    for m in modules:
        mid = m.get("id", "?")
        missing = REQUIRED_KEYS - set(m)
        if missing:
            errs.append(f"[{mid}] clés manquantes : {sorted(missing)}")
            continue
        if mid in seen:
            errs.append(f"[{mid}] id dupliqué")
        seen.add(mid)
        if m["etat"] not in ALLOWED_ETATS:
            errs.append(f"[{mid}] etat invalide '{m['etat']}' (autorisés : {sorted(ALLOWED_ETATS)})")
        if not isinstance(m["depends_on"], list):
            errs.append(f"[{mid}] depends_on doit être une liste")
        else:
            for dep in m["depends_on"]:
                if dep not in ids:
                    errs.append(f"[{mid}] depends_on référence un id inconnu : '{dep}'")
        if m["live_field"] is not None and not isinstance(m["live_field"], str):
            errs.append(f"[{mid}] live_field doit être str ou null")
        # Sections riches du panneau : textes obligatoires + listes/null typés.
        for key in ("role", "comment", "apporte", "etat_detail"):
            if not isinstance(m[key], str) or not m[key].strip():
                errs.append(f"[{mid}] '{key}' doit être un texte non vide")
        if m["limites"] is not None and not isinstance(m["limites"], str):
            errs.append(f"[{mid}] limites doit être str ou null")
        if not isinstance(m["preuves"], list) or any(not isinstance(p, str) for p in m["preuves"]):
            errs.append(f"[{mid}] preuves doit être une liste de chaînes")
        code = m["code"]
        if code is not None:
            relpath = code.split(":", 1)[0]
            if not os.path.exists(os.path.join(ROOT, relpath)):
                errs.append(f"[{mid}] ancre code introuvable : '{relpath}'")
        elif m["etat"] != "manquant":
            errs.append(f"[{mid}] code null n'est autorisé que pour etat 'manquant'")
    return errs


def main() -> int:
    if not os.path.exists(ARCHI):
        print(f"FAIL: {ARCHI} introuvable")
        return 1
    errs = validate(ARCHI)
    if errs:
        print("FAIL:")
        for e in errs:
            print(f"  - {e}")
        return 1
    print("OK: architecture.json valide")
    return 0


if __name__ == "__main__":
    sys.exit(main())
