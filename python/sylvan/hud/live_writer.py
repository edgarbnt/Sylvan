"""Écriture atomique de data/hud/live.json (valeurs runtime pour l'Archi-HUD).
Pas d'horloge murale (règle projet) : `ts` est un entier monotone fourni par l'appelant."""
import json
import os
import tempfile


def write_live(path: str, *, ts: int, episode: int, step: int, fields: dict) -> None:
    """Écrit {ts, episode, step, fields} en JSON, de façon atomique (tmp + os.replace).
    Crée le dossier parent au besoin. Ne lève jamais sur un champ non-sérialisable :
    les valeurs non-JSON sont ignorées (le HUD est best-effort, jamais bloquant)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    safe_fields = {}
    for k, v in fields.items():
        try:
            json.dumps(v)
            safe_fields[k] = v
        except (TypeError, ValueError):
            continue
    payload = {"ts": int(ts), "episode": int(episode), "step": int(step), "fields": safe_fields}
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
