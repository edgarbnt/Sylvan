"""TABLEAU DE BORD DES GATES — où en est-on, d'un coup d'œil.

POURQUOI (docs/design_outil_matrice_information.md §4). « Aujourd'hui les verdicts sont dispersés
dans des messages de commit ; on ne peut pas voir d'un coup d'œil où on en est. » Le projet décide
par gates pré-enregistrés (CLAUDE.md §1) et il en a maintenant des dizaines : sans vue d'ensemble on
re-teste ce qui est tranché et on empile sur ce qui ne l'est pas.

CE QU'IL FAIT, ET SURTOUT CE QU'IL NE FAIT PAS. Il n'y a **aucun registre écrit à la main** : un
registre recopié dériverait du code en une semaine et mentirait — le défaut que ce projet combat
partout ailleurs. Tout est DÉRIVÉ :
  * la LISTE des gates  <- les scripts `diagnostics/diag_*.py` dont le docstring pré-enregistre un
                           critère (c'est la définition opérationnelle d'un gate ici) ;
  * la QUESTION         <- la première ligne de leur docstring ;
  * le CRITÈRE          <- le bloc « CRITÈRES PRÉ-ENREGISTRÉS / SUCCÈS / KILL » du docstring ;
  * le VERDICT          <- le sujet du dernier commit qui a touché le script, cité VERBATIM depuis
                           git. Le tableau ne réécrit jamais un verdict, il va le chercher.
  * le MODULE jugé      <- rapproché de `tools/archi_hud/architecture.json` (spec §5 : se brancher).

Un gate re-joué par `--run` est écrit dans `tools/gates/ledger.jsonl` : c'est la seule chose que cet
outil produise, et elle est datée, horodatée et distinguée du verdict DÉCLARÉ par l'historique.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_gates_board.py [--chantier obstacle]
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_gates_board.py --run info_matrix
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAGS = os.path.join(ROOT, "diagnostics")
ARCHI = os.path.join(ROOT, "tools", "archi_hud", "architecture.json")
LEDGER = os.path.join(ROOT, "tools", "gates", "ledger.jsonl")

# Un docstring qui pré-enregistre un critère = un gate. Ces marqueurs sont ceux réellement employés
# dans le dépôt ; ils ne sont pas inventés pour l'occasion.
CRITERION_MARKS = ("CRITÈRES PRÉ-ENREGISTRÉS", "CRITÈRE PRÉ-ENREGISTRÉ", "GATE GRATUIT",
                   "CRITÈRES", "SUCCÈS", "KILL", "PASS", "BARRE")
# Verdicts tels qu'ils apparaissent dans les commits du projet (anglais et français mêlés).
# VOLONTAIREMENT ÉTROIT : « proven », « beats », « measured » déclenchaient sur des commits qui ne
# parlaient pas du gate du tout — un tableau qui attribue un verdict au hasard est pire que pas de
# tableau. On n'accepte qu'un mot de verdict EXPLICITE.
VERDICTS = [
    ("ÉCHEC", ("FAIL", "ECHEC", "ÉCHEC", "REFUTED", "REFUTES", "RÉFUTÉ", "NEGATIVE", "INVALID")),
    ("STOP", ("STOP", "GELÉ", "FROZEN", "DIFFÉRÉ")),
    ("PASS", ("PASS", "POSITIVE")),
]
# Un gate est COÛTEUX si son CODE lance le monde ou un entraînement. On ne lit surtout pas le
# docstring pour ça : presque tous les diagnostics PARLENT d'entraînement — ils existent pour
# l'éviter. Se fier au discours inversait le drapeau et marquait « cher » les tests gratuits.
COSTLY_CODE = re.compile(r"subprocess|godot|serve_ppo|scripts\.train_|train_ppo|Popen")


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def docstring(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return ast.get_docstring(ast.parse(fh.read())) or ""
    except (OSError, SyntaxError):
        return ""


def criterion(doc: str) -> str:
    """Le bloc de critère pré-enregistré, tel qu'il est écrit dans le script."""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if any(m in line for m in CRITERION_MARKS[:3]):
            block = [x.strip() for x in lines[i + 1:i + 5] if x.strip()]
            return " ".join(block)[:200]
    for i, line in enumerate(lines):                       # repli : la première ligne SUCCÈS/KILL
        if re.search(r"\b(SUCCÈS|KILL|BARRE)\b", line):
            return " ".join(x.strip() for x in lines[i:i + 2])[:200]
    return ""


def verdict_word(text: str) -> str:
    for name, keys in VERDICTS:
        if any(k in text for k in keys):
            return name
    return ""


def mentions(text: str, gid: str) -> bool:
    """Le commit parle-t-il VRAIMENT de ce gate ? Sans cette garde, le tableau attribue à un gate le
    verdict du commit qui l'a effleuré — un mensonge silencieux, exactement ce qu'on veut éviter."""
    s = text.lower().replace("-", "")
    toks = [t for t in gid.split("_") if t]
    nums = [t for t in toks if re.fullmatch(r"g\d\w*", t)]
    words = [t for t in toks if len(t) > 3 and t not in nums]
    if nums and not any(n in s for n in nums):
        return False
    return any(w in s for w in words) if words else bool(nums)


def history(rel: str, gid: str) -> tuple[str, str, str, str]:
    """-> (verdict, sha, date, sujet). Le verdict vient du commit le plus récent qui porte À LA FOIS
    un mot de verdict explicite ET une mention du gate (sujet OU corps du message). Sinon « ? » et
    l'on n'affiche que le dernier commit, comme CONTEXTE — jamais comme verdict."""
    raw = git("log", "--format=%h\x1f%ad\x1f%s\x1f%b\x1e", "--date=short", "--", rel)
    entries = []
    for rec in raw.split("\x1e"):
        parts = rec.strip("\n").split("\x1f")
        if len(parts) >= 3 and parts[0]:
            entries.append((parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else ""))
    # Le SUJET prime sur le CORPS : un commit « G1 PASS » dont le corps mentionne par ailleurs un
    # chantier GELÉ rendait « gelé » pour un gate passé. Le sujet est le verdict, le corps le contexte.
    for source in ("sujet", "corps"):
        for sha, date, subject, body in entries:
            v = verdict_word(subject if source == "sujet" else body)
            if v and mentions(subject + "\n" + body, gid):
                return v, sha, date, subject
    if entries:
        sha, date, subject, _ = entries[0]
        return "?", sha, date, subject
    return "?", "", "", ""


def modules() -> dict[str, str]:
    try:
        with open(ARCHI, encoding="utf-8") as fh:
            data = json.load(fh)
        return {m["id"]: m["etat"] for m in data["modules"]} | {"__focus__": data.get("focus", "")}
    except (OSError, KeyError, json.JSONDecodeError):
        return {}


def guess_module(name: str, doc: str, known: dict[str, str]) -> str:
    """Rapproche un gate du module d'archi qu'il juge — par le mot-clé le plus long qui matche."""
    hay = (name + " " + doc[:400]).lower()
    best = ""
    for mid in known:
        if mid.startswith("__"):
            continue
        key = mid.split("_")[-1]
        if len(key) > 3 and key in hay and len(mid) > len(best):
            best = mid
    return best


def discover() -> list[dict]:
    gates = []
    for path in sorted(os.listdir(DIAGS)):
        if not path.startswith("diag_") or not path.endswith(".py"):
            continue
        full = os.path.join(DIAGS, path)
        with open(full, encoding="utf-8") as fh:
            src = fh.read()
        doc = docstring(full)
        crit = criterion(doc)
        if not (crit or re.search(r"_g\d", path)) or path == os.path.basename(__file__):
            continue                                       # le tableau n'est pas un gate de lui-même
        rel = os.path.relpath(full, ROOT)
        gid = path[len("diag_"):-3]
        verdict, sha, date, subject = history(rel, gid)
        gates.append({
            "id": gid, "script": rel, "critere": crit,
            "question": (doc.splitlines() or [""])[0].strip(),
            "sha": sha, "date": date, "subject": subject, "verdict": verdict,
            "cher": bool(COSTLY_CODE.search(src)),
        })
    return gates


def ledger() -> list[dict]:
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_gate(gate: dict, force: bool, timeout: float) -> int:
    if gate["cher"] and not force:
        print(f"🚨 « {gate['id']} » est marqué COÛTEUX (son docstring parle d'entraînement ou de "
              f"collecte).\n   Rejouer un gate cher est une décision, pas un rafraîchissement : "
              f"--force pour l'assumer.")
        return 2
    # `sys.executable` et non un chemin de venv codé : le tableau est DÉJÀ lancé par le bon
    # interpréteur, et un chemin en dur casse dès qu'on travaille depuis un worktree git.
    cmd = [sys.executable, "-u", gate["script"]]
    print(f"→ {' '.join(cmd)}\n")
    env = dict(os.environ, PYTHONPATH="python")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        out, code = proc.stdout, proc.returncode
        err = proc.stderr
    except subprocess.TimeoutExpired as exc:
        # Un gate qui ne rend pas la main est un FAIT à inscrire, pas une session bloquée.
        out, code, err = (exc.stdout or b"").decode(errors="replace"), 124, f"timeout {timeout:g} s"
    tail = out.strip().splitlines()[-25:]
    print("\n".join(tail))
    if code != 0:
        print(f"\n(échec, sortie {code}) {err.strip().splitlines()[-5:] if err else ''}")
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    entry = {"id": gate["id"], "date": dt.datetime.now().isoformat(timespec="seconds"),
             "exit": code, "script": gate["script"], "tail": "\n".join(tail[-6:])}
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n[ledger] rejoué le {entry['date']}, sortie {code} -> {os.path.relpath(LEDGER, ROOT)}")
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=None, metavar="ID", help="rejouer un gate et l'inscrire au ledger")
    ap.add_argument("--force", action="store_true", help="assumer le rejeu d'un gate coûteux")
    ap.add_argument("--timeout", type=float, default=900.0, help="secondes avant d'abandonner un rejeu")
    ap.add_argument("--chantier", default=None, help="filtre sur l'id du gate (sous-chaîne)")
    ap.add_argument("--full", action="store_true", help="afficher le critère pré-enregistré de chacun")
    args = ap.parse_args()

    known = modules()
    gates = discover()
    if args.chantier:
        gates = [g for g in gates if args.chantier in g["id"]]
    if args.run:
        match = [g for g in gates if g["id"] == args.run]
        if not match:
            raise SystemExit(f"gate inconnu : {args.run} ; connus : {[g['id'] for g in gates]}")
        return run_gate(match[0], args.force, args.timeout)

    replayed = {}
    for e in ledger():
        replayed[e["id"]] = e
    focus = known.get("__focus__", "")
    sym = {"PASS": "✅", "ÉCHEC": "❌", "STOP": "⏸️ ", "?": "  "}

    print(f"TABLEAU DE BORD DES GATES — {len(gates)} gates découverts dans diagnostics/")
    print(f"focus de la carte d'archi : « {focus} » ({known.get(focus, '?')})")
    print(f"{'gate':<26} {'v':<3} {'dernier commit':<12} {'module jugé':<20} sujet du commit")
    print("-" * 122)
    counts = {"PASS": 0, "ÉCHEC": 0, "STOP": 0, "?": 0}
    for g in sorted(gates, key=lambda g: (g["date"] or "0000"), reverse=True):
        counts[g["verdict"]] += 1
        mod = guess_module(g["id"], g["question"], known)
        tag = f"{mod}{'*' if mod == focus else ''}" if mod else "—"
        mark = "$" if g["cher"] else " "
        rej = f" [rejoué {replayed[g['id']]['date'][:10]}]" if g["id"] in replayed else ""
        print(f"{g['id']:<26}{mark}{sym[g['verdict']]:<3}{g['date'] or '—':<12} {tag:<20} "
              f"{(g['subject'] or 'jamais commité')[:56]}{rej}")
        if args.full and g["critere"]:
            print(f"{'':<26}    critère pré-enregistré : {g['critere'][:110]}")
    print("-" * 122)
    print(f"  ✅ {counts['PASS']} passés   ❌ {counts['ÉCHEC']} échoués/réfutés   "
          f"⏸️ {counts['STOP']} gelés   ? {counts['?']} sans verdict lisible dans le commit")
    print("  $ = gate COÛTEUX : son CODE lance le monde ou un entraînement · "
          "* = module au focus de la carte")
    print("  Le verdict est CITÉ depuis le commit le plus récent qui porte un mot de verdict ET\n"
          "  mentionne ce gate : ce tableau ne réécrit rien. « ? » = aucun commit ne porte de\n"
          "  verdict lisible pour ce gate — ce n'est PAS « non testé », c'est « non retrouvable ici ».")
    print(f"  Rejouer : --run <gate>  (inscrit dans {os.path.relpath(LEDGER, ROOT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
