"""G10 GRATUIT — LE DÉTERMINISME DU MONDE COMPLET : deux vies à même graine sont-elles IDENTIQUES ?

PÉRIMÈTRE. Aucune collecte retenue, aucun entraînement. Deux courtes vies, on compare, on jette.

POURQUOI, ET POURQUOI C'EST BLOQUANT (design_foret_complete.md §6quater F, tranché en GATE au
§6quinquies F). La forêt multiplie les consommateurs de hasard : 45 arbres, 6 peuplements, 3
clairières, 6 distracteurs, 12 bosquets de nourriture, 12 flaques, le regard, la teinte par arbre.
Si l'un d'eux tire dans le MÊME flux que les commandes, il DÉCALE ce flux : deux runs à graine égale
divergent, et avec eux tombent **tous** les juges contrefactuels du projet — le hook SYLVAN_CF_TICK
force une commande à un tick donné et compare deux futurs ; sans rejeu bit-identique, il compare
deux mondes différents et rapporte du bruit avec l'assurance d'une mesure.

Ce n'est pas une hypothèse : le défaut (c) de G3 l'a déjà produit. Les tirages du regard venaient du
flux des commandes, donc « activer le regard » changeait la trajectoire du CORPS. La parade — un
flux de hasard DÉDIÉ par mécanique (regard +7171, distracteurs +3131, eau +777) — est en place ;
cette sonde vérifie qu'elle tient une fois TOUT allumé ensemble, ce qui n'a jamais été fait.

CE QUE LA SONDE COMPARE, ET POURQUOI ÇA SUFFIT. Deux runs du monde complet à graine identique, puis
comparaison de TOUT le corpus écrit, champ par champ. Le corpus contient la commande, la pose du
torse, la rétine entière, le regard, les positions de ressources : si quoi que ce soit dans le monde
diverge — un arbre déplacé, une flaque au mauvais niveau, un distracteur en avance d'un tick — la
rétine le voit et le hash change. Un hash identique est donc une preuve FORTE, pas un sondage.

UN SEUL CHAMP EST EXCLU, ET IL EST NOMMÉ : `info.timestamp`, l'horloge murale. Ce n'est pas une
tolérance (§2 interdirait d'assouplir le critère pour le faire passer) : c'est un champ qui ne
décrit rien de la simulation et ne peut pas être reproductible. La distinction est MESURÉE, pas
supposée — au premier passage, la sonde a échoué, et l'inspection des deux corpus a montré que sur
720 lignes `timestamp` était la SEULE clé à différer : tout le reste était déjà bit-identique. On
exclut donc un champ identifié, pas une classe d'écarts, et le selfcheck le vérifie (une commande
qui change fait toujours diverger le hash).

CRITÈRES PRÉ-ENREGISTRÉS :
  T1 REJEU IDENTIQUE .... deux runs, même graine → corpus bit-identiques (hash SHA-256 égal).
  T2 LA GRAINE COMPTE ... un run à graine DIFFÉRENTE produit un corpus DIFFÉRENT. Sans ce contrôle,
                          T1 passerait trivialement si le monde ignorait la graine (par exemple si
                          rien n'était réellement tiré) — le témoin qui rend T1 falsifiable.
  T3 LE MONDE EST SERVI . les logs prouvent que les mécaniques tournaient VRAIMENT pendant le test
                          (forêt, terrain, regard, distracteurs, flaques) : un déterminisme mesuré
                          sur un monde vide ne dirait rien du monde complet.

CE QUE LA SONDE NE DIT PAS : que le déterminisme survive à un autre nombre de threads ou à un
serveur planner réutilisé. La recette complète (graine + torch mono-thread + serveur FRAIS) reste
requise côté Python ; ici on établit la moitié Godot, celle que la forêt met en danger.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g10_determinisme.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g10_determinisme.py --selfcheck
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.world import FORET_V1  # noqa: E402

GODOT = os.path.join(ROOT, "tools", "godot", "godot")

# Les témoins que le monde a réellement tourné (§6bis : chaque module rapporte le MESURÉ).
TEMOINS = {
    "forêt": "[forest]",
    "terrain": "[terrain]",
    "regard": "[gaze]",
    "distracteurs": "[distractor]",
    "locomotion": "[locomotion]",
}


def _run(label: str, seed: int, episodes: int, steps: int) -> tuple[str, str]:
    """Un run du monde COMPLET. Renvoie (hash du corpus, stdout+stderr)."""
    run_dir = f"/tmp/foret_g10_{label}"
    os.system(f"rm -rf {run_dir}")
    e = dict(os.environ)
    e.update(FORET_V1.to_env())
    e.update({
        "SYLVAN_COLLECT": "1", "SYLVAN_WM_COLLECT": "1", "SYLVAN_COLLECTOR_MODE": "babbling",
        "SYLVAN_CPG": "1", "SYLVAN_RESIDUAL_GAIN": "0.0", "SYLVAN_TURN_FADE": "0",
        "SYLVAN_WM_WMAX": "0.6",
        "SYLVAN_POLICY_EXPLORATION_STD_INITIAL": "0", "SYLVAN_POLICY_EXPLORATION_STD_FINAL": "0",
        "SYLVAN_REFLEX_STRENGTH": "0", "SYLVAN_ASSIST_RATIO": "0",
        "SYLVAN_NUM_EPISODES": str(episodes), "SYLVAN_MAX_EPISODE_STEPS": str(steps),
        "SYLVAN_SEED": str(seed), "SYLVAN_RUN_DIR": run_dir,
    })
    p = subprocess.run([GODOT, "--path", os.path.join(ROOT, "godot"), "--headless"],
                       env=e, capture_output=True, text=True, timeout=900)
    out = p.stdout + p.stderr
    for fatal in ("Parse Error", "Failed to load script"):
        if fatal in out:
            first = next((ln for ln in out.splitlines() if fatal in ln), fatal)
            raise SystemExit(f"[{label}] Godot n'a PAS chargé le script — mesure invalide.\n  {first}")
    files = sorted(glob.glob(os.path.join(run_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"[{label}] aucun jsonl écrit — la collecte n'a rien produit")
    return _hash(files), out


# Le SEUL champ exclu de la comparaison, et il est nommé : l'HORLOGE MURALE que le collecteur écrit
# dans chaque enregistrement. Ce n'est pas un assouplissement du critère (§2) — c'est un champ qui,
# par construction, ne peut pas être reproductible et ne décrit rien de la simulation. Mesuré : sur
# 720 lignes, `info.timestamp` était la SEULE clé à différer entre deux runs de même graine ; tout le
# reste (commande, pose, rétine, regard, ressources, et le reste de `info`) était déjà bit-identique.
# On l'exclut donc PAR NOM, jamais par tolérance : n'importe quelle autre divergence fait toujours
# échouer T1.
_HORLOGE = ("info", "timestamp")


def _hash(files: list[str]) -> str:
    h = hashlib.sha256()
    for path in files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                bloc = rec.get(_HORLOGE[0])
                if isinstance(bloc, dict):
                    bloc.pop(_HORLOGE[1], None)
                h.update(json.dumps(rec, sort_keys=True).encode())
    return h.hexdigest()


def _temoins_servis(out: str) -> dict[str, bool]:
    return {nom: (marqueur in out) for nom, marqueur in TEMOINS.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print(f"MONDE : {FORET_V1.name} COMPLET ({len(FORET_V1.to_env())} variables) | "
          f"{a.episodes} vies x {a.steps} ticks\n")

    h1, out1 = _run("a", a.seed, a.episodes, a.steps)
    h2, _ = _run("b", a.seed, a.episodes, a.steps)
    h3, _ = _run("autre_graine", a.seed + 1, a.episodes, a.steps)

    print(f"  graine {a.seed}   run A : {h1[:16]}…")
    print(f"  graine {a.seed}   run B : {h2[:16]}…")
    print(f"  graine {a.seed + 1}   run C : {h3[:16]}…\n")

    ok = True
    t1 = h1 == h2
    ok &= t1
    print(f"{'✅' if t1 else '❌'} T1 REJEU IDENTIQUE   deux runs à graine {a.seed} : "
          f"{'bit-identiques' if t1 else 'DIVERGENTS — un flux de hasard est partagé'}")

    t2 = h1 != h3
    ok &= t2
    print(f"{'✅' if t2 else '❌'} T2 LA GRAINE COMPTE  graine {a.seed + 1} : "
          f"{'corpus différent' if t2 else 'IDENTIQUE — le monde ignore sa graine, T1 ne prouve rien'}")

    servis = _temoins_servis(out1)
    t3 = all(servis.values())
    ok &= t3
    detail = " | ".join(f"{n} {'✓' if v else '✗'}" for n, v in servis.items())
    print(f"{'✅' if t3 else '❌'} T3 LE MONDE SERVI    {detail}")

    print("=" * 92)
    print(f"GATE G10 = {'PASS' if ok else 'ÉCHEC'}")
    if ok:
        print("⇒ le monde complet REJOUE à l'identique : les juges contrefactuels tiennent.")
    else:
        print("⇒ BLOQUANT : sans rejeu bit-identique, tous les juges contrefactuels tombent (§6quater F).")
    return 0 if ok else 1


def selfcheck() -> int:
    assert FORET_V1.forest_count > 0 and FORET_V1.distractor_count > 0 and FORET_V1.gaze, \
        "le preset testé doit vraiment allumer les mécaniques, sinon T3 est vide"
    print(f"  [ok] foret_v1 allume bien forêt ({FORET_V1.forest_count} arbres), distracteurs "
          f"({FORET_V1.distractor_count}), regard, flaques ({FORET_V1.water_puddle_period} ticks)")

    e = FORET_V1.to_env()
    assert e.get("SYLVAN_SPEED_COST") and e.get("SYLVAN_TERRAIN_SLOW")
    print(f"  [ok] to_env() émet {len(e)} variables — le monde comparé est bien le monde complet")

    # contrôle du comparateur : SEULE l'horloge est ignorée, tout le reste fait échouer
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        base = {"info": {"timestamp": "T1", "seed": 1}, "wm": {"cmd": [0.6, 0.1]}}
        autre_horloge = {"info": {"timestamp": "T2", "seed": 1}, "wm": {"cmd": [0.6, 0.1]}}
        autre_monde = {"info": {"timestamp": "T1", "seed": 1}, "wm": {"cmd": [0.6, 0.2]}}
        def ecrire(nom, rec):
            f = os.path.join(d, nom)
            open(f, "w").write(json.dumps(rec) + "\n")
            return [f]
        assert _hash(ecrire("a.jsonl", base)) == _hash(ecrire("b.jsonl", autre_horloge)), \
            "l'horloge murale doit être ignorée"
        assert _hash(ecrire("a.jsonl", base)) != _hash(ecrire("c.jsonl", autre_monde)), \
            "une commande différente DOIT faire diverger le hash"
    print("  [ok] le comparateur ignore l'horloge murale et RIEN d'autre "
          "(une commande qui change fait diverger)")

    assert os.path.exists(GODOT), GODOT
    print("  [ok] binaire Godot présent")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
