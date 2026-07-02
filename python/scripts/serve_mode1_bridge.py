r"""Mode-1 <-> Mode-2 BRIDGE : dual-process controller served as ONE TCP server.

Deux cerveaux, un seul socket (contrat Godot INCHANGE : payload {proprio, retina, vision,
vision_water, energy, thirst} -> {action[18], command[vx, omega]}).

  * Mode-1 (REFLEXE, rapide)      = _Mode1Service (serve_mode1.py)  : BC DriveSymmetricPolicy
                                    -> (vx, omega) -> residu gele -> action[18].  Aucun WM, aucun
                                    look-ahead : pure reaction perception->action.
  * Mode-2 (DELIBERATION, lente)  = _PlannerService (serve_planner_command.py) : WM + slot + MPC
                                    (look-ahead en espace-commande).  C'est le forager multi-pulsions
                                    vivant (survie mediane ~2075-2300 en solo).

POLITIQUE DU PONT : Mode-1 pilote PAR DEFAUT ; on DEFERE a Mode-2 uniquement quand une pulsion est
presque vide (panique de mort imminente).  Idee = payer le look-ahead cher SEULEMENT aux moments
critiques ou l'arbitrage myope du reflexe tue, garder la vitesse du reflexe le reste du temps.

===============================================================================================
   /!\  /!\  SCAFFOLD -- DECLENCHEUR CODE-MAIN, PLACEHOLDER, PAS LA FORME FINALE  /!\  /!\
===============================================================================================
Le declencheur `min(energy, thirst)/100 < seuil` est un ECHAFAUDAGE code-main (compte-a-rebours de
mort brut).  Prouve par diag_bridge_trigger.py : AUC 0.999 MAIS tautologique (~100-150 pas d'avance
seulement).  Ce n'est PAS une vraie solution -- juste un "paniquer quand on va mourir".

    LE VRAI DECLENCHEUR (principe, TODO) = DERIVE DU MODELE, pas code-main :
      - l'INCERTITUDE de Mode-1 lui-meme (entropie / log_std de la politique), OU
      - la SURPRISE du WM (erreur de prediction).
    -> defere quand le reflexe "ne sait pas" / quand le monde surprend le modele.  Travail futur.

Ce fichier FLAGGE l'echafaudage a 3 endroits (docstring ici, bloc-commentaire au declencheur,
ligne de log au demarrage) pour qu'il ne soit JAMAIS pris pour la solution finale (CLAUDE.md §2).
===============================================================================================

KEEP MODE-2 WARM (point de conception cle) : la perception de Mode-2 est STATEFUL (EMA du radar
eau/food, slot object-centric, cadence de replan).  Si Mode-2 n'etait reveille QU'AUX ~6 % de pas
de panique avec un etat FROID (EMA vierge, slot non suivi), il arbitrerait mal et le test serait
confondu.  => on appelle `planner.predict_full` A CHAQUE PAS.  Cela garde EMA/slot CHAUDS, et comme
le MPC cher est DEJA auto-gate en interne par `replan_every` (il ne tourne qu'un pas sur K), le cout
reste ~= celui du forager solo.  On n'UTILISE la commande de Mode-2 que quand le declencheur tire ;
sinon on renvoie la commande du reflexe.  (C'est le "fallback acceptable" du cahier des charges,
mais il est en fait quasi-optimal ici puisque le MPC est deja gate.)

Usage (depuis la racine) :
    SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \\
    PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.serve_mode1_bridge \\
        --wm data/checkpoints/wm_objcentric_s1/wm_best.pt \\
        --residual data/checkpoints/hexapod_v2/policy_best.pt \\
        --bc-policy data/checkpoints/mode1_bc/policy.pt \\
        --host 127.0.0.1 --port 6062 --horizon 80 --replan-every 10 --trigger-thr 0.15
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
from pathlib import Path

from scripts.serve_mode1 import _Mode1Service
from scripts.serve_planner_command import _PlannerService
from sylvan.config import SylvanConfig
from sylvan.control.planning.command_planner import CommandPlanConfig


class _BridgeService:
    """Dispatch dual-process : garde les DEUX modes chauds, choisit lequel PILOTE selon le declencheur.

    - Mode-1 (reflexe) et Mode-2 (planner) sont interroges A CHAQUE PAS (etat chaud des deux cotes ;
      le MPC de Mode-2 est auto-gate par replan_every, donc pas de surcout par-pas).
    - `defer = min(energy, thirst)/100 < trigger_thr` -> on renvoie la reponse de Mode-2, sinon Mode-1.
    - Compte le defer-rate (fraction de pas pilotes par Mode-2) et le logge a chaque reset d'episode.
    """

    def __init__(self, mode1: _Mode1Service, planner: _PlannerService, trigger_thr: float) -> None:
        self.mode1 = mode1
        self.planner = planner
        self.trigger_thr = float(trigger_thr)
        self.action_dim = mode1.action_dim  # 18 (partage : utilise par les fallbacks TCP du handler)
        self._lock = threading.Lock()       # protege UNIQUEMENT les compteurs (chaque service a son lock)
        # Compteurs de defer -- cumul (tout l'episode courant) et global (tous les episodes)
        self._steps = 0
        self._defers = 0
        self._steps_total = 0
        self._defers_total = 0
        self._episode = 0

    def predict_full(self, payload: dict) -> dict:
        # --- Garder les DEUX modes CHAUDS : on interroge les deux services a chaque pas. ---
        # Mode-2 met a jour sa perception stateful (EMA radar eau/food + slot) a chaque pas ; son MPC
        # cher est auto-gate par replan_every (ne tourne qu'un pas sur K). Ainsi, au moment ou l'on
        # defere, la commande de Mode-2 repose sur un etat perceptif deja chaud (pas un demarrage froid).
        m1_resp = self.mode1.predict_full(payload)
        m2_resp = self.planner.predict_full(payload)

        energy = float(payload.get("energy", 0.0))
        thirst = float(payload.get("thirst", 100.0))

        # =====================================================================================
        #  /!\ SCAFFOLD -- DECLENCHEUR CODE-MAIN (PLACEHOLDER, PAS LA FORME FINALE) /!\
        # -------------------------------------------------------------------------------------
        #  `min(energy, thirst)/100 < seuil` = compte-a-rebours de mort brut (panique tardive).
        #  Tautologique (AUC 0.999 mais ~100-150 pas d'avance seulement, cf diag_bridge_trigger.py).
        #  VRAI declencheur (TODO) = derive du modele : INCERTITUDE de Mode-1 (entropie/log_std) OU
        #  SURPRISE du WM (erreur de prediction).  NE PAS presenter ce seuil comme une vraie solution.
        # =====================================================================================
        min_drive = min(energy, thirst) / 100.0
        defer = min_drive < self.trigger_thr

        with self._lock:
            self._steps += 1
            self._steps_total += 1
            if defer:
                self._defers += 1
                self._defers_total += 1
            # Log periodique du defer-rate GLOBAL : Godot ne renvoie pas toujours 'reset' entre episodes
            # (et le serveur est kill -9 en fin de run -> pas de finally), donc on trace ici pour ne jamais
            # perdre la mesure cle du pont.
            if self._steps_total % 2000 == 0:
                rate = 100.0 * self._defers_total / self._steps_total
                print(f"[bridge] {self._steps_total} pas cumules — DEFER-RATE global = {rate:.1f}% "
                      f"({self._defers_total} deferes a Mode-2)", flush=True)

        if defer:
            resp = dict(m2_resp)
            resp["mode"] = 2          # champ diagnostique (Godot l'ignore) : quel cerveau a pilote
        else:
            resp = dict(m1_resp)
            resp["mode"] = 1
        return resp

    def reset(self) -> None:
        # Logge le defer-rate de l'episode qui se termine, puis reinitialise les deux services.
        with self._lock:
            if self._steps > 0:
                rate = 100.0 * self._defers / self._steps
                print(f"[bridge] episode {self._episode} termine : {self._steps} pas, "
                      f"{self._defers} deferes a Mode-2 ({rate:.1f}%)", flush=True)
            self._episode += 1
            self._steps = 0
            self._defers = 0
        self.mode1.reset()
        self.planner.reset()

    def defer_summary(self) -> str:
        with self._lock:
            if self._steps_total == 0:
                return "[bridge] aucun pas."
            rate = 100.0 * self._defers_total / self._steps_total
            return (f"[bridge] GLOBAL : {self._steps_total} pas, {self._defers_total} deferes a "
                    f"Mode-2 -> DEFER-RATE = {rate:.1f}%")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            # Robustesse : jamais tuer le thread handler (sinon serveurs orphelins). Catch -> fallback sur.
            try:
                payload = json.loads(raw.decode("utf-8"))
                if payload.get("reset"):
                    self.server.service.reset()
                    resp = {"action": [0.0] * self.server.service.action_dim,
                            "command": [0.5, 0.0]}
                else:
                    resp = self.server.service.predict_full(payload)
            except Exception as exc:  # noqa: BLE001 -- deliberement large : ne jamais crasher le serveur
                print(f"[bridge] request error: {exc!r} — sending safe fallback", flush=True)
                resp = {"action": [0.0] * self.server.service.action_dim,
                        "command": [0.5, 0.0], "error": str(exc)}
            self.wfile.write(json.dumps(resp).encode("utf-8") + b"\n")
            self.wfile.flush()


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr, service):
        super().__init__(addr, _Handler)
        self.service = service


def main() -> None:
    ap = argparse.ArgumentParser(description="Sylvan Mode-1 <-> Mode-2 BRIDGE TCP server (dual-process)")
    # -- checkpoints Mode-2 (planner) --
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_s1/wm_best.pt",
                    help="WM object-centric (slot DANS le WM) pour Mode-2")
    ap.add_argument("--residual", default="data/checkpoints/hexapod_v2/policy_best.pt",
                    help="Residu gele (hexapod_v2), PARTAGE par les deux modes")
    # -- checkpoint Mode-1 (reflexe) --
    ap.add_argument("--bc-policy", default="data/checkpoints/mode1_bc/policy.pt",
                    help="Politique BC Mode-1 (DriveSymmetricPolicy)")
    # -- reseau --
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "6062")))
    # -- planner (Mode-2) --
    ap.add_argument("--horizon", type=int, default=80, help="Horizon MPC de Mode-2")
    ap.add_argument("--replan-every", type=int, default=10, help="Cadence de replan (partagee M1/M2)")
    ap.add_argument("--energy-weight", type=float, default=2.0)
    # -- declencheur (SCAFFOLD) --
    ap.add_argument("--trigger-thr", type=float, default=0.15,
                    help="SCAFFOLD : defere a Mode-2 quand min(energy,thirst)/100 < seuil (PLACEHOLDER)")
    args = ap.parse_args()

    _cfg = SylvanConfig()  # pour valider les dims au demarrage (action_dim=18 etc.)
    del _cfg

    # Mode-1 (reflexe) et Mode-2 (planner), instancies UNE fois dans le meme process.
    mode1 = _Mode1Service(
        residual_ckpt=Path(args.residual),
        bc_policy_ckpt=Path(args.bc_policy),
        replan_every=args.replan_every,
    )
    plan_cfg = CommandPlanConfig(horizon=args.horizon, energy_weight=args.energy_weight)
    planner = _PlannerService(
        wm_ckpt=Path(args.wm),
        residual_ckpt=Path(args.residual),
        cfg=plan_cfg,
        replan_every=args.replan_every,
    )
    service = _BridgeService(mode1, planner, trigger_thr=args.trigger_thr)

    # -------------------------------------------------------------------------------------
    #  SCAFFOLD -- ligne de log de demarrage OBLIGATOIRE (owner) : le declencheur est un PLACEHOLDER.
    # -------------------------------------------------------------------------------------
    print(f"[bridge] SCAFFOLD stakes-trigger (min_drive<{args.trigger_thr}) — PLACEHOLDER; "
          f"principled trigger = Mode-1 uncertainty / WM surprise (TODO)", flush=True)
    print(f"[bridge] Mode-1 (reflexe) + Mode-2 (planner) charges dans un seul process ; "
          f"Mode-2 garde CHAUD a chaque pas (MPC auto-gate par replan_every={args.replan_every}).",
          flush=True)

    server = _Server((args.host, args.port), service)
    print(f"[bridge] serving on {args.host}:{args.port} — Ctrl-C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print(service.defer_summary(), flush=True)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
