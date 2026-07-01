"""Mode-1 Phase 2 : serveur de COLLECTE RL (politique stochastique) + résidu gelé.

Fork de scripts.serve_mode1 (Task 4, déterministe). Deux changements :
  (a) à chaque tick de replan, ÉCHANTILLONNE la commande dans la gaussienne de la politique
      drive-symétrique : z, logprob = policy.sample(...) ; commande actionnée = map_action(z),
      tenue K ticks à travers le résidu gelé (comme serve_mode1) ;
  (b) LOGGE des transitions en ESPACE-COMMANDE (cadence de replan) dans un replay buffer JSONL,
      avec la RÉCOMPENSE de survie calculée CÔTÉ SERVEUR (miroir exact de
      reward_manager.gd::_reward_survival_pure), pour train_mode1_ppo (Task 8).

Pourquoi côté serveur : la politique Mode-1 agit en ESPACE-COMMANDE (obs=(proprio,tokens),
action=(vx,ω) 2-D) décidée tous les K ticks ; le writer JSONL de Godot logge des transitions
18-D résidu par-tick (mauvais espace) → inutilisable ici. Le serveur reçoit energy+thirst chaque
tick → il calcule lui-même la récompense.

Frontières d'épisode : signal EXPLICITE de Godot (plus d'inférence par niveau de drive — ambiguë :
manger+boire à plein en pleine vie fabriquait une fausse frontière, une troncation drives-pleins en
ratait une). policy_player.gd::predict_planner envoie désormais :
  - "episode_step" = index de pas DANS l'épisode (episode_manager.current_step_id, remis à 0 au respawn) ;
  - "prev_term" ∈ {"none","death","truncated"} = raison de fin de l'épisode PRÉCÉDENT, portée au 1er
    tick du nouvel épisode (predict_planner tourne AVANT que `done` soit connu → on tague le 1er tick
    post-respawn, pas le tick terminal).
Détection = CHUTE de episode_step (respawn → 0). Classification depuis prev_term ("death"→done terminal
bootstrap 0 ; "truncated"→truncated non-terminal bootstrap V). Fallback drive-level LOGGÉ seulement si le
champ explicite est absent (rétro-compat serveur ancien Godot).

Schéma JSONL (une ligne = une macro-transition ; contrat consommé par Task 8) :
  {"proprio":[132],"retina":[144],"energy":float,"thirst":float,
   "command_raw":[z0,z1],"command_act":[vx,om],
   "reward":float,"steps":int,"done":bool,"truncated":bool}

Usage (depuis la racine) :
    PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.serve_mode1_collect \\
        --residual data/checkpoints/hexapod_v2/policy_best.pt \\
        --policy data/checkpoints/mode1_bc/policy.pt \\
        --out data/replay_buffer/mode1_rl_a --seed 1 \\
        --host 127.0.0.1 --port 6053 --replan-every 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socketserver
import threading
from pathlib import Path
from typing import Callable

import torch

from sylvan.config import SylvanConfig
from sylvan.control.mode1.policy import DriveSymmetricPolicy, map_action
from sylvan.control.mode1.obs import build_tokens
from sylvan.control.mode1.residual import residual_action
from sylvan.control.ppo.policy import GaussianActorCritic
from sylvan.training.checkpointing import load_checkpoint

# Commande neutre par défaut (centre du régime propre hexapode)
_DEFAULT_CMD: tuple[float, float] = (0.65, 0.0)

# Seuils du FALLBACK drive-level (utilisé UNIQUEMENT si le signal explicite episode_step/prev_term est
# absent — rétro-compat serveur ancien Godot ; le chemin normal ne s'en sert plus).
RESET_LEVEL = 99.5   # reset_state() met les 2 drives à 100 → "les deux ≥ 99.5" = respawn
DEATH_LEVEL = 1.0    # is_critical à <=0 ; le dernier tick observé avant reset a un drain ~0.05..0.13


def _finite_list(xs) -> list:
    """Remplace tout float non-fini (NaN/±inf) par 0.0. `clamp` NE retire PAS les NaN et json.dumps émet
    un token `NaN` littéral = JSON invalide qui empoisonne le buffer Task-8. Miroir du nan_to_num du résidu."""
    return [(v if math.isfinite(v) else 0.0) for v in (float(x) for x in xs)]


def _finite_f(v) -> float:
    v = float(v)
    return v if math.isfinite(v) else 0.0


class Mode1CollectState:
    """Machine à états PURE (sans Godot/torch) : fenêtre de récompense + ouverture/fermeture des
    macro-transitions + classification done/truncated. Testable en isolation.

    `on_tick` est appelée UNE fois par décision-serveur. `sample_fn` (fourni par le serveur) renvoie
    (command_raw, command_act) ; il n'est appelé QU'aux ticks de replan (ouverture d'une transition)."""

    def __init__(
        self,
        replan_every: int,
        pain_shaping_w: float = 0.0,
        death_level: float = DEATH_LEVEL,
        reset_level: float = RESET_LEVEL,
    ) -> None:
        self.replan_every = max(1, int(replan_every))
        self.pain_shaping_w = float(pain_shaping_w)
        self.death_level = float(death_level)
        self.reset_level = float(reset_level)

        self._open: dict | None = None      # macro-transition ouverte (None avant le 1er tick)
        self._prev_e: float | None = None    # drives du tick précédent (fallback drive-level uniquement)
        self._prev_t: float | None = None
        self._prev_step: int | None = None   # episode_step du tick précédent (détection de frontière explicite)
        self._episode_tick = 0               # ticks depuis le début de l'épisode courant (phase replan)
        self._force_replan = True            # le tout 1er tick (et chaque début d'épisode) = replan
        self._last_cmd: list = list(_DEFAULT_CMD)  # dernière commande valide (fallback si sample_fn échoue)
        self._warned_fallback = False        # log une seule fois si on retombe sur l'heuristique drive-level

    # ------------------------------------------------------------------ #
    # Détection / classification des frontières d'épisode
    # ------------------------------------------------------------------ #
    def _detect_boundary(self, episode_step, e: float, t: float) -> bool:
        """Frontière = CHUTE de episode_step (respawn → 0). Fallback drive-level si le champ est absent."""
        if episode_step is not None and int(episode_step) >= 0:
            prev = self._prev_step
            self._prev_step = int(episode_step)
            if prev is None:
                return False
            return int(episode_step) < prev  # respawn : le compteur intra-épisode est retombé
        # --- FALLBACK (rétro-compat) : ancien front montant "les deux drives ≥ reset_level" ------------
        if not self._warned_fallback:
            print("[serve-mode1-collect] WARN: payload sans 'episode_step' → fallback frontière "
                  "drive-level (Godot obsolète ?)", flush=True)
            self._warned_fallback = True
        if self._prev_e is None:
            return False
        both_now = e >= self.reset_level and t >= self.reset_level
        both_prev = self._prev_e >= self.reset_level and self._prev_t >= self.reset_level
        return both_now and not both_prev

    def _classify(self, prev_term) -> tuple[bool, bool]:
        """(done, truncated) depuis la raison EXPLICITE. Fallback drive-level LOGGÉ si absente/inconnue."""
        if prev_term == "death":
            return True, False
        if prev_term == "truncated":
            return False, True
        # prev_term "none"/absent à une VRAIE frontière = anormal → log + heuristique drive-level.
        pe = self._prev_e if self._prev_e is not None else 100.0
        pt = self._prev_t if self._prev_t is not None else 100.0
        print(f"[serve-mode1-collect] WARN: frontière avec prev_term={prev_term!r} → classification "
              f"drive-level de secours", flush=True)
        is_death = min(pe, pt) <= self.death_level
        return is_death, not is_death

    def _survival_reward(self, e: float, t: float) -> float:
        """Miroir EXACT de reward_manager.gd::_reward_survival_pure (base +1 ; malus pain-shaping
        annulé si SYLVAN_PAIN_SHAPING_W<=0 ; thirst absent → 100 → malus 0, rétro-compatible)."""
        reward = 1.0
        if self.pain_shaping_w > 0.0:
            ec = min(max(e / 100.0, 0.0), 1.0)
            tc = min(max(t / 100.0, 0.0), 1.0)
            reward -= self.pain_shaping_w * ((1.0 - ec) ** 2 + (1.0 - tc) ** 2)
        return reward

    @staticmethod
    def _finalize(tr: dict, done: bool, truncated: bool) -> dict:
        tr["done"] = bool(done)
        tr["truncated"] = bool(truncated)
        return tr

    def on_tick(
        self,
        energy: float,
        thirst: float,
        proprio: list,
        retina: list,
        sample_fn: Callable[[], tuple[list, list]],
        episode_step=None,
        prev_term: str = "none",
    ) -> tuple[list[dict], list]:
        """Traite un tick. Renvoie (transitions_fermées, command_act_à_actionner).

        A#2 : toute transition FERMÉE est ajoutée à `closed` AVANT l'appel de sample_fn, et sample_fn est
        gardé par un try/except → une rétine malformée (build_tokens qui lève) NE perd JAMAIS une
        transition déjà fermée (l'état reste cohérent, la commande retombe sur la dernière valide)."""
        closed: list[dict] = []
        e, t = float(energy), float(thirst)

        # -- 1) Frontière d'épisode EXPLICITE = chute de episode_step (respawn) -----------------------
        if self._detect_boundary(episode_step, e, t) and self._open is not None:
            done, truncated = self._classify(prev_term)  # depuis la raison de Godot, PAS des drives
            closed.append(self._finalize(self._open, done=done, truncated=truncated))
            self._open = None
            self._episode_tick = 0
            self._force_replan = True  # ce tick (le respawn) = 1er tick du nouvel épisode = replan

        # -- 2) Replan ? (début d'épisode forcé, ou cadence K) ---------------------------------------
        is_replan = self._force_replan or (self._episode_tick % self.replan_every == 0)
        if is_replan:
            # Fermer la transition milieu-d'épisode AVANT d'échantillonner : `closed` est déjà peuplé,
            # donc même si sample_fn lève, la transition fermée n'est pas perdue (A#2).
            if self._open is not None:  # replan MILIEU d'épisode : ferme la précédente (non-terminal)
                closed.append(self._finalize(self._open, done=False, truncated=False))
                self._open = None
            try:
                command_raw, command_act = sample_fn()
            except Exception as exc:  # noqa: BLE001 — token-building peut lever sur rétine malformée
                # NE PAS perdre `closed` : on renonce à ouvrir une transition ce tick, on force le replan
                # au prochain tick sain, et on retombe sur la dernière commande valide.
                print(f"[serve-mode1-collect] WARN: sample_fn a échoué ({exc!r}) → transitions fermées "
                      f"préservées, replan différé", flush=True)
                self._force_replan = True
                self._episode_tick += 1
                self._prev_e, self._prev_t = e, t
                return closed, list(self._last_cmd)
            self._open = {
                "proprio": _finite_list(proprio),      # sanitize : NaN → 0.0 (JSON valide, cf A#1)
                "retina": _finite_list(retina),
                "energy": _finite_f(e),
                "thirst": _finite_f(t),
                "command_raw": _finite_list(command_raw),
                "command_act": _finite_list(command_act),
                "reward": 0.0,
                "steps": 0,
            }
            self._last_cmd = self._open["command_act"]
            self._force_replan = False

        # -- 3) Accumule la récompense de survie de CE tick dans la transition ouverte ----------------
        self._open["reward"] += self._survival_reward(e, t)
        self._open["steps"] += 1

        self._episode_tick += 1
        self._prev_e, self._prev_t = e, t
        return closed, self._open["command_act"]


class _Mode1CollectService:
    """Serveur Mode-1 stochastique : politique BC (échantillonnée) → (vx,ω) → résidu gelé → action[18],
    avec logging des macro-transitions espace-commande dans un buffer JSONL."""

    def __init__(
        self,
        residual_ckpt: Path,
        policy_ckpt: Path,
        out_dir: Path,
        seed: int,
        replan_every: int,
        pain_shaping_w: float,
    ) -> None:
        torch.set_num_threads(int(os.environ.get("SYLVAN_PLANNER_THREADS", "4")))
        config = SylvanConfig()
        self.action_dim = config.env.action_dim  # 18

        # -- Résidu gelé (hexapod_v2) — IDENTIQUE à serve_mode1 --------------------------------------
        self.residual = GaussianActorCritic(
            obs_dim=config.env.policy_input_dim,
            hidden_dim=config.controller.hidden_dim,
            action_dim=config.env.action_dim,
        )
        load_checkpoint(residual_ckpt, self.residual)
        self.residual.eval()
        for p in self.residual.parameters():
            p.requires_grad_(False)

        # -- Politique Mode-1 (DriveSymmetricPolicy, warm-start = poids BC) --------------------------
        _ck = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
        self._pol = DriveSymmetricPolicy(proprio_dim=_ck["meta"]["proprio_dim"])
        self._pol.load_state_dict(_ck["model"])
        self._pol.eval()
        for p in self._pol.parameters():
            p.requires_grad_(False)

        # -- Générateur stochastique (seedé ; serveur live, pas un workflow reprenable) --------------
        self._gen = torch.Generator()
        self._gen.manual_seed(int(seed))

        self.replan_every = max(1, replan_every)
        self._state = Mode1CollectState(
            replan_every=self.replan_every, pain_shaping_w=pain_shaping_w
        )
        self._cmd: tuple[float, float] = _DEFAULT_CMD
        self._lock = threading.Lock()

        # -- Buffer de sortie -----------------------------------------------------------------------
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_path = out_dir / f"part-{int(seed)}.jsonl"
        self._fh = open(self._out_path, "w", encoding="utf-8")
        self._n_written = 0

        print(
            f"[serve-mode1-collect] policy={policy_ckpt.name} "
            f"(proprio_dim={_ck['meta']['proprio_dim']}) | residual={residual_ckpt.name} | "
            f"replan_every={self.replan_every} | pain_w={pain_shaping_w} | "
            f"log_std_mean={float(self._pol.log_std.clamp(-3.0, 0.7).exp().mean()):.3f} | "
            f"out={self._out_path}",
            flush=True,
        )

    # -- Échantillonnage de la commande à partir de tokens DÉJÀ construits (validés en amont) --------
    def _sample_from_tokens(self, p_t: torch.Tensor, tok_t: torch.Tensor) -> tuple[list, list]:
        z, _logp = self._pol.sample(p_t.unsqueeze(0), tok_t.unsqueeze(0), generator=self._gen)
        command_act = map_action(z)[0]                     # (vx,ω) bornées par l'actionneur
        return [float(v) for v in z[0].tolist()], [float(v) for v in command_act.tolist()]

    @torch.no_grad()
    def predict_full(self, payload: dict) -> dict:
        if not isinstance(payload.get("proprio"), list):
            raise TypeError("request must contain proprio (list)")
        proprio = payload["proprio"]
        retina = payload.get("retina")
        energy = float(payload.get("energy", 0.0))
        thirst = float(payload.get("thirst", 100.0))  # thirst absent → 100 (rétro-compatible)
        episode_step = payload.get("episode_step")     # None si absent (rétro-compat → fallback)
        prev_term = payload.get("prev_term", "none")

        # A#2 : VALIDER/CONSTRUIRE les tokens AVANT toute mutation d'état. build_tokens lève sur rétine
        # malformée → l'exception remonte ICI (avant on_tick), le handler renvoie un fallback sûr et la
        # transition ouverte reste INTACTE. On réutilise ces tokens pour l'échantillonnage (pas de double
        # build) ; le try/except dans on_tick est une défense en profondeur.
        p_t, tok_t, _meta = build_tokens(payload)

        with self._lock:
            closed, cmd_act = self._state.on_tick(
                energy, thirst, proprio, retina,
                lambda: self._sample_from_tokens(p_t, tok_t),
                episode_step=episode_step, prev_term=prev_term,
            )
            for tr in closed:
                self._write(tr)
            vx, om = float(cmd_act[0]), float(cmd_act[1])
            self._cmd = (vx, om)
            # Obs-résidu PARTAGÉE (helper unique, byte-identique à serve_mode1)
            action = residual_action(self.residual, proprio, vx, om)

        return {"action": action, "command": [vx, om]}

    def _write(self, tr: dict) -> None:
        # Ordre de champs = contrat de schéma (Task 8). Les floats sont déjà sanitizés à l'ouverture de la
        # transition ; on re-garantit ici qu'aucun NaN/inf n'atteint json.dumps (garde-fou A#1).
        line = {
            "proprio": _finite_list(tr["proprio"]),
            "retina": _finite_list(tr["retina"]),
            "energy": _finite_f(tr["energy"]),
            "thirst": _finite_f(tr["thirst"]),
            "command_raw": _finite_list(tr["command_raw"]),
            "command_act": _finite_list(tr["command_act"]),
            "reward": _finite_f(tr["reward"]),
            "steps": int(tr["steps"]),
            "done": bool(tr["done"]),
            "truncated": bool(tr["truncated"]),
        }
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()  # flush régulier → un run tué garde ses données
        self._n_written += 1

    def reset(self) -> None:
        # Godot n'envoie PAS de reset dans ce chemin (frontières détectées via episode_step). On reste
        # neutre pour ne pas perturber la machine à états si un reset arrivait.
        with self._lock:
            self._cmd = _DEFAULT_CMD

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()
            print(
                f"[serve-mode1-collect] {self._n_written} transitions écrites → {self._out_path}",
                flush=True,
            )


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            try:
                payload = json.loads(raw.decode("utf-8"))
                if payload.get("reset"):
                    self.server.service.reset()
                    resp = {
                        "action": [0.0] * self.server.service.action_dim,
                        "command": list(self.server.service._cmd),
                    }
                else:
                    resp = self.server.service.predict_full(payload)
            except Exception as exc:  # noqa: BLE001
                print(f"[serve-mode1-collect] request error: {exc!r} — safe fallback", flush=True)
                resp = {
                    "action": [0.0] * self.server.service.action_dim,
                    "command": list(_DEFAULT_CMD),
                    "error": str(exc),
                }
            self.wfile.write(json.dumps(resp).encode("utf-8") + b"\n")
            self.wfile.flush()


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr, service):
        super().__init__(addr, _Handler)
        self.service = service


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sylvan Mode-1 RL collection server (stochastic policy + frozen residual)"
    )
    ap.add_argument("--residual", default="data/checkpoints/hexapod_v2/policy_best.pt",
                    help="Checkpoint du résidu gelé (hexapod_v2)")
    ap.add_argument("--bc-policy", dest="policy", default=None,
                    help="(alias de --policy) checkpoint de la politique drive-symétrique")
    ap.add_argument("--policy", dest="policy", default="data/checkpoints/mode1_bc/policy.pt",
                    help="Checkpoint de la politique (warm-start = poids BC)")
    ap.add_argument("--out", required=True, help="Répertoire du replay buffer (créé si absent)")
    ap.add_argument("--seed", type=int, default=1, help="Graine du générateur stochastique + nom de fichier")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "6053")))
    ap.add_argument("--replan-every", type=int, default=10,
                    help="Nombre de ticks entre deux échantillonnages de commande")
    args = ap.parse_args()

    pain_w_env = os.environ.get("SYLVAN_PAIN_SHAPING_W", "")
    pain_shaping_w = float(pain_w_env) if pain_w_env != "" else 0.0

    service = _Mode1CollectService(
        residual_ckpt=Path(args.residual),
        policy_ckpt=Path(args.policy),
        out_dir=Path(args.out),
        seed=args.seed,
        replan_every=args.replan_every,
        pain_shaping_w=pain_shaping_w,
    )
    server = _Server((args.host, args.port), service)
    print(f"[serve-mode1-collect] serving on {args.host}:{args.port} — Ctrl-C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
