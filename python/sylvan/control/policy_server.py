"""TCP inference server bridging Godot observations to the encoder + controller.

Note sur l'inférence CPU vs GPU :
  Le policy server tourne en CPU par défaut. Raisons :
    1. Godot (même en --headless) peut initialiser Vulkan/HSA sur le même GPU,
       ce qui entre en conflit avec ROCm → HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION.
    2. Le modèle (3 couches Linear sur un vecteur de 51 floats) prend < 0.1ms sur CPU.
       Le GPU n'apporte aucun gain mesurable ici.
    3. Le GPU est réservé entièrement pour l'entraînement (night cycle), ce qui compte.

  Pour forcer GPU : SYLVAN_POLICY_DEVICE=cuda (non recommandé avec Godot concurrent)
"""

from __future__ import annotations

import json
import os
import socketserver
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from ..config import SylvanConfig
from ..models.world_model import WorldModelV0
from ..training.checkpointing import load_checkpoint
from .controller import LocomotionController


class _PolicyInferenceService:
    def __init__(
        self,
        config: SylvanConfig,
        *,
        world_model_checkpoint: Path,
        controller_checkpoint: Path,
    ) -> None:
        self.config = config
        # CPU par défaut : voir note en tête de fichier.
        # Godot concurrrent + ROCm = HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION.
        requested = os.environ.get("SYLVAN_POLICY_DEVICE", "cpu").strip().lower()
        self.device = torch.device(requested if requested in {"cpu", "cuda"} else "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
        print(f"[Python] Policy server device: {self.device}")
        self._lock = threading.Lock()
        env = config.env
        train = config.train
        self.world_model = WorldModelV0(
            proprio_dim=env.proprio_dim,
            action_dim=env.action_dim,
            metrics_dim=env.metrics_dim,
            hidden_dim=train.hidden_dim,
            latent_dim=train.latent_dim,
        ).to(self.device)
        self.controller = LocomotionController(
            input_dim=train.latent_dim,
            hidden_dim=config.controller.hidden_dim,
            action_dim=env.action_dim,
        ).to(self.device)
        load_checkpoint(world_model_checkpoint, self.world_model)
        load_checkpoint(controller_checkpoint, self.controller)
        self.world_model.eval()
        self.controller.eval()
        for p in self.world_model.parameters():
            p.requires_grad_(False)
        for p in self.controller.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, payload: dict[str, object]) -> list[float]:
        proprio = payload.get("proprio")
        if not isinstance(proprio, list):
            raise TypeError("Policy request must contain a proprio list")
        t = torch.tensor(proprio, dtype=torch.float32, device=self.device).unsqueeze(0)
        with self._lock:
            latent = self.world_model.encoder(t)
            action = self.controller.act(latent)[0]
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return [float(v) for v in action.detach().cpu().tolist()]


class _PolicyTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, request_handler_class, *, inference_service):
        super().__init__(server_address, request_handler_class)
        self.inference_service = inference_service


class _PolicyRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw_line = self.rfile.readline()
            if not raw_line:
                return
            # Robustness: never let a malformed payload / inference error kill the handler thread
            # (root cause of the orphan-server pain). Catch, log, send a safe zero action, keep going.
            try:
                payload = json.loads(raw_line.decode("utf-8"))
                action = self.server.inference_service.predict(payload)
            except Exception as exc:  # noqa: BLE001 — deliberately broad
                # Zero action of the CORRECT width (was hardcoded 12 → silently fed 12-dim zeros to a
                # 13-DOF salamander, masking the real error and freezing the agent). Match action_dim.
                n = getattr(self.server.inference_service, "action_dim", 12)
                print(f"[policy-server] request error: {exc!r} — sending zero action (dim {n})", flush=True)
                action = [0.0] * n
            response = json.dumps({"action": action}).encode("utf-8") + b"\n"
            self.wfile.write(response)
            self.wfile.flush()


@contextmanager
def serve_policy_controller(
    config: SylvanConfig,
    *,
    world_model_checkpoint: Path,
    controller_checkpoint: Path,
) -> Iterator[dict[str, object]]:
    inference_service = _PolicyInferenceService(
        config,
        world_model_checkpoint=world_model_checkpoint,
        controller_checkpoint=controller_checkpoint,
    )
    server = _PolicyTCPServer(
        (config.godot.policy_host, config.godot.policy_port),
        _PolicyRequestHandler,
        inference_service=inference_service,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield {"host": host, "port": port}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
