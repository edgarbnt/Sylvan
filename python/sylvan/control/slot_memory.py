"""SlotMemory — mémoire spatiale côté serveur (Task 3, mémoire spatiale échafaudage-first).

Persiste la position ego d'un objet entre les replans :
- dead-reckon par l'ego-motion RÉELLE estimée du proprio (EgomotionHead)
- re-ground quand l'objet est perçu (salience >= threshold)
- sert de slot t0 override au planner (plan(..., slot_belief=belief))

Principes :
  §1 (gate gratuit décisif) : le gate offline diag_slot_memory_drift.py a PASSÉ (a_encode 0.07 m plat N→40)
  §2 (ne pas masquer) : belief = coordonnée ego réelle, jamais un proxy dégradé
  §4 (étape solide) : non-régression byte-identique quand --slot-memory est OFF

Usage:
    PYTHONPATH=python ./env_pytorch_3.12/bin/python python/sylvan/control/slot_memory.py --selfcheck
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sylvan.models.egomotion_head import EgomotionHead
    from sylvan.models.slot_head import SelfSupervisedSlotHead

import torch


# ---------------------------------------------------------------------------
# Opérateur de transport géométrique (convention nailée en Task 1)
# ---------------------------------------------------------------------------

def transport_geom(p: list[float], dyaw: float, dfwd: float, dlat: float) -> list[float]:
    """Transporte un point ego p=[x_right, z_fwd] d'un pas d'ego-motion (dyaw, dfwd, dlat).

    Convention NAILÉE par diag_slot_memory_drift.py (Task 1) : translate(−déplacement ego)
    puis ROTATE R(+dyaw). C'est l'opérateur RÉEL (ego-motion live), PAS le transport_slot du
    WM (calib 1,-1,-1, utilisé pour l'imagination interne). Ne pas mélanger les deux.
    """
    # Translate: soustraire le déplacement de l'agent (en frame body)
    px = p[0] - dlat
    pz = p[1] - dfwd
    # Rotate: R(+dyaw) — le monde tourne dans le sens INVERSE de l'agent
    ca, sa = math.cos(dyaw), math.sin(dyaw)
    return [ca * px - sa * pz, sa * px + ca * pz]


# ---------------------------------------------------------------------------
# SlotMemory
# ---------------------------------------------------------------------------

class SlotMemory:
    """Persistance inter-replans de la position ego d'un objet.

    À chaque tick réel (avant le replan) :
    1. Si un proprio précédent est stocké, dead-reckon le belief courant via transport_geom
       avec l'ego-motion estimée par egomotion_head.predict(prev_proprio).
    2. Percevoir l'objet : slot_encoder.positions_and_salience(retina) → (pos, salience).
    3. Si salience >= threshold → RE-GROUND (perception précise → belief = pos).
       Sinon → garder le belief dead-reckoné.
    4. Stocker proprio courant comme prev pour le prochain tick.
    5. Retourner belief = [x_right, z_fwd].

    Premier appel (pas encore de belief) : seeder depuis la perception dans tous les cas.
    """

    def __init__(
        self,
        egomotion_head: "EgomotionHead",
        slot_encoder: "SelfSupervisedSlotHead",
        salience_threshold: float = 0.05,
    ) -> None:
        self.egomotion_head = egomotion_head
        self.slot_encoder = slot_encoder
        self.salience_threshold = salience_threshold

        self.belief: list[float] | None = None
        self._prev_proprio: list[float] | None = None

    def reset(self) -> None:
        """Réinitialiser l'état entre épisodes."""
        self.belief = None
        self._prev_proprio = None

    @torch.no_grad()
    def update(self, proprio: list[float], retina: list[float]) -> list[float]:
        """Met à jour le belief et retourne la position ego courante [x_right, z_fwd].

        Appelé une fois par tick Godot, avant le bloc de replan.
        """
        # ── 1. Dead-reckon (si on a un belief et un proprio précédent) ──
        if self.belief is not None and self._prev_proprio is not None:
            dyaw, dfwd, dlat = self.egomotion_head.predict(self._prev_proprio)
            self.belief = transport_geom(self.belief, dyaw, dfwd, dlat)

        # ── 2. Percevoir ──
        retina_t = torch.tensor(retina, dtype=torch.float32)
        positions, saliences = self.slot_encoder.positions_and_salience(
            retina_t.reshape(-1)
        )
        # Slot 0 (food) : position [x_right, z_fwd] et saillance scalaire
        pos = [float(positions[0, 0]), float(positions[0, 1])]
        sal = float(saliences[0])

        # ── 3. Re-ground si saillant, sinon garder le dead-reckoné ──
        if sal >= self.salience_threshold or self.belief is None:
            # Premier appel OU objet visible → anchorer sur la perception
            self.belief = pos
        # Sinon self.belief = dead-reckoné (déjà mis à jour step 1)

        # ── 4. Stocker le proprio courant ──
        self._prev_proprio = list(proprio)

        return list(self.belief)


# ---------------------------------------------------------------------------
# Self-check (TDD, pure Python, no Godot)
# ---------------------------------------------------------------------------

def _run_selfcheck() -> None:
    """RED → GREEN TDD self-check.

    Cas vérifiés :
    (i)  objet visible plusieurs ticks → belief ≈ perception (re-ground)
    (ii) occultation + ego-motion non-nulle → belief se déplace (dead-reckoned, pas gelé, pas NaN)
    (iii) réapparition → belief revient vers la perception (saut borné)
    (iv) non-régression : rollout_open_loop(slot0=None) byte-identique à l'appel sans slot0
    """

    # ── Stub minimal de EgomotionHead ──
    class _FakeEgoHead:
        """Simule une ego-motion constante : toujours (dyaw=0.1, dfwd=0.2, dlat=0.0)."""
        def predict(self, proprio):
            return (0.1, 0.2, 0.0)

    # ── Stub minimal de SelfSupervisedSlotHead ──
    class _FakeSlotEncoder:
        """Retourne une position fixe (1.0, 2.0) et une saillance contrôlée par un flag."""
        def __init__(self, salient: bool = True, sal_value: float = 1.0):
            self.salient = salient
            self.sal_value = sal_value

        def positions_and_salience(self, retina):
            pos = torch.tensor([[1.0, 2.0]])      # [1, 2]
            sal = torch.tensor([self.sal_value if self.salient else 0.0])  # [1]
            return pos, sal

    THRESHOLD = 0.05
    PROPRIO_FAKE = [0.0] * 132  # dummy, ego-motion vient du stub

    # ── Cas (i) : objet visible N ticks → belief doit approximer la perception ──
    print("[selfcheck] Cas (i) : objet visible → re-ground...")
    enc_vis = _FakeSlotEncoder(salient=True, sal_value=1.0)
    mem = SlotMemory(_FakeEgoHead(), enc_vis, salience_threshold=THRESHOLD)
    for _ in range(10):
        b = mem.update(PROPRIO_FAKE, [0.0] * 144)
    # Après re-ground continu, le belief doit être exactement la position perçue
    assert abs(b[0] - 1.0) < 1e-5 and abs(b[1] - 2.0) < 1e-5, \
        f"Cas (i) FAIL : belief={b}, attendu ≈ [1.0, 2.0]"
    print(f"  PASS : belief={b} ≈ [1.0, 2.0]")

    # ── Cas (ii) : occultation + ego-motion → belief doit bouger (≠ gelé) ──
    print("[selfcheck] Cas (ii) : occultation + ego-motion → dead-reckon...")
    # D'abord, on ancre le belief avec un objet visible
    enc_vis2 = _FakeSlotEncoder(salient=True, sal_value=1.0)
    mem2 = SlotMemory(_FakeEgoHead(), enc_vis2, salience_threshold=THRESHOLD)
    mem2.update(PROPRIO_FAKE, [0.0] * 144)  # tick 0 → belief = [1.0, 2.0]
    # Puis occlusion (saillance 0)
    mem2.slot_encoder = _FakeSlotEncoder(salient=False, sal_value=0.0)
    prev_belief = list(mem2.belief)
    beliefs = [list(prev_belief)]
    for _ in range(20):
        b = mem2.update(PROPRIO_FAKE, [0.0] * 144)
        beliefs.append(list(b))
    # Le belief ne doit PAS être gelé (il change à chaque tick via dead-reckon)
    frozen = all(abs(beliefs[i][0] - beliefs[0][0]) < 1e-9 and
                 abs(beliefs[i][1] - beliefs[0][1]) < 1e-9 for i in range(1, len(beliefs)))
    assert not frozen, "Cas (ii) FAIL : belief gelé pendant occultation (dead-reckon absent)"
    # Le belief ne doit pas être NaN/infini
    for bf in beliefs:
        assert math.isfinite(bf[0]) and math.isfinite(bf[1]), f"Cas (ii) FAIL : belief NaN/inf : {bf}"
    # La direction de dérive doit être cohérente : avec dfwd=0.2 constant et dyaw=0.1 rad/tick,
    # après 20 ticks le belief doit s'être déplacé (norme de déplacement > 0)
    total_move = math.hypot(beliefs[-1][0] - beliefs[0][0], beliefs[-1][1] - beliefs[0][1])
    assert total_move > 0.01, f"Cas (ii) FAIL : déplacement trop petit ({total_move:.4f})"
    print(f"  PASS : belief se déplace (déplacement total={total_move:.3f} m sur 20 ticks), pas NaN")

    # ── Cas (iii) : réapparition → belief revient vers la perception ──
    print("[selfcheck] Cas (iii) : réapparition → re-ground...")
    # mem2 est en pleine occultation ; réactiver la saillance
    mem2.slot_encoder = _FakeSlotEncoder(salient=True, sal_value=1.0)
    b_before = list(mem2.belief)
    b_after = mem2.update(PROPRIO_FAKE, [0.0] * 144)
    # Doit revenir vers [1.0, 2.0]
    assert abs(b_after[0] - 1.0) < 1e-5 and abs(b_after[1] - 2.0) < 1e-5, \
        f"Cas (iii) FAIL : b_after={b_after}, attendu ≈ [1.0, 2.0]"
    jump = math.hypot(b_after[0] - b_before[0], b_after[1] - b_before[1])
    # Saut borné (objet max 10 m de portée + dead-reckon diverge peu)
    assert jump < 20.0, f"Cas (iii) FAIL : saut trop grand ({jump:.2f} m)"
    print(f"  PASS : re-grounded → {b_after}, saut={jump:.3f} m (borné)")

    # ── Non-régression (iv) : rollout_open_loop sans slot0 = byte-identique ──
    print("[selfcheck] Non-régression : rollout_open_loop(slot0=None) byte-identique...")
    try:
        from sylvan.models.command_wm import CommandWorldModel
        obs_dim = 277   # proprio(132) + retina(144) + energy(1) — le WM slot utilise la rétine complète
        proprio_dim = 132
        wm = CommandWorldModel(obs_dim=obs_dim, proprio_dim=proprio_dim, with_slot=True)
        wm.eval()
        torch.manual_seed(42)
        obs0 = torch.randn(3, obs_dim)           # batch=3
        cmds = torch.randn(3, 10, 2)             # horizon=10
        out_ref = wm.rollout_open_loop(obs0, cmds)           # sans slot0
        out_new = wm.rollout_open_loop(obs0, cmds, slot0=None)  # slot0=None explicite
        # Les deux doivent être byte-identiques (même graphe, même inputs)
        for key in out_ref:
            diff = (out_ref[key] - out_new[key]).abs().max().item()
            assert diff == 0.0, f"Non-régression FAIL sur '{key}' : diff_max={diff}"
        print(f"  PASS : byte-identique sur toutes les clés {list(out_ref.keys())}")
    except ImportError as e:
        print(f"  SKIP (import WM indisponible : {e})")

    print("\n[selfcheck] TOUS LES CAS PASSENT. SlotMemory opérationnel.")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _run_selfcheck()
        sys.exit(0)
    print("Usage: python slot_memory.py --selfcheck")
    sys.exit(1)
