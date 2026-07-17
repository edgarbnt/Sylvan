"""PeriodicRemeasure — re-mesure périodique de la perception typée (embryon jour/nuit).

Gate-capacité (docs/design_gate_capacite.md) : bufferise, PAR TICK, le rayon touchant le plus
proche (rgbn normalisé + distance) et l'événement de conséquence vécu (relief énergie/soif,
dégât) entre deux ticks consécutifs, dans une fenêtre GLISSANTE (`window`, >> `every` — un
segment disjoint de `every` pas est trop souvent monochrome en monde épars, cf `PeriodicRemeasure`
docstring) ; toutes les `every` pas, relance la MESURE cluster+lien
(scripts.build_typed_slots.stage_a_cluster/stage_b_bind — RÉUTILISÉS tels quels, zéro
gradient) sur cette fenêtre et met à jour LIVE color_queries/query_thr du slot_encoder,
UNIQUEMENT pour les drives dont un groupe a été identifié ce cycle (fenêtre trop pauvre →
aucune mise à jour, les requêtes précédentes tiennent). N'écrit jamais l'assignation
slot-index→drive (food_idx/water_idx/hazard_idx) : elle reste celle DÉCOUVERTE à la
construction du WM typé — seule la couleur/marge de CE slot est ré-estimée.

OFF (every<=0, ou jamais instancié par l'appelant) : zéro effet — bit-identique.

Usage (selfcheck, gratuit, sur données synthétiques) :
    PYTHONPATH=python ./env_pytorch_3.12/bin/python python/sylvan/control/remeasure.py --selfcheck
"""

from __future__ import annotations

import sys
from collections import deque

import numpy as np
import torch

from scripts.build_typed_slots import RELIEF, stage_a_cluster, stage_b_bind
from scripts.train_danger_saliency import DMG_DROP, LIFE_JUMP
from sylvan.models.slot_head import DEPTH_OFFSET, NRAY, RANGE

OUTCOMES = ("energy", "thirst", "damage")


def _nearest_touch(retina: list[float]) -> tuple[np.ndarray, float] | None:
    """Rayon touchant le plus proche → (rgb normalisé, distance en mètres). None si rien ne touche.

    Même règle que scripts.build_typed_slots.scan_run (mesure identique, live au lieu d'offline).
    Précondition (non re-vérifiée ici, comme le reste de la perception côté serveur) :
    len(retina) == NRAY*4.
    """
    best_k, best_d = -1, 2.0
    for k in range(NRAY):
        d = retina[4 * k]
        if d < 0.999 and d < best_d:
            best_d, best_k = d, k
    if best_k < 0:
        return None
    v = np.array(retina[4 * best_k + 1:4 * best_k + 4], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return v / n, best_d * RANGE + DEPTH_OFFSET


class PeriodicRemeasure:
    """État bufferisé + déclencheur de re-mesure, indépendant du serveur (testable isolément).

    Le buffer est une fenêtre GLISSANTE de taille `window` (>> `every`, par défaut) : la mesure,
    déclenchée tous les `every` pas, porte sur les dernières `window` observations vécues, pas
    sur le seul segment écoulé depuis le dernier déclenchement. Nécessaire en monde épars : un
    segment DISJOINT de `every` pas est souvent MONOCHROME (l'entité reste un moment près d'une
    seule ressource) → aucune diversité à regrouper. La fenêtre glissante donne une chance réelle
    de voir plusieurs types récemment sans changer N (toujours la période DÉCLARÉE du déclencheur).

    `window` DOIT couvrir plusieurs VIES, pas la seule vie en cours : le harnais de collecte
    (`SYLVAN_COLLECTOR_MODE=policy_server`, cf collect_critic_corpus_kin.sh) n'envoie JAMAIS de
    message "reset" au serveur planner entre deux vies (vérifié : `_PlannerService.reset()` n'est
    invoqué qu'à l'ouverture du process) — l'état de re-mesure vit donc à l'échelle du PROCESS
    serveur, pas de la vie. En monde épars 1+1, un relief (repas/gorgée) survient environ tous les
    ~500 pas poolés (mesuré, docs/design_gate_capacite.md) : il faut plusieurs MILLIERS de pas —
    donc plusieurs vies — pour voir assez de reliefs de chaque type et que K=3 se découvre proprement
    (mesuré : window<=1500 quasi jamais de mesure exploitable ; window=4000-6000 stabilise ; au-delà,
    rendements décroissants sur ce corpus). `reset_life()` reste appelée (défensive, cf serve_planner_
    command.py) mais n'a d'effet que si un futur mode envoie un vrai reset par vie."""

    def __init__(self, every: int, window: int | None = None, min_samples: int = 40,
                seed: int = 0) -> None:
        self.every = max(0, every)
        self.window = window if window is not None else 6000
        self.min_samples = min_samples
        self._rng = np.random.default_rng(seed)
        self._tick = 0
        self._pending: tuple[np.ndarray, float] | None = None
        self._prev_drives: tuple[float, float, float] | None = None
        self._buf_rgbn: deque[np.ndarray] = deque(maxlen=self.window)
        self._buf_dist: deque[float] = deque(maxlen=self.window)
        self._buf_y: dict[str, deque[float]] = {o: deque(maxlen=self.window) for o in OUTCOMES}
        self.last_bound: dict[int, str] | None = None   # trace/log (dernière mesure réussie)
        self.n_updates = 0                               # compteur de mises à jour APPLIQUÉES

    def reset_life(self) -> None:
        """Nouvelle vie : la fenêtre glissante ne doit pas mélanger deux vies (comme le swap Godot)."""
        self._tick = 0
        self._pending = None
        self._prev_drives = None
        self._buf_rgbn.clear()
        self._buf_dist.clear()
        for o in OUTCOMES:
            self._buf_y[o].clear()

    def observe(self, retina: list[float], energy: float, thirst: float, health: float) -> None:
        """Un point par tick, LABELLISÉ un tick en retard (comme scan_run : le rayon à t prédit
        la conséquence entre t et t+1 — on ne connaît celle-ci qu'au tick SUIVANT). Les deques
        `maxlen=window` évincent silencieusement le plus ancien point : fenêtre glissante."""
        drives = (energy, thirst, health)
        if self._prev_drives is not None and self._pending is not None:
            e0, t0, h0 = self._prev_drives
            boundary = (energy - e0 > LIFE_JUMP or thirst - t0 > LIFE_JUMP or health - h0 > LIFE_JUMP)
            rgbn, dist = self._pending
            self._buf_rgbn.append(rgbn)
            self._buf_dist.append(dist)
            self._buf_y["energy"].append(float(not boundary and energy - e0 > RELIEF))
            self._buf_y["thirst"].append(float(not boundary and thirst - t0 > RELIEF))
            self._buf_y["damage"].append(float(not boundary and h0 - health > DMG_DROP))
        self._prev_drives = drives
        self._pending = _nearest_touch(retina)
        self._tick += 1

    def due(self) -> bool:
        return self.every > 0 and self._tick >= self.every

    def measure(self) -> dict | None:
        """Cluster+lien sur la fenêtre glissante COURANTE ; le buffer n'est PAS vidé (il continue
        de glisser) — seul le compteur de périodicité repart à 0. None si trop peu de données OU
        mesure dégénérée (fenêtre pauvre : ex. un seul type d'apparence vu récemment) — jamais de
        crash serveur pour une fenêtre creuse."""
        self._tick = 0
        n = len(self._buf_rgbn)
        if n < self.min_samples:
            return None
        rgbn = np.array(self._buf_rgbn)
        dist = np.array(self._buf_dist)
        y = {o: np.array(v) for o, v in self._buf_y.items()}
        try:
            A = stage_a_cluster(rgbn, self._rng)
            B = stage_b_bind(A["C"], {"rgbn": rgbn, "dist": dist, "y": y})
        except (ValueError, FloatingPointError):
            return None
        # BIJECTION (même esprit que le G-bind offline, scripts.build_typed_slots.main) : quand
        # PLUSIEURS groupes argmax-ent vers le MÊME drive — K sur-découpé sur une fenêtre encore
        # peu diverse, OU cas attendu du swap (l'ancienne ET la nouvelle apparence sont toutes
        # deux valablement liées au même drive dans une fenêtre qui chevauche les deux régimes) —
        # on garde le groupe le mieux SOUTENU par le contact vécu récent (n_contact, depuis
        # stage_b_bind). C'est une mesure (plus d'évidence récente gagne), pas une préférence
        # codée sur une couleur ; les autres drives non-disputés restent inchangés.
        by_outcome: dict[str, list[int]] = {}
        for j, o in B["bound"].items():
            by_outcome.setdefault(o, []).append(j)
        clean_bound = {max(js, key=lambda g: B["n_contact"][g]): o for o, js in by_outcome.items()}
        return {"C": A["C"], "thr": A["thr"], "bound": clean_bound}

    def apply(self, wm, food_idx: int, water_idx: int | None, hazard_idx: int | None) -> list[str]:
        """Mesure + applique en live sur wm.slot_encoder (color_queries/query_thr), UN drive à la
        fois (n'écrase que les indices dont un groupe a été identifié). Retourne les drives mis à jour."""
        res = self.measure()
        if res is None:
            return []
        want = {"energy": food_idx, "thirst": water_idx, "damage": hazard_idx}
        updated: list[str] = []
        with torch.no_grad():
            for j, outcome in res["bound"].items():
                idx = want.get(outcome)
                if idx is None:
                    continue
                wm.slot_encoder.color_queries[idx] = torch.tensor(res["C"][j], dtype=torch.float32)
                if wm.slot_encoder.query_thr is not None:
                    wm.slot_encoder.query_thr[idx] = float(res["thr"][j])
                updated.append(outcome)
        if updated:
            self.last_bound = res["bound"]
            self.n_updates += 1
        return updated


# ------------------------------------------------------------------ selfcheck (gratuit, synthétique)

def _run_selfcheck() -> None:
    rng = np.random.default_rng(0)

    def make_retina(rgb: tuple[float, float, float], dist_m: float) -> list[float]:
        # une seule direction touchante (rayon 0), le reste "à l'infini" (depth=1, RGB=0)
        depth = min(max((dist_m - DEPTH_OFFSET) / RANGE, 0.0), 0.998)
        ret = [1.0, 0.0, 0.0, 0.0] * NRAY
        ret[0:4] = [depth, rgb[0], rgb[1], rgb[2]]
        return ret

    RED, BLUE, GREEN = (0.9, 0.2, 0.1), (0.1, 0.2, 0.9), (0.15, 0.85, 0.1)

    rm = PeriodicRemeasure(every=90, min_samples=20, seed=0)
    e, t, h = 70.0, 70.0, 100.0
    assert not rm.due()
    for i in range(90):
        # cycle à travers les 3 apparences, contact permanent (dist < CONTACT_M du module bind)
        color = (RED, BLUE, GREEN)[i % 3]
        jitter = tuple(float(np.clip(c + rng.normal(0, 0.02), 0.0, 1.0)) for c in color)
        rm.observe(make_retina(jitter, 1.0), e, t, h)
        # conséquence vécue : manger sur le rouge, boire sur le bleu, dégât sur le vert —
        # SEULEMENT après avoir observé ce point (le prochain tick porte le label, comme scan_run)
        if i % 3 == 0:
            e += RELIEF + 1.0
        elif i % 3 == 1:
            t += RELIEF + 1.0
        else:
            h -= DMG_DROP + 0.1
    assert rm.due(), "90 ticks bufferisés -> due() doit être vrai"

    class _FakeSlotEncoder:
        def __init__(self) -> None:
            self.color_queries = torch.zeros(3, 3)
            self.query_thr = torch.zeros(3)

    class _FakeWM:
        def __init__(self) -> None:
            self.slot_encoder = _FakeSlotEncoder()

    wm = _FakeWM()
    updated = rm.apply(wm, food_idx=0, water_idx=1, hazard_idx=2)
    print(f"[selfcheck] mis à jour : {sorted(updated)} bound={rm.last_bound}")
    assert sorted(updated) == ["damage", "energy", "thirst"], updated
    q = wm.slot_encoder.color_queries
    assert float(torch.nn.functional.cosine_similarity(q[0:1], torch.tensor([RED]), dim=1)) > 0.98
    assert float(torch.nn.functional.cosine_similarity(q[1:2], torch.tensor([BLUE]), dim=1)) > 0.98
    assert float(torch.nn.functional.cosine_similarity(q[2:3], torch.tensor([GREEN]), dim=1)) > 0.98
    print("[selfcheck] Cas riche : 3 groupes retrouvés, liés au bon drive, écrits au bon index.")

    # fenêtre pauvre (trop peu de données) -> pas de crash, aucune mise à jour, buffer reparti
    rm2 = PeriodicRemeasure(every=5, min_samples=20, seed=0)
    for _ in range(5):
        rm2.observe(make_retina(RED, 1.0), 70.0, 70.0, 100.0)
    assert rm2.due()
    assert rm2.apply(wm, 0, 1, 2) == []
    print("[selfcheck] Fenêtre pauvre (n<min_samples) : aucune mise à jour, zéro crash.")

    # reset_life() doit vider proprement même une fenêtre partiellement remplie
    rm3 = PeriodicRemeasure(every=100, min_samples=20, seed=0)
    for _ in range(10):
        rm3.observe(make_retina(RED, 1.0), 70.0, 70.0, 100.0)
    rm3.reset_life()
    assert not rm3.due() and len(rm3._buf_rgbn) == 0 and rm3._prev_drives is None
    print("[selfcheck] reset_life() vide le buffer et le compteur.")

    print("\n[selfcheck] TOUS LES CAS PASSENT. PeriodicRemeasure opérationnel.")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _run_selfcheck()
    else:
        print("Usage: python remeasure.py --selfcheck")
