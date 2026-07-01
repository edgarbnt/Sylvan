# python/sylvan/control/mode1/test_mode1_collect.py
# Tests Task-7 : (A) policy.sample (tirage gaussien sur z BRUT) ; (C) machine à états de collecte
# (fenêtre de récompense + frontières done/truncated), sans Godot.
# Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python -m pytest \
#            python/sylvan/control/mode1/test_mode1_collect.py -q
import math

import torch
from torch.distributions import Normal

from sylvan.control.mode1.policy import (
    DriveSymmetricPolicy,
    map_action,
    TOK,
    LOG_STD_FLOOR,
    LOG_STD_CEIL,
)
from scripts.serve_mode1_collect import Mode1CollectState


# --------------------------------------------------------------------------- #
# (A) policy.sample
# --------------------------------------------------------------------------- #
def test_sample_shapes_and_logprob_match():
    torch.manual_seed(0)
    pol = DriveSymmetricPolicy()
    proprio = torch.randn(4, 132)
    tokens = torch.randn(4, 2, TOK)

    gen = torch.Generator().manual_seed(123)
    z, logp = pol.sample(proprio, tokens, generator=gen)
    assert z.shape == (4, 2), z.shape
    assert logp.shape == (4,), logp.shape

    # logprob = Normal(mean, std).log_prob(z).sum(-1), avec mean = forward() BRUT (pré-map_action)
    mean = pol.forward(proprio, tokens)
    std = pol.log_std.clamp(LOG_STD_FLOOR, LOG_STD_CEIL).exp()
    logp_ref = Normal(mean, std).log_prob(z).sum(-1)
    assert torch.allclose(logp, logp_ref, atol=1e-6)


def test_sample_is_reproducible_with_generator():
    pol = DriveSymmetricPolicy()
    proprio = torch.randn(2, 132)
    tokens = torch.randn(2, 2, TOK)
    z1, _ = pol.sample(proprio, tokens, generator=torch.Generator().manual_seed(7))
    z2, _ = pol.sample(proprio, tokens, generator=torch.Generator().manual_seed(7))
    assert torch.equal(z1, z2)


def test_sample_z_is_unclamped_but_map_action_bounds():
    # z n'est PAS clampé (bruit gaussien libre) ; map_action(z) borne la commande actionnée.
    pol = DriveSymmetricPolicy()
    with torch.no_grad():
        pol.log_std.fill_(math.log(2.0))  # gros bruit → z sort de [-1,1]
    proprio = torch.zeros(2000, 132)
    tokens = torch.zeros(2000, 2, TOK)
    z, _ = pol.sample(proprio, tokens, generator=torch.Generator().manual_seed(1))
    assert (z.abs() > 1.0).any(), "z devrait dépasser [-1,1] (non clampé)"
    cmd = map_action(z)
    assert (cmd[:, 0] >= 0.55).all() and (cmd[:, 0] <= 0.75).all()
    assert (cmd[:, 1] >= -0.6).all() and (cmd[:, 1] <= 0.6).all()


def test_log_std_floor_respected():
    pol = DriveSymmetricPolicy()
    with torch.no_grad():
        pol.log_std.fill_(math.log(1e-6))  # sous le plancher
    std = pol.log_std.clamp(LOG_STD_FLOOR, LOG_STD_CEIL).exp()
    assert torch.allclose(std, torch.full_like(std, 0.05), atol=1e-6)


# --------------------------------------------------------------------------- #
# (C) Mode1CollectState — fenêtre de récompense + frontières EXPLICITES (episode_step/prev_term)
# --------------------------------------------------------------------------- #
import json


def _counting_sample_fn():
    """Renvoie un sample_fn déterministe qui incrémente z à chaque replan (pour tracer l'ouverture)."""
    box = {"i": 0}

    def fn():
        i = box["i"]
        box["i"] += 1
        return [float(i), float(-i)], [0.65, 0.0]

    return fn, box


def _feed(state, sample_fn, ticks):
    """Joue une liste de (energy, thirst, episode_step, prev_term) ; renvoie les transitions fermées."""
    closed_all = []
    for e, t, es, pt in ticks:
        closed, _cmd = state.on_tick(
            e, t, proprio=[0.0] * 132, retina=[0.0] * 144,
            sample_fn=sample_fn, episode_step=es, prev_term=pt,
        )
        closed_all.extend(closed)
    return closed_all


def test_full_window_reward_equals_K():
    # w=0 → +1/tick ; une fenêtre complète de K ticks ferme avec reward==K et steps==K.
    K = 3
    state = Mode1CollectState(replan_every=K, pain_shaping_w=0.0)
    fn, _ = _counting_sample_fn()
    # 7 ticks sains d'UN MÊME épisode (episode_step croissant, pas de respawn) → 2 fenêtres pleines fermées
    ticks = [(90 - i, 90 - i, i, "none") for i in range(7)]
    closed = _feed(state, fn, ticks)
    assert len(closed) == 2, [c["steps"] for c in closed]
    for c in closed:
        assert c["steps"] == K
        assert abs(c["reward"] - float(K)) < 1e-9
        assert c["done"] is False and c["truncated"] is False


def test_death_boundary_marks_done():
    # episode_step retombe à 0 (respawn) + prev_term="death" → transition terminale (done).
    # La classification vient de prev_term, PAS des drives : ici thirst reste plein tout du long.
    K = 10
    state = Mode1CollectState(replan_every=K, pain_shaping_w=0.0)
    fn, box = _counting_sample_fn()
    ticks = [(20.0, 100.0, 0, "none"), (10.0, 100.0, 1, "none"), (0.05, 100.0, 2, "none")]
    ticks += [(100.0, 100.0, 0, "death")]  # respawn : episode_step 2→0, raison = death
    closed = _feed(state, fn, ticks)
    assert len(closed) == 1, closed
    tr = closed[0]
    assert tr["done"] is True and tr["truncated"] is False
    assert tr["steps"] == 3          # 3 ticks avant le respawn
    assert abs(tr["reward"] - 3.0) < 1e-9
    assert tr["command_raw"] == [0.0, -0.0]  # ouverte au 1er sample (i=0)
    assert box["i"] == 2             # 1 open pour l'ép mort + 1 open au respawn


def test_death_classification_ignores_healthy_drives():
    # prev_term="death" alors que les DEUX drives sont sains au dernier tick (mort par chute/health) :
    # l'ancienne heuristique min(drive)<=1 aurait classé truncated → prev_term corrige.
    K = 10
    state = Mode1CollectState(replan_every=K, pain_shaping_w=0.0)
    fn, _ = _counting_sample_fn()
    ticks = [(80.0, 80.0, 0, "none"), (75.0, 75.0, 1, "none")]
    ticks += [(100.0, 100.0, 0, "death")]
    closed = _feed(state, fn, ticks)
    assert len(closed) == 1
    assert closed[0]["done"] is True and closed[0]["truncated"] is False


def test_truncation_boundary_marks_truncated():
    # episode_step retombe à 0 + prev_term="truncated" → non-terminal (truncated, pas done).
    K = 10
    state = Mode1CollectState(replan_every=K, pain_shaping_w=0.0)
    fn, _ = _counting_sample_fn()
    ticks = [(60.0, 55.0, 0, "none"), (59.0, 54.0, 1, "none"), (58.0, 53.0, 2, "none")]
    ticks += [(100.0, 100.0, 0, "truncated")]
    closed = _feed(state, fn, ticks)
    assert len(closed) == 1
    tr = closed[0]
    assert tr["done"] is False and tr["truncated"] is True
    assert tr["steps"] == 3


def test_eat_and_drink_to_full_midlife_is_NOT_a_boundary():
    # RÉGRESSION du DÉFAUT-RACINE : manger ET boire à plein EN PLEINE VIE (les 2 drives ≥ 99.5) ne
    # doit PLUS fragmenter l'épisode. episode_step continue de croître → AUCUNE frontière.
    K = 10
    state = Mode1CollectState(replan_every=K, pain_shaping_w=0.0)
    fn, _ = _counting_sample_fn()
    ticks = [
        (50.0, 50.0, 0, "none"),
        (80.0, 80.0, 1, "none"),
        (100.0, 100.0, 2, "none"),   # DEUX drives pleins → l'ancienne règle aurait fabriqué une frontière
        (100.0, 100.0, 3, "none"),
        (99.0, 99.0, 4, "none"),
    ]
    closed = _feed(state, fn, ticks)
    assert closed == [], [(c["steps"], c["done"], c["truncated"]) for c in closed]
    # une seule transition ouverte, jamais classée, 5 ticks accumulés
    assert state._open is not None
    assert state._open["steps"] == 5


def test_back_to_back_death_then_truncation():
    # Épisodes dos-à-dos : ép A meurt, ép B tronqué (raisons EXPLICITES, drives non-utilisés).
    K = 2
    state = Mode1CollectState(replan_every=K, pain_shaping_w=0.0)
    fn, _ = _counting_sample_fn()
    ticks = [
        (80.0, 80.0, 0, "none"), (79.0, 79.0, 1, "none"),   # ép A : fenêtre pleine (K=2)
        (2.0, 79.0, 2, "none"), (0.05, 79.0, 3, "none"),    # ép A : continue
        (100.0, 100.0, 0, "death"),                         # respawn A → mort
        (50.0, 50.0, 1, "none"),                            # ép B : continue
        (100.0, 100.0, 0, "truncated"),                     # respawn B → troncation
    ]
    closed = _feed(state, fn, ticks)
    dones = [c for c in closed if c["done"]]
    truncs = [c for c in closed if c["truncated"]]
    assert len(dones) == 1, [(c["steps"], c["done"], c["truncated"]) for c in closed]
    assert len(truncs) == 1
    d = dones[0]
    assert abs(d["reward"] - float(d["steps"])) < 1e-9


def test_nan_values_sanitized_to_finite_json():
    # A#1 : NaN/inf dans proprio/retina/command → sanitizés à 0.0 ; json.dumps ne doit PAS émettre
    # de token NaN/Infinity (JSON invalide qui empoisonnerait le buffer Task-8).
    import math

    state = Mode1CollectState(replan_every=10, pain_shaping_w=0.0)

    def nan_fn():
        return [float("nan"), 0.5], [float("inf"), -float("inf")]

    proprio = [float("nan")] + [0.0] * 131
    retina = [float("inf")] + [0.0] * 143
    # ouvre une transition (episode_step=5) avec des valeurs non-finies…
    state.on_tick(50.0, 50.0, proprio, retina, nan_fn, episode_step=5, prev_term="none")
    # …puis la ferme par une frontière (episode_step retombe à 0)
    closed, _ = state.on_tick(50.0, 50.0, [0.0] * 132, [0.0] * 144, nan_fn,
                              episode_step=0, prev_term="truncated")
    assert len(closed) == 1
    tr = closed[0]
    for key in ("proprio", "retina", "command_raw", "command_act"):
        assert all(math.isfinite(v) for v in tr[key]), key
    # allow_nan=False lève si un non-fini subsiste → assure un JSON valide
    s = json.dumps(tr, allow_nan=False)
    assert "NaN" not in s and "Infinity" not in s


def test_bad_tick_after_boundary_preserves_closed_transition():
    # A#2 : une frontière ferme une transition PUIS sample_fn lève (rétine malformée). La transition
    # déjà fermée NE doit PAS être perdue ; la commande retombe sur la dernière valide.
    K = 10
    state = Mode1CollectState(replan_every=K, pain_shaping_w=0.0)
    good_fn, _ = _counting_sample_fn()
    _feed(state, good_fn, [(50.0, 50.0, 0, "none"), (49.0, 49.0, 1, "none"), (48.0, 48.0, 2, "none")])

    def bad_fn():
        raise ValueError("malformed retina")

    closed, cmd = state.on_tick(100.0, 100.0, [0.0] * 132, [0.0] * 144, bad_fn,
                                episode_step=0, prev_term="death")
    assert len(closed) == 1, "la transition fermée à la frontière doit survivre à l'échec du sampling"
    assert closed[0]["done"] is True and closed[0]["truncated"] is False
    assert closed[0]["steps"] == 3
    assert cmd == list(state._last_cmd)   # fallback sur la dernière commande valide
    assert state._force_replan is True    # replan re-tenté au prochain tick sain


def test_missing_episode_step_falls_back_to_drive_level(capsys):
    # RÉTRO-COMPAT : sans episode_step, on retombe sur l'ancienne heuristique drive-level (loggée).
    K = 10
    state = Mode1CollectState(replan_every=K, pain_shaping_w=0.0)
    fn, _ = _counting_sample_fn()
    ticks = [(20.0, 100.0, None, "none"), (0.05, 100.0, None, "none")]
    ticks += [(100.0, 100.0, None, "none")]  # front montant "les 2 ≥ 99.5" → frontière fallback
    closed = _feed(state, fn, ticks)
    assert len(closed) == 1
    # prev_term="none" à la frontière → classification drive-level de secours : energy critique = death
    assert closed[0]["done"] is True
    assert "fallback" in capsys.readouterr().out.lower()


def test_pain_shaping_reward_matches_formula():
    # w>0 → reward = 1 - w*((1-e)^2+(1-t)^2), miroir de _reward_survival_pure.
    w = 0.5
    state = Mode1CollectState(replan_every=100, pain_shaping_w=w)
    fn, _ = _counting_sample_fn()
    e, t = 40.0, 60.0
    state.on_tick(e, t, proprio=[0.0] * 132, retina=[0.0] * 144, sample_fn=fn,
                  episode_step=0, prev_term="none")
    ec, tc = e / 100.0, t / 100.0
    expected = 1.0 - w * ((1.0 - ec) ** 2 + (1.0 - tc) ** 2)
    assert abs(state._open["reward"] - expected) < 1e-9


def test_thirst_absent_defaults_full_no_penalty():
    # Serveur : thirst absent → 100 → malus 0 même avec w>0 (rétro-compat mono-drive).
    w = 0.5
    state = Mode1CollectState(replan_every=100, pain_shaping_w=w)
    fn, _ = _counting_sample_fn()
    state.on_tick(100.0, 100.0, proprio=[0.0] * 132, retina=[0.0] * 144, sample_fn=fn,
                  episode_step=0, prev_term="none")
    assert abs(state._open["reward"] - 1.0) < 1e-9
