# Tests Task-8 : critique drive-symétrique + interface EMPAQUETÉE de DriveSymmetricPolicy.
# Inclut la RÉGRESSION de compat Task-7 (sample inchangé pour seed+poids fixes).
# Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python -m pytest \
#            python/sylvan/control/mode1/test_policy_critic.py -q
import copy

import torch
from torch.distributions import Normal

from sylvan.control.mode1.policy import (
    DriveSymmetricPolicy,
    LOG_STD_FLOOR,
    LOG_STD_CEIL,
    TOK,
)


def _policy(seed=0):
    torch.manual_seed(seed)
    return DriveSymmetricPolicy()


# --------------------------------------------------------------------------- #
# Régression Task-7 : `sample` et `forward` INCHANGÉS par l'ajout du critique.
# --------------------------------------------------------------------------- #
def test_sample_unchanged_regression():
    pol = _policy()
    proprio = torch.randn(3, 132)
    tokens = torch.randn(3, 2, TOK)

    g1 = torch.Generator().manual_seed(123)
    z1, lp1 = pol.sample(proprio, tokens, generator=g1)

    # Reproduit EXACTEMENT le calcul historique (pré-critique) → doit coïncider bit-à-bit.
    g2 = torch.Generator().manual_seed(123)
    mean = pol.forward(proprio, tokens)
    std = pol.log_std.clamp(LOG_STD_FLOOR, LOG_STD_CEIL).exp()
    eps = torch.randn(mean.shape, generator=g2)
    z_ref = mean + std * eps
    lp_ref = Normal(mean, std).log_prob(z_ref).sum(-1)

    assert torch.allclose(z1, z_ref, atol=0), "sample z a changé (régression Task-7)"
    assert torch.allclose(lp1, lp_ref, atol=0), "sample logprob a changé (régression Task-7)"
    assert z1.shape == (3, 2) and lp1.shape == (3,)


def test_forward_permutation_invariant_still():
    pol = _policy()
    proprio = torch.randn(4, 132)
    tokens = torch.randn(4, 2, TOK)
    out = pol.forward(proprio, tokens)
    out_swap = pol.forward(proprio, tokens.flip(dims=[1]))
    assert torch.allclose(out, out_swap, atol=1e-6)


# --------------------------------------------------------------------------- #
# Empaquetage : round-trip pack → unpack identique, pour D=2 et D=3.
# --------------------------------------------------------------------------- #
def test_pack_unpack_roundtrip():
    pol = _policy()
    for d in (2, 3):
        proprio = torch.randn(5, 132)
        tokens = torch.randn(5, d, TOK)
        packed = pol.pack_obs(proprio, tokens)
        assert packed.shape == (5, 132 + d * TOK)
        p2, t2 = pol.unpack_obs(packed)
        assert torch.equal(p2, proprio), f"proprio altéré (D={d})"
        assert torch.equal(t2, tokens), f"tokens altérés (D={d})"
        assert t2.shape == (5, d, TOK), f"D mal inféré (D={d}): {t2.shape}"


# --------------------------------------------------------------------------- #
# evaluate_actions : log_prob == Normal(forward, std).log_prob(z).sum(-1) ; value finie.
# --------------------------------------------------------------------------- #
def test_evaluate_actions_matches_forward():
    pol = _policy()
    proprio = torch.randn(6, 132)
    tokens = torch.randn(6, 2, TOK)
    z = torch.randn(6, 2)
    packed = pol.pack_obs(proprio, tokens)

    log_prob, entropy, value = pol.evaluate_actions(packed, z)

    mean = pol.forward(proprio, tokens)
    std = pol.log_std.clamp(LOG_STD_FLOOR, LOG_STD_CEIL).exp()
    dist = Normal(mean, std)
    assert torch.allclose(log_prob, dist.log_prob(z).sum(-1), atol=1e-6)
    assert torch.allclose(entropy, dist.entropy().sum(-1), atol=1e-6)
    assert value.shape == (6,) and torch.isfinite(value).all()


def test_value_and_mean_packed():
    pol = _policy()
    proprio = torch.randn(6, 132)
    tokens = torch.randn(6, 3, TOK)  # D=3 : le critique doit gérer une pulsion de plus
    packed = pol.pack_obs(proprio, tokens)
    v = pol.value(packed)
    m = pol.mean(packed)
    assert v.shape == (6,) and torch.isfinite(v).all()
    assert m.shape == (6, 2)
    assert torch.allclose(m, pol.forward(proprio, tokens), atol=1e-6)
    assert isinstance(pol.mean_std(), float)


# --------------------------------------------------------------------------- #
# load_state_dict tolérant : un checkpoint BC (SANS value_head) se charge (critique frais).
# --------------------------------------------------------------------------- #
def test_load_bc_without_value_head():
    pol = _policy()
    bc_state = {k: v for k, v in pol.state_dict().items() if not k.startswith("value_head")}
    fresh = DriveSymmetricPolicy()
    missing, unexpected = fresh.load_state_dict(bc_state)  # strict=False par défaut
    assert all(k.startswith("value_head") for k in missing), f"missing inattendu: {missing}"
    assert unexpected == [], f"unexpected: {unexpected}"
    # un checkpoint COMPLET (avec value_head) recharge exactement
    full = DriveSymmetricPolicy()
    m2, u2 = full.load_state_dict(pol.state_dict())
    assert m2 == [] and u2 == []
