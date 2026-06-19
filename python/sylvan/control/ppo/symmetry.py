"""Left-right MIRROR maps for the HEXAPOD (tripod gait), for symmetry (equivariance) augmentation.

The body is geometrically symmetric about the sagittal plane (x=0; +x is the body's right).
A policy trained without symmetry drifts one way (measured: hexapod_v1 veers ~+12 deg/100 on a
straight command). Enforcing pi(mirror(obs)) == mirror(pi(obs)) removes that learned drift cleanly
(not a reward patch) and tends to clean up the gait coordination.

Reflection across x=0. Under it:
  - true vectors (lin vel, COM, orientation-axis world coords): negate the x component.
  - pseudovectors (angular velocity): keep x, negate y,z.
  - left<->right legs swap (l1<->r1, l2<->r2, l3<->r3); the trunk is on the centerline (maps to self).
  - the lateral DOF (hip_z abduction) negates; sagittal (hip_x) and knee keep sign.
  - CPG command [vx, omega, 0x10]: keep vx, NEGATE omega (a mirrored left-turn IS a right-turn).

HEXAPOD proprio layout (132), from sylvan_agent.gd::_rebuild_proprioception:
  [0] height | [1:4] torso lin vel xyz | [4:7] torso ang vel xyz
  [7:85] 13 bodies x 6 (basis.y xyz, -basis.z xyz); BODY_NAMES order
         [torso, l1u,l1l, r1u,r1l, l2u,l2l, r2u,r2l, l3u,l3l, r3u,r3l]
  [85:91] foot contacts [l1,r1,l2,r2,l3,r3] | [91:94] COM xyz
  [94:112] 18 joint angles: 6 legs x [hip_x,hip_z,knee] (l1,r1,l2,r2,l3,r3)
  [112:130] joint velocities (same order) | [130:132] gait clock [sin,cos]
Action (18): same per-leg [hip_x,hip_z,knee] order (l1,r1,l2,r2,l3,r3).
Obs fed to the policy = [proprio(132) ++ vision(12)] = 144.
"""

from __future__ import annotations

import torch

_PROPRIO = 132
_VISION = 12
_OBS = _PROPRIO + _VISION  # 144
_ACT = 18

# --- Action (18): swap l<->r per pair (l1<->r1, l2<->r2, l3<->r3); negate hip_z (mid of each triple) ---
_ACT_PERM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 15, 16, 17, 12, 13, 14]
_ACT_SIGN = [1., -1., 1.] * 6


def _build_proprio_maps() -> tuple[list[int], list[float]]:
    perm = list(range(_PROPRIO))
    sign = [1.0] * _PROPRIO

    def setp(i: int, src: int, s: float) -> None:
        perm[i] = src
        sign[i] = s

    setp(0, 0, 1)                                   # height
    setp(1, 1, -1); setp(2, 2, 1); setp(3, 3, 1)    # torso lin vel: negate x
    setp(4, 4, 1); setp(5, 5, -1); setp(6, 6, -1)   # torso ang vel (pseudo): keep x, negate y,z

    # 13 body orientation blocks (6 each) at offset 7; per-block sign [-1,1,1,-1,1,1] (negate x of basis.y
    # and of -basis.z). Trunk (0) maps to self; legs swap L<->R within each pair (front/mid/rear).
    # BODY_NAMES idx: 0 torso, 1 l1u,2 l1l, 3 r1u,4 r1l, 5 l2u,6 l2l, 7 r2u,8 r2l, 9 l3u,10 l3l, 11 r3u,12 r3l
    src_block = [0, 3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10]
    bsign = [-1., 1., 1., -1., 1., 1.]
    for k in range(13):
        sk = src_block[k]
        for j in range(6):
            setp(7 + 6 * k + j, 7 + 6 * sk + j, bsign[j])

    # foot contacts [85:91] = [l1,r1,l2,r2,l3,r3]: swap l<->r per pair
    setp(85, 86, 1); setp(86, 85, 1); setp(87, 88, 1); setp(88, 87, 1); setp(89, 90, 1); setp(90, 89, 1)
    setp(91, 91, -1); setp(92, 92, 1); setp(93, 93, 1)                  # COM: negate x

    # joint angles [94:112] and velocities [112:130]: swap L<->R legs per pair, negate hip_z (mid of triple).
    # Triples: l1@base, r1@base+3, l2@base+6, r2@base+9, l3@base+12, r3@base+15.
    for base in (94, 112):
        for (a, b) in ((base, base + 3), (base + 6, base + 9), (base + 12, base + 15)):
            # a (left) <-> b (right): swap, negate hip_z (offset +1 within each triple)
            setp(a, b, 1);     setp(a + 1, b + 1, -1);     setp(a + 2, b + 2, 1)
            setp(b, a, 1);     setp(b + 1, a + 1, -1);     setp(b + 2, a + 2, 1)

    # gait clock [sin,cos]: the tripod is chiral vs the clock — its L-R mirror is the SAME gait half a
    # cycle later (tripod A <-> tripod B), i.e. phi->phi+0.5 => sin->-sin, cos->-cos. Involutive.
    setp(130, 130, -1); setp(131, 131, -1)
    return perm, sign


def _build_obs_maps() -> tuple[list[int], list[float]]:
    import os
    pperm, psign = _build_proprio_maps()
    if os.environ.get("SYLVAN_MIRROR_COMMAND") == "1":
        # CPG COMMAND mode: vision = [vx, omega, 0*10]. Keep vx, NEGATE omega; zero pad identity.
        vperm = [_PROPRIO + i for i in range(_VISION)]
        vsign = [1.0, -1.0] + [1.0] * (_VISION - 2)
    else:
        # RADAR mode (egocentric food radar): sector s <-> (V-1)-s.
        vperm = [_PROPRIO + (_VISION - 1 - s) for s in range(_VISION)]
        vsign = [1.0] * _VISION
    return pperm + vperm, psign + vsign


_OBS_PERM_L, _OBS_SIGN_L = _build_obs_maps()
OBS_PERM = torch.tensor(_OBS_PERM_L, dtype=torch.long)
OBS_SIGN = torch.tensor(_OBS_SIGN_L, dtype=torch.float32)
ACT_PERM = torch.tensor(_ACT_PERM, dtype=torch.long)
ACT_SIGN = torch.tensor(_ACT_SIGN, dtype=torch.float32)


def mirror_obs(obs: torch.Tensor) -> torch.Tensor:
    """[..., 144] -> left-right mirrored observation."""
    return obs[..., OBS_PERM.to(obs.device)] * OBS_SIGN.to(obs.device, obs.dtype)


def mirror_action(act: torch.Tensor) -> torch.Tensor:
    """[..., 18] -> left-right mirrored action."""
    return act[..., ACT_PERM.to(act.device)] * ACT_SIGN.to(act.device, act.dtype)


def self_check() -> None:
    """mirror is an involution (mirror(mirror(x)) == x) and the action map swaps L<->R legs
    (a left-skid turn mirrors to a right-skid turn)."""
    x = torch.randn(5, _OBS)
    assert torch.allclose(mirror_obs(mirror_obs(x)), x, atol=1e-6), "obs mirror not involutive"
    a = torch.randn(5, _ACT)
    assert torch.allclose(mirror_action(mirror_action(a)), a, atol=1e-6), "action mirror not involutive"
    # turn-skid: left legs hip_x +0.5, right legs hip_x -0.5 → mirror swaps to the opposite skid.
    turnL = torch.zeros(1, _ACT)
    turnR = torch.zeros(1, _ACT)
    for tri in (0, 6, 12):       # left triples l1,l2,l3 (hip_x at triple base)
        turnL[0, tri] = 0.5
        turnR[0, tri] = -0.5
    for tri in (3, 9, 15):       # right triples r1,r2,r3
        turnL[0, tri] = -0.5
        turnR[0, tri] = 0.5
    assert torch.allclose(mirror_action(turnL), turnR), "action mirror != expected turnL<->turnR"
