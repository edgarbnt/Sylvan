"""Modèle d'apparence SYNTHÉTIQUE — perturbe des rétines à couleur plate pour SIMULER un monde à
apparences variées, SANS toucher à Godot (docs/design_perception_types.md, gate G-pré + diag WM).

Sert les diags GRATUITS qui gatent le travail Godot : « la reconnaissance / le WM tiennent-ils si
l'apparence varie ? » — perturbation appliquée UNIQUEMENT aux rayons touchants (depth<0.999), la
profondeur et les rayons vides intacts. Trois axes : teinte (rotation autour du gris), texture
(bruit par-rayon), désaturation (tirage vers le gris). ⚠️ Synthétique ≠ rendu réel : borne le
risque pour zéro coût, ne remplace pas le check open-loop du vrai bump.
"""

from __future__ import annotations

import math

import torch

from sylvan.models.perception_head import RETINA_DIM

NRAY = RETINA_DIM // 4                    # 36 rayons × [depth, R, G, B]
# magnitudes graduées ; index MODERATE = « modéré réaliste » (porte les verdicts pré-enregistrés).
HUE_DEG = [0.0, 10.0, 20.0, 30.0]
TEX_SIGMA = [0.0, 0.02, 0.05, 0.10]
DESAT = [0.0, 0.2, 0.4, 0.6]
MODERATE = 2


def hue_matrix(deg: float) -> torch.Tensor:
    """Rotation RGB autour de l'axe gris (1,1,1) — « même luminance, teinte différente »."""
    th = math.radians(deg)
    k = torch.tensor([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    kx = torch.tensor([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return (math.cos(th) * torch.eye(3) + math.sin(th) * kx
            + (1.0 - math.cos(th)) * torch.outer(k, k))


def touch_mask(retina: torch.Tensor) -> torch.Tensor:
    """[.,144] → [.,36] bool : rayon ayant touché un objet coloré (depth<0.999)."""
    return retina.view(*retina.shape[:-1], NRAY, 4)[..., 0] < 0.999


def apply_rgb(retina: torch.Tensor, fn) -> torch.Tensor:
    """Applique fn(rgb)→rgb aux rayons touchants, clampe [0,1] ; depth + rayons vides intacts."""
    r = retina.view(*retina.shape[:-1], NRAY, 4).clone()
    m = r[..., 0] < 0.999
    rgb = r[..., 1:4]
    new = fn(rgb).clamp(0.0, 1.0)
    r[..., 1:4] = torch.where(m.unsqueeze(-1), new, rgb)
    return r.view(retina.shape)


def perturb(retina: torch.Tensor, kind: str, mag: float, gen: torch.Generator) -> torch.Tensor:
    """kind ∈ {hue (deg), texture (σ), desat (α∈[0,1])}. Déterministe via `gen`."""
    if kind == "hue":
        M = hue_matrix(mag)
        return apply_rgb(retina, lambda rgb: rgb @ M.T)
    if kind == "texture":
        return apply_rgb(retina, lambda rgb: rgb + mag * torch.randn(rgb.shape, generator=gen))
    if kind == "desat":
        return apply_rgb(retina, lambda rgb: rgb + mag * (rgb.mean(-1, keepdim=True) - rgb))
    raise ValueError(kind)


def selfcheck() -> None:
    assert NRAY == 36 and RETINA_DIM == 144
    assert torch.allclose(hue_matrix(0.0), torch.eye(3), atol=1e-6)
    g = torch.tensor([[0.4, 0.4, 0.4]])
    assert torch.allclose(g @ hue_matrix(20.0).T, g, atol=1e-5)             # gris invariant
    v = torch.tensor([[0.9, 0.1, 0.1]])
    assert abs(float((v @ hue_matrix(20.0).T).norm()) - float(v.norm())) < 1e-4  # norme préservée
    ret = torch.tensor([[0.2, 0.9, 0.1, 0.1] + [1.0, 0.0, 0.0, 0.0] * 35])
    gen = torch.Generator().manual_seed(0)
    assert torch.allclose(perturb(ret, "texture", 0.0, gen), ret)           # σ=0 = identité
    assert torch.allclose(perturb(ret, "desat", 0.0, gen), ret)             # α=0 = identité
    p = perturb(ret, "desat", 1.0, gen).view(NRAY, 4)
    assert torch.allclose(p[0, 1:4], p[0, 1:4].mean().expand(3), atol=1e-5)  # desat total = gris
    assert torch.all(perturb(ret, "hue", 30.0, gen).view(NRAY, 4)[1] == ret.view(NRAY, 4)[1])  # vide intact
    assert bool(touch_mask(ret)[0, 0]) and not bool(touch_mask(ret)[0, 1])
    print("[selfcheck] appearance_synth OK — teinte/texture/désat, rayons vides intacts")


if __name__ == "__main__":
    selfcheck()
