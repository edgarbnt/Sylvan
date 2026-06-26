# python/sylvan/control/mode1/test_policy_sanity.py
# Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python python/sylvan/control/mode1/test_policy_sanity.py
import torch
from sylvan.control.mode1.policy import DriveSymmetricPolicy, map_action, TOK
from sylvan.control.mode1.obs import build_tokens, N_RAYS

def main():
    torch.manual_seed(0)
    pol = DriveSymmetricPolicy()
    proprio = torch.randn(4, 132)
    tokens = torch.randn(4, 2, TOK)  # 2 drives
    out = pol(proprio, tokens)
    assert out.shape == (4, 2), out.shape
    # INVARIANCE PAR PERMUTATION : échanger les 2 drives ne change pas la sortie
    out_swap = pol(proprio, tokens.flip(dims=[1]))
    assert torch.allclose(out, out_swap, atol=1e-6), "PAS invariant par permutation des drives !"
    # SCALABILITÉ : 1 drive et 3 drives passent sans erreur de poids (mêmes paramètres)
    assert pol(proprio, torch.randn(4, 1, TOK)).shape == (4, 2)
    assert pol(proprio, torch.randn(4, 3, TOK)).shape == (4, 2)
    # BORNES action
    cmd = map_action(torch.randn(1000, 2) * 5)
    assert (cmd[:, 0] >= 0.55).all() and (cmd[:, 0] <= 0.75).all(), "vx hors bornes"
    assert (cmd[:, 1] >= -0.6).all() and (cmd[:, 1] <= 0.6).all(), "ω hors bornes"
    print("OK: policy sanity (dims, invariance permutation, scalabilité 1/2/3 drives, bornes action)")

    # --- SANITY build_tokens ---
    # Construire une rétine synthétique : rays 0-1 clairement rouges, rays 2-3 clairement bleus
    retina = [0.0] * (4 * N_RAYS)  # 144 floats, tout à zéro (pas de hit = depth 0.0)
    # ray 0 : rouge fort, depth 0.5
    retina[0*4+0] = 0.5; retina[0*4+1] = 0.9; retina[0*4+2] = 0.1; retina[0*4+3] = 0.1
    # ray 1 : rouge fort, depth 0.3
    retina[1*4+0] = 0.3; retina[1*4+1] = 0.8; retina[1*4+2] = 0.05; retina[1*4+3] = 0.05
    # ray 2 : bleu fort, depth 0.6
    retina[2*4+0] = 0.6; retina[2*4+1] = 0.05; retina[2*4+2] = 0.05; retina[2*4+3] = 0.85
    # ray 3 : bleu fort, depth 0.4
    retina[3*4+0] = 0.4; retina[3*4+1] = 0.05; retina[3*4+2] = 0.05; retina[3*4+3] = 0.75

    payload_full = {
        "proprio": [0.0] * 132,
        "retina": retina,
        "energy": 50.0,
        "thirst": 80.0,
    }

    # Test avec 2 drives (faim + soif)
    p, toks, meta = build_tokens(payload_full)
    assert p.shape == (132,), f"proprio shape: {p.shape}"
    assert toks.shape == (2, 38), f"tokens shape avec soif: {toks.shape}"
    assert meta == ["food", "water"], f"meta: {meta}"

    # Vérifier que le gating rouge fonctionne : rays 0 et 1 doivent avoir leur depth dans le token food
    food_tok = toks[0]  # [38]: [energy, valence, depth_ray0, ..., depth_ray35]
    assert abs(food_tok[2].item() - 0.5) < 1e-5, f"ray0 rouge: attendu 0.5, obtenu {food_tok[2].item()}"
    assert abs(food_tok[3].item() - 0.3) < 1e-5, f"ray1 rouge: attendu 0.3, obtenu {food_tok[3].item()}"
    # rays 2 et 3 ne sont pas rouges → doivent valoir 1.0 dans le token food
    assert abs(food_tok[4].item() - 1.0) < 1e-5, f"ray2 dans food: attendu 1.0 (pas rouge), obtenu {food_tok[4].item()}"

    # Vérifier que le gating bleu fonctionne : rays 2 et 3 doivent avoir leur depth dans le token water
    water_tok = toks[1]  # [38]
    assert abs(water_tok[4].item() - 0.6) < 1e-5, f"ray2 bleu: attendu 0.6, obtenu {water_tok[4].item()}"
    assert abs(water_tok[5].item() - 0.4) < 1e-5, f"ray3 bleu: attendu 0.4, obtenu {water_tok[5].item()}"
    # ray 0 n'est pas bleu → doit valoir 1.0 dans le token water
    assert abs(water_tok[2].item() - 1.0) < 1e-5, f"ray0 dans water: attendu 1.0 (pas bleu), obtenu {water_tok[2].item()}"

    # Test sans thirst : 1 seul drive (faim seulement)
    payload_mono = {
        "proprio": [0.0] * 132,
        "retina": retina,
        "energy": 50.0,
    }
    p2, toks2, meta2 = build_tokens(payload_mono)
    assert p2.shape == (132,), f"proprio shape mono: {p2.shape}"
    assert toks2.shape == (1, 38), f"tokens shape sans soif: {toks2.shape}"
    assert meta2 == ["food"], f"meta mono: {meta2}"

    print("OK: build_tokens (proprio[132], tokens[2,38]/[1,38], gating rouge+bleu, meta food/water)")

if __name__ == "__main__":
    main()
