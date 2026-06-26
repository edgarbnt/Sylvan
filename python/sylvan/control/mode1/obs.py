# python/sylvan/control/mode1/obs.py
import torch

N_RAYS = 36
# couleurs-corps (food_manager.gd : rouge=bouffe, bleu=eau)
RED = "red"; BLUE = "blue"

def _color_gated_depths(retina, color):
    """retina = 144 floats = 36×[depth,R,G,B]. Retourne 36 profondeurs : depth si le rayon matche la couleur, sinon 1.0."""
    out = []
    for r in range(N_RAYS):
        d, R, G, B = retina[4*r], retina[4*r+1], retina[4*r+2], retina[4*r+3]
        if color == RED:
            hit = (R > G) and (R > B) and (R > 0.3) and (d < 0.999)
        else:  # BLUE
            hit = (B > R) and (B > G) and (B > 0.3) and (d < 0.999)
        out.append(d if hit else 1.0)
    return out

def build_tokens(payload: dict):
    """Construit (proprio[132], tokens[D,38], drive_meta) depuis le payload TCP du serveur.
    Drives actifs : faim (toujours, rouge) ; soif (si 'thirst' présent, bleu). Valence +1 (approcher-consommer)."""
    proprio = torch.tensor(payload["proprio"], dtype=torch.float32)
    retina = payload["retina"]
    assert len(retina) == 4 * N_RAYS, f"retina attendue {4*N_RAYS}, reçue {len(retina)} (SYLVAN_RETINA_PLANNER=1 ?)"
    toks, meta = [], []
    # token FAIM (rouge)
    e = float(payload.get("energy", 0.0)) / 100.0
    toks.append([e, 1.0] + _color_gated_depths(retina, RED)); meta.append("food")
    # token SOIF (bleu) — seulement si la pulsion existe
    if "thirst" in payload and payload["thirst"] is not None:
        t = float(payload["thirst"]) / 100.0
        toks.append([t, 1.0] + _color_gated_depths(retina, BLUE)); meta.append("water")
    tokens = torch.tensor(toks, dtype=torch.float32)  # [D,38]
    return proprio, tokens, meta
