"""Diagnostic : le slot centroïde-t-il plusieurs instances de nourriture ?

Le slot localise la nourriture par centroïde soft-argmax sur TOUS les rayons
qui passent sa requête couleur. Quand plusieurs instances sont visibles dans le
champ, il pointe leur barycentre — qui n'est la position d'aucune d'elles.

Ce script vérifie l'hypothèse en comparant la sortie du slot à DEUX cibles :
- la nourriture la PLUS PROCHE (food_rel0, oracle)
- le centroïde de TOUS les rayons qui passent la requête couleur

Si slot ≈ centroïde ≠ plus proche → l'hypothèse est vérifiée.
Il compte aussi le nombre d'instances visibles par tick (clustering angulaire).

Gratuit : zéro Godot, zéro entraînement.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.models.command_wm import CommandWorldModel  # noqa: E402

# Reproduit les constantes de diag_slot_localise
RANGE_M = 10.0
DEPTH_OFFSET = 0.35
_HALF_FOV = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")) / 2.0


def load_pairs(corpus: str, key: str) -> tuple[torch.Tensor, torch.Tensor, list]:
    """(rétines, positions vraies, raw_lines) sur les ticks où la ressource est visible."""
    ret, tgt, lines = [], [], []
    for f in sorted(glob.glob(os.path.join(corpus, "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("wm", {}).get(key)
            if not v or len(v) < 3 or v[2] <= 0.5:
                continue
            bearing = math.degrees(math.atan2(v[0], v[1]))
            if abs(bearing) > _HALF_FOV:
                continue
            ret.append(r["obs"]["retina"])
            tgt.append([v[0], v[1]])
            lines.append(r)
    if not ret:
        raise SystemExit(f"aucun tick avec {key} visible dans {corpus}")
    return torch.tensor(ret, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32), lines


def prepare_wm(wm_path: str) -> CommandWorldModel:
    payload = torch.load(wm_path, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    if not meta.get("with_slot"):
        raise SystemExit(f"{wm_path} n'a pas de canal-slot")
    wm = CommandWorldModel.from_checkpoint(payload)
    wm.load_state_dict(payload["model"])
    qthr = meta.get("query_thr")
    if qthr is not None and getattr(wm.slot_encoder, "query_thr", None) is not None:
        wm.slot_encoder.query_thr.copy_(torch.tensor(qthr, dtype=torch.float32))
    fov = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360"))
    if abs(fov - 360.0) > 1e-6:
        n = wm.slot_encoder.sin.shape[0]
        th = torch.tensor(
            [(k if k <= n // 2 else k - n) * math.radians(fov) / n for k in range(n)],
            dtype=torch.float32
        )
        wm.slot_encoder.sin.copy_(torch.sin(th))
        wm.slot_encoder.cos.copy_(torch.cos(th))
    wm.eval()
    return wm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2_slot/wm_best.pt")
    ap.add_argument("--corpus", default="data/replay_buffer/gate_foret_cl")
    a = ap.parse_args()

    print(f"=== SLOT = CENTROÏDE MULTI-INSTANCE ? | {a.wm} | {a.corpus} ===")
    wm = prepare_wm(a.wm)
    se = wm.slot_encoder
    cos_thr = se.query_thr[0].item()
    n_rays = se.sin.shape[0]
    fov = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360"))

    retina, truth_f, _ = load_pairs(a.corpus, "food_rel0")
    print(f"Chargé : {retina.shape[0]} ticks, FOV {fov:.0f}°, seuil cos bouffe = {cos_thr:.3f}\n")

    # Slot output
    with torch.no_grad():
        slot_pos = se.positions(retina)  # [B, n_resources, 2]

    live = slot_pos[:, 0, :].abs().sum(dim=1) > 1e-6
    n_live = live.sum().item()

    # Rétine en (B, 36, 4) pour analyse manuelle
    ret36 = retina.view(-1, 36, 4)
    dist = ret36[..., 0].float() * RANGE_M + DEPTH_OFFSET  # [B, 36]

    # Bearing arrays (precomputed)
    sin_a = se.sin.to(retina.device).unsqueeze(0)  # [1, 36]
    cos_a = se.cos.to(retina.device).unsqueeze(0)

    err_nearest_list, err_centroid_list, sc_list = [], [], []
    n_instances_list = []
    nearest_dir_err_list, centroid_dir_err_list = [], []

    for b in range(retina.shape[0]):
        if not live[b]:
            continue
        rgb = ret36[b, :, 1:4].float()
        rgb_n = rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-8)
        q = se.color_queries[0]
        hit = (rgb_n * q).sum(dim=-1) > cos_thr
        nh = hit.sum().item()
        if nh < 1:
            continue

        dx = dist[b] * sin_a[0]
        dz = dist[b] * cos_a[0]
        cx = dx[hit].mean().item()
        cz = dz[hit].mean().item()

        # Erreurs (m)
        tn = torch.tensor([truth_f[b, 0].item(), truth_f[b, 1].item()])
        err_n = float(torch.norm(slot_pos[b, 0, :] - tn).item())
        err_c = float(math.hypot(cx - tn[0].item(), cz - tn[1].item()))
        sc = float(math.hypot(cx - slot_pos[b, 0, 0].item(), cz - slot_pos[b, 0, 1].item()))
        err_nearest_list.append(err_n)
        err_centroid_list.append(err_c)
        sc_list.append(sc)

        # Erreurs angulaires (deg)
        b_slot = math.degrees(math.atan2(slot_pos[b, 0, 1].item(), slot_pos[b, 0, 0].item()))
        b_true = math.degrees(math.atan2(tn[1].item(), tn[0].item()))
        b_cent = math.degrees(math.atan2(cz, cx))
        def delta(a, b):
            d = a - b
            while d > 180: d -= 360
            while d < -180: d += 360
            return abs(d)
        nearest_dir_err_list.append(delta(b_slot, b_true))
        centroid_dir_err_list.append(delta(b_cent, b_true))

        # Clustering angulaire : instances visibles
        bearings = [(k, math.degrees(math.atan2(sin_a[0, k].item(), cos_a[0, k].item())))
                    for k in range(n_rays) if hit[k].item()]
        bearings.sort(key=lambda x: x[1])
        clusters = 1 if len(bearings) >= 1 else 0
        for i in range(1, len(bearings)):
            if bearings[i][1] - bearings[i-1][1] > 12:
                clusters += 1
        n_instances_list.append(clusters)

    if not err_nearest_list:
        print("Aucun tick exploitable.")
        return 1

    def fmt(vals):
        t = torch.tensor(vals, dtype=torch.float32)
        return f"méd={t.median():.2f} moy={t.mean():.2f}"

    print(f"1. Slot vs plus PROCHE (food_rel0):")
    print(f"   erreur position {fmt(err_nearest_list)} m")
    print(f"   erreur gisement {fmt(nearest_dir_err_list)}")
    print()

    print(f"2. Centroide TOUS rayons bouffe vs plus PROCHE:")
    print(f"   erreur position {fmt(err_centroid_list)} m")
    print(f"   erreur gisement {fmt(centroid_dir_err_list)}")
    print()

    print(f"3. Slot vs centroide:")
    print(f"   erreur {fmt(sc_list)} m")
    print()

    # Interprétation
    m_sc = torch.tensor(sc_list, dtype=torch.float32).median()
    m_en = torch.tensor(err_nearest_list, dtype=torch.float32).median()
    m_ec = torch.tensor(err_centroid_list, dtype=torch.float32).median()
    if m_sc < 0.5 and abs(m_en - m_ec) < 0.5:
        print("✅ CONFIRMÉ : slot ≈ centroïde multi-instance")
        print("   slot = centroïde ≠ plus proche → le problème est le centroïde")
    elif m_sc < 0.5 and m_ec < m_en:
        print("🟡 PARTIEL : slot ≈ centroïde, mais le centroïde n'est pas pire que le plus proche")
    elif m_sc > 0.5:
        print("🟡 NÉGATIF : slot ≠ centroïde multi-instance")
        print(f"   slot vs centroïde: {m_sc:.2f} m → autre cause probable")

    print(f"\n4. Instances visibles par tick (clusters angulaires)")
    cnt = {}
    for v in n_instances_list:
        cnt[v] = cnt.get(v, 0) + 1
    for v in sorted(cnt):
        print(f"   {v} instance(s) : {cnt[v]} ticks ({100*cnt[v]/len(n_instances_list):.0f}%)")
    t_inst = torch.tensor(n_instances_list, dtype=torch.float32)
    print(f"   médiane={t_inst.median():.0f} moyenne={t_inst.mean():.2f}")

    # Corrélation nb-instances vs erreur (torch)
    t_inst2 = torch.tensor(n_instances_list[:len(err_nearest_list)], dtype=torch.float32)
    t_err2 = torch.tensor(err_nearest_list, dtype=torch.float32)
    r_mat = torch.corrcoef(torch.stack([t_inst2, t_err2]))
    r = float(r_mat[0, 1])
    print(f"   corrélation nb-instances vs erreur slot: r={r:.3f}")

    return 0 if m_sc < 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
