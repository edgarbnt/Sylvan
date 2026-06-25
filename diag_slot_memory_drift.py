"""diag_slot_memory_drift.py — MÉMOIRE SPATIALE, gate gratuit décisif (CLAUDE.md §1).
Mesure la dérive d'un belief-slot dead-reckoné par l'ego-motion RÉELLE (torso) pendant une
occlusion artificielle, vs vérité-terrain food_rel0. Décide si la mémoire est FAISABLE avant tout build.
Usage:
  PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_slot_memory_drift.py --selfcheck
  BUFS="retina_wm_a retina_wm_b" PYTHONPATH=python ./env_pytorch_3.12/bin/python diag_slot_memory_drift.py
"""
import os, sys, json, glob, math, collections
import torch

DEVICE = "cpu"
BUFS = os.environ.get("BUFS", "retina_wm_a retina_wm_b").split()
WM_CKPT = os.environ.get("WM_CKPT", "data/checkpoints/wm_objcentric_s1/wm_best.pt")
NS = [5, 10, 20, 40]
STRIDE_K = 15


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def egomotion_from_torso(t0, t1):
    x0, z0, yaw0 = t0; x1, z1, yaw1 = t1
    dyaw = wrap(yaw1 - yaw0)
    dx, dz = x1 - x0, z1 - z0
    dfwd = dx * math.sin(yaw0) + dz * math.cos(yaw0)
    dlat = dx * math.cos(yaw0) - dz * math.sin(yaw0)
    return dyaw, dfwd, dlat


def transport_geom(p, dyaw, dfwd, dlat):
    # Convention NAILÉE par la calibration test5/test6 (kyaw=-1) et command_wm.transport_slot (ky=-1) :
    # un point monde-statique vu en ego à t, après un pas d'ego-motion réelle (dyaw,dfwd,dlat),
    # se transporte par translate(−déplacement ego) puis ROTATE R(+dyaw). (Le raw test5 R(−dyaw) était pré-calib.)
    px, pz = p[0] - dlat, p[1] - dfwd
    ca, sa = math.cos(dyaw), math.sin(dyaw)
    return [ca * px - sa * pz, sa * px + ca * pz]


def selfcheck():
    # (1) agent immobile → belief inchangé
    p = [0.4, 2.1]
    assert max(abs(a - b) for a, b in zip(transport_geom(p, 0.0, 0.0, 0.0), p)) < 1e-9
    # (2) ROUND-TRIP convention : un point monde-statique vu à t0, après un pas réel de l'agent,
    #     transporté par l'ego-motion de CE pas, doit retomber sur sa position ego réelle à t1.
    import random
    random.seed(0)

    def to_ego(t, f):
        dx, dz = f[0] - t[0], f[1] - t[1]; y = t[2]
        # repère ego cohérent avec egomotion_from_torso : x_right = dx·cos − dz·sin, z_fwd = dx·sin + dz·cos
        return [dx * math.cos(y) - dz * math.sin(y), dx * math.sin(y) + dz * math.cos(y)]

    worst = 0.0
    for _ in range(5000):
        yaw0 = random.uniform(-math.pi, math.pi)
        t0 = [random.uniform(-3, 3), random.uniform(-3, 3), yaw0]
        t1 = [t0[0] + random.uniform(-0.3, 0.3), t0[1] + random.uniform(-0.3, 0.3),
              wrap(yaw0 + random.uniform(-0.4, 0.4))]
        fw = [random.uniform(-4, 4), random.uniform(-4, 4)]
        p0 = to_ego(t0, fw); p1 = to_ego(t1, fw)
        dyaw, dfwd, dlat = egomotion_from_torso(t0, t1)
        pred = transport_geom(p0, dyaw, dfwd, dlat)
        worst = max(worst, math.hypot(pred[0] - p1[0], pred[1] - p1[1]))
    assert worst < 1e-6, f"round-trip cassé: worst={worst} (convention egomotion/transport)"
    print(f"[selfcheck] OK — convention egomotion↔transport_geom validée (round-trip worst={worst:.2e}).")


def load_eps():
    eps = []
    for buf in BUFS:
        for f in sorted(glob.glob(f"data/replay_buffer/{buf}/*.jsonl")):
            seq = []
            for line in open(f):
                r = json.loads(line); w = r.get("wm", {})
                ret = w.get("retina0"); fr = w.get("food_rel0"); t0 = w.get("torso0")
                if not ret or not fr or not t0:
                    continue
                seq.append({"retina": ret, "food": [float(fr[0]), float(fr[1])],
                            "vis": float(fr[2]), "torso": [float(t0[0]), float(t0[1]), float(t0[2])]})
            if len(seq) > 50:
                eps.append(seq)
    return eps


def load_encoder():
    from sylvan.models.command_wm import CommandWorldModel
    ck = torch.load(WM_CKPT, map_location=DEVICE, weights_only=False)
    m = ck["meta"]
    wm = CommandWorldModel(obs_dim=m["obs_dim"], proprio_dim=m["proprio_dim"],
                           predictor_arch=m.get("predictor_arch", "shallow"),
                           with_slot=True, slot_resources=m.get("slot_resources", 1))
    wm.load_state_dict(ck["model"]); wm.eval()
    return wm


@torch.no_grad()
def encode_slot_from_retina(wm, retina):
    r = torch.tensor(retina, dtype=torch.float32).unsqueeze(0)
    return wm.slot_encoder.positions(r)[0, 0, :].tolist()


def median(xs):
    xs = sorted(xs); n = len(xs)
    return float("nan") if n == 0 else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def drift_runs(eps, wm):
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    maxN = max(NS)
    for seq in eps:
        for k in range(0, len(seq) - maxN - 1, STRIDE_K):
            if seq[k]["vis"] < 0.5:
                continue
            bk_a = encode_slot_from_retina(wm, seq[k]["retina"])
            bk_b = list(seq[k]["food"])
            beliefs = {"a_encode": list(bk_a), "b_truth": list(bk_b)}
            frozen = {"a_encode": list(bk_a), "b_truth": list(bk_b)}
            for n in range(1, maxN + 1):
                dyaw, dfwd, dlat = egomotion_from_torso(seq[k + n - 1]["torso"], seq[k + n]["torso"])
                for v in beliefs:
                    beliefs[v] = transport_geom(beliefs[v], dyaw, dfwd, dlat)
                if n in NS:
                    truth = seq[k + n]["food"]
                    bt = math.atan2(truth[0], truth[1])
                    for v in beliefs:
                        bp = math.atan2(beliefs[v][0], beliefs[v][1])
                        agg[v]["deadreckon"][n].append((abs(math.degrees(wrap(bp - bt))),
                                                        math.hypot(beliefs[v][0] - truth[0], beliefs[v][1] - truth[1])))
                        fp = math.atan2(frozen[v][0], frozen[v][1])
                        agg[v]["frozen"][n].append((abs(math.degrees(wrap(fp - bt))),
                                                    math.hypot(frozen[v][0] - truth[0], frozen[v][1] - truth[1])))
    return agg


def noise_sweep(eps, wm, fracs=(0.05, 0.10, 0.20)):
    """Sensibilité à l'erreur d'ego-motion (cas DÉPLOIEMENT : ego-motion estimée du proprio, pas l'oracle).
    Bruit gaussien par-pas sur (dyaw,dfwd,dlat), std = frac × RMS de la composante. F2 (+0.98) ≈ ~14% résiduel."""
    import random
    rng = random.Random(7)
    # RMS par composante sur l'échantillon
    sd = {"dyaw": [], "dfwd": [], "dlat": []}
    for seq in eps[:80]:
        for i in range(len(seq) - 1):
            dyaw, dfwd, dlat = egomotion_from_torso(seq[i]["torso"], seq[i + 1]["torso"])
            sd["dyaw"].append(dyaw); sd["dfwd"].append(dfwd); sd["dlat"].append(dlat)
    rms = {k: (sum(x * x for x in v) / max(1, len(v))) ** 0.5 for k, v in sd.items()}
    print(f"\n  RMS ego-motion/pas : dyaw={rms['dyaw']:.3f}rad dfwd={rms['dfwd']:.3f}m dlat={rms['dlat']:.3f}m")
    print("  Sensibilité (a_encode, ego-motion BRUITÉE) — position MAE médiane (m) :")
    maxN = max(NS)
    for frac in fracs:
        per_n = {n: [] for n in NS}
        for seq in eps:
            for k in range(0, len(seq) - maxN - 1, STRIDE_K):
                if seq[k]["vis"] < 0.5:
                    continue
                belief = list(encode_slot_from_retina(wm, seq[k]["retina"]))
                for n in range(1, maxN + 1):
                    dyaw, dfwd, dlat = egomotion_from_torso(seq[k + n - 1]["torso"], seq[k + n]["torso"])
                    dyaw += rng.gauss(0, frac * rms["dyaw"])
                    dfwd += rng.gauss(0, frac * rms["dfwd"])
                    dlat += rng.gauss(0, frac * rms["dlat"])
                    belief = transport_geom(belief, dyaw, dfwd, dlat)
                    if n in NS:
                        tr = seq[k + n]["food"]
                        per_n[n].append(math.hypot(belief[0] - tr[0], belief[1] - tr[1]))
        line = " ".join(f"N={n}:{median(per_n[n]):.2f}m" for n in NS)
        print(f"    bruit {int(frac*100):>2}% RMS : {line}")


def main():
    eps = load_eps()
    print(f"épisodes={len(eps)} ; frames={sum(len(e) for e in eps)} ; bufs={BUFS}")
    if not eps:
        print("AUCUN épisode chargé — vérifier les buffers / champs torso0,food_rel0,retina0."); return
    wm = load_encoder()
    agg = drift_runs(eps, wm)
    PASS_BRG, PASS_POS, PASS_N = 20.0, 0.5, 30
    print("\nMÉDIANES (bearing MAE °, position MAE m) par variante / N :")
    for v in ("b_truth", "a_encode"):
        print(f"\n  variante {v}:")
        for n in NS:
            dr = agg[v]["deadreckon"][n]; fr = agg[v]["frozen"][n]
            drb, drp = median([e[0] for e in dr]), median([e[1] for e in dr])
            frb, frp = median([e[0] for e in fr]), median([e[1] for e in fr])
            print(f"    N={n:>2}: dead-reckon brg={drb:5.1f}° pos={drp:4.2f}m | "
                  f"gelé brg={frb:5.1f}° pos={frp:4.2f}m | n={len(dr)}")
    v = "a_encode"; dr = agg[v]["deadreckon"]; fr = agg[v]["frozen"]
    ncheck = max([n for n in NS if n <= PASS_N])
    drb = median([e[0] for e in dr[ncheck]]); drp = median([e[1] for e in dr[ncheck]])
    frp = median([e[1] for e in fr[ncheck]])
    beats_frozen = drp < 0.8 * frp
    ok = (drb < PASS_BRG) and (drp < PASS_POS) and beats_frozen
    print(f"\n  >>> VERDICT (variante réaliste a_encode, N={ncheck}): "
          f"brg={drb:.1f}°(<{PASS_BRG}) pos={drp:.2f}m(<{PASS_POS}) bat_gelé={beats_frozen}(pos<{0.8*frp:.2f}) → "
          f"{'PASS — build autorisé' if ok else 'FAIL/KILL — STOP + escalade (CLAUDE.md §1)'}")
    noise_sweep(eps, wm)


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck(); sys.exit(0)
    main()
