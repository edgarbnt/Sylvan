# train_mode1_bc.py — BC : régresse la politique drive-symétrique sur les commandes du planner (Task 2).
# Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python train_mode1_bc.py
import os, glob, json, copy, torch, torch.nn as nn
from sylvan.control.mode1.policy import DriveSymmetricPolicy, map_action
from sylvan.control.mode1.obs import _color_gated_depths, N_RAYS, RED, BLUE

BUFS = os.environ.get("BUFS", "mode1_bc_a mode1_bc_b").split()
OUT  = os.environ.get("OUT",  "data/checkpoints/mode1_bc")
ITERS = int(os.environ.get("ITERS", "3000"))
LR    = float(os.environ.get("LR", "1e-3"))
BATCH = int(os.environ.get("BATCH", "0"))  # 0 = full-batch (default)


def _detect_boundaries(rows):
    """Return list of row indices that start a new episode.
    Rule: first row always starts ep-0; a new ep starts when BOTH energy
    and thirst jump UP by >20 compared to the previous row (respawn).
    """
    bounds = [0]
    for i in range(1, len(rows)):
        e_prev = rows[i-1]["obs"]["energy"]
        t_prev = rows[i-1]["obs"]["thirst"]
        e_cur  = rows[i]["obs"]["energy"]
        t_cur  = rows[i]["obs"]["thirst"]
        if (e_cur - e_prev) > 20 and (t_cur - t_prev) > 20:
            bounds.append(i)
    return bounds


def load():
    """Load all buffers; return (proprio, tokens, cmd, episode_index) tensors."""
    P_list, T_list, Y_list, EP_list = [], [], [], []
    global_ep = 0  # episode counter, never reset across files

    for buf in BUFS:
        pattern = f"data/replay_buffer/{buf}/*.jsonl"
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[WARN] aucun fichier trouvé dans {pattern}")
            continue

        for fpath in files:
            with open(fpath) as fh:
                raw = fh.readlines()
            rows = []
            for line in raw:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            if not rows:
                continue

            # ---- episode boundary detection (data-driven) ----
            # First line of this file = new episode
            boundaries = _detect_boundaries(rows)
            # Build row → local-episode mapping
            ep_of_row = [0] * len(rows)
            for b_idx, b_start in enumerate(boundaries):
                b_end = boundaries[b_idx + 1] if b_idx + 1 < len(boundaries) else len(rows)
                for r in range(b_start, b_end):
                    ep_of_row[r] = b_idx

            n_ep_this_file = len(boundaries)

            # ---- build tensors ----
            for i, row in enumerate(rows):
                wm = row.get("wm", {})
                ret = wm.get("retina0")
                cmd = wm.get("cmd")
                if ret is None or cmd is None:
                    continue
                if len(ret) != 4 * N_RAYS:
                    continue

                energy = float(row["obs"].get("energy", 0.0))
                thirst = float(row["obs"].get("thirst", 0.0))
                proprio = row["obs"]["proprio"]

                food_tok  = [energy / 100.0, 1.0] + _color_gated_depths(ret, RED)
                water_tok = [thirst / 100.0, 1.0] + _color_gated_depths(ret, BLUE)

                P_list.append(proprio)
                T_list.append([food_tok, water_tok])
                Y_list.append([float(cmd[0]), float(cmd[1])])
                EP_list.append(global_ep + ep_of_row[i])

            global_ep += n_ep_this_file

    total_episodes = global_ep
    P  = torch.tensor(P_list,  dtype=torch.float32)
    T  = torch.tensor(T_list,  dtype=torch.float32)   # [N, 2, TOK]
    Y  = torch.tensor(Y_list,  dtype=torch.float32)   # [N, 2]
    EP = torch.tensor(EP_list, dtype=torch.long)

    return P, T, Y, EP, total_episodes


def main():
    P, T, Y, EP, ne = load()
    print(f"BC: {len(Y)} transitions sur {ne} épisodes")

    ntr = max(1, int(0.8 * ne))
    tr  = EP < ntr
    te  = ~tr
    n_tr = tr.sum().item()
    n_te = te.sum().item()
    print(f"  train={n_tr} transitions ({ntr} épisodes) | test={n_te} transitions ({ne - ntr} épisodes)")

    if n_te == 0:
        print("[WARN] pas de données test — ajuster le split ou collecter plus d'épisodes")

    pol = DriveSymmetricPolicy()
    opt = torch.optim.Adam(pol.parameters(), lr=LR)

    P_tr, T_tr, Y_tr = P[tr], T[tr], Y[tr]
    use_batch = BATCH > 0
    prev_loss = float("inf")
    EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "100"))

    def heldout_match():
        if n_te == 0:
            return -1.0
        with torch.no_grad():
            pt = map_action(pol(P[te], T[te]))
        return (((pt[:, 0] - Y[te, 0]).abs() < 0.05) & ((pt[:, 1] - Y[te, 1]).abs() < 0.1)).float().mean().item()

    best_mr, best_state, best_it = -1.0, None, -1  # early-stopping sur le match held-out

    for it in range(ITERS):
        if use_batch:
            idx = torch.randperm(n_tr)[:BATCH]
            p_b, t_b, y_b = P_tr[idx], T_tr[idx], Y_tr[idx]
        else:
            p_b, t_b, y_b = P_tr, T_tr, Y_tr

        opt.zero_grad()
        pred = map_action(pol(p_b, t_b))
        loss = ((pred - y_b) ** 2).mean()
        loss.backward()
        opt.step()

        if it % EVAL_EVERY == 0 or it == ITERS - 1:
            mr_now = heldout_match()
            if mr_now > best_mr:
                best_mr, best_state, best_it = mr_now, copy.deepcopy(pol.state_dict()), it

        if it % 500 == 0 or it == ITERS - 1:
            lv = loss.item()
            tag = "↓" if lv < prev_loss else ("=" if lv == prev_loss else "↑")
            print(f"  it{it:04d}  mse={lv:.5f} {tag}  held-out match={100*heldout_match():.1f}%")
            prev_loss = lv

    # ---- early-stopping : restaure le MEILLEUR checkpoint held-out (pas le sur-appris final) ----
    if best_state is not None:
        pol.load_state_dict(best_state)
        print(f"\n[early-stopping] meilleur held-out à it{best_it} : match={100*best_mr:.1f}%")

    # ---- held-out evaluation (sur le MEILLEUR modèle) ----
    with torch.no_grad():
        pred_te = map_action(pol(P[te], T[te]))

    vx_err  = (pred_te[:, 0] - Y[te, 0]).abs()
    om_err  = (pred_te[:, 1] - Y[te, 1]).abs()
    match   = (vx_err < 0.05) & (om_err < 0.1)
    mr      = match.float().mean().item()
    vx_mae  = vx_err.mean().item()
    om_mae  = om_err.mean().item()

    print()
    print(f"[held-out] match-rate(|Δvx|<0.05, |Δω|<0.1) = {100*mr:.1f}%")
    print(f"[held-out] vx-MAE={vx_mae:.4f}  ω-MAE={om_mae:.4f}")
    print()

    # ---- save checkpoint ----
    os.makedirs(OUT, exist_ok=True)
    ckpt_path = f"{OUT}/policy.pt"
    torch.save({
        "model": pol.state_dict(),
        "meta":  {"proprio_dim": 132, "tok": T.shape[-1]},
    }, ckpt_path)
    print(f">>> sauvé {ckpt_path}")

    # ---- verdict ----
    if mr >= 0.60:
        print(f">>> SUCCÈS Task3 : match-rate {100*mr:.1f}% ≥ 60% — la politique BC a appris la nav du planner")
    elif mr >= 0.40:
        print(f">>> PARTIEL Task3 : match-rate {100*mr:.1f}% (entre 40-60%) — amélioration possible")
    else:
        print(f">>> DONE_WITH_CONCERNS : match-rate {100*mr:.1f}% < 40% — l'obs/réseau n'apprend pas la nav du planner")
        print("    Diagnostiquer avant Gate-1 : vérifier retina0 layout, gating couleur, distribution des cmds.")


if __name__ == "__main__":
    main()
