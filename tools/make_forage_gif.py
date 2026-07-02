"""Generate a clean, paper-quality top-down foraging GIF from logged replay data.

Reads one episode of `retina_eat_a` replay (positions already logged), reconstructs
world food positions from the ego-frame food vector + an integrated-yaw heading, and
renders an animated top-down view of the agent navigating to food and eating.

No Godot / no training: pure data -> matplotlib. Reusable.

Data per JSONL line (top-level keys):
  wm.torso0    = agent world position [x, y, z]  (ground plane = x, z)
  wm.food_rel0 = food in agent EGO frame [x_lat, y_fwd, present_flag=1]  (plane = [0], [1])
  wm.ate       = 1/40 on the step food is eaten
  obs.energy   = 0..100
  obs.metrics.yaw_rate = body yaw rate (integrated for heading)

Heading note: the agent wobbles in a small area so step-to-step motion direction is
pure gait noise -> useless for yaw. Integrating yaw_rate (dt ~= -1/60, sign from a
least-squares fit to the ego food-bearing) recovers heading up to a per-segment offset.
Food world = agent_xz + R(heading) . food_ego ; validated ~constant between eats
(per-segment std ~0.2-0.7 m at a ~1 m food distance; see FOOD VALIDATION printout).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.patches import FancyArrow

# ----------------------------------------------------------------------------- config
REPO = Path(__file__).resolve().parents[1]
EPISODE = REPO / "godot/data/replay_buffer/retina_eat_a/episode_0025.jsonl"
OUT = REPO / "assets/forage.gif"
TRIM_END = 1050          # steps: keep the first 3 clean eats (691, 801, 1023)
DT = -0.0175             # yaw-rate integration step (fit; ~ -1/60)
TARGET_FRAMES = 200      # ~13 s at 15 fps
FPS = 15
TRAIL = 40               # trailing path length (steps)
FIGSIZE = 6.0            # inches (square)
DPI = 80

BG = "#0e1117"
FG = "#e6edf3"
TRAIL_C = "#38bdf8"
AGENT_C = "#f8fafc"
ARROW_C = "#fbbf24"
FOOD_C = "#ef4444"


def load(path: Path):
    rows = [json.loads(l) for l in path.open() if l.strip()]
    torso = np.array([r["wm"]["torso0"] for r in rows], dtype=float)
    frel = np.array([r["wm"]["food_rel0"] for r in rows], dtype=float)
    ate = np.array([r["wm"]["ate"] for r in rows], dtype=float)
    yaw_rate = np.array([r["obs"]["metrics"]["yaw_rate"] for r in rows], dtype=float)
    energy = np.array([r["obs"]["energy"] for r in rows], dtype=float)
    return torso, frel, ate, yaw_rate, energy


def reconstruct(torso, frel, ate, yaw_rate, n):
    """Return per-step world food, per-step heading, eat steps, food-validation stds."""
    xz = torso[:n, [0, 2]]
    ego_x = frel[:n, 0]
    ego_y = frel[:n, 1]
    head_raw = DT * np.cumsum(yaw_rate[:n])
    eats = [i for i, a in enumerate(ate[:n]) if a > 0.5]

    bounds = [0] + eats + [n - 1]
    heading = np.zeros(n)
    food = np.zeros((n, 2))
    stds = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = np.arange(a + 1, b + 1)
        if len(idx) == 0:
            idx = np.arange(a, b + 1)
        # choose the per-segment offset that makes the world food most constant
        best = (1e9, 0.0, np.zeros(2))
        for off in np.linspace(-math.pi, math.pi, 180):
            h = head_raw[idx] + off
            c, s = np.cos(h), np.sin(h)
            wx = xz[idx, 0] + c * ego_x[idx] - s * ego_y[idx]
            wy = xz[idx, 1] + s * ego_x[idx] + c * ego_y[idx]
            sd = math.hypot(wx.std(), wy.std())
            if sd < best[0]:
                best = (sd, off, np.array([np.median(wx), np.median(wy)]))
        sd, off, fpos = best
        stds.append(sd)
        # segment spans (a, b]; step a itself belongs to previous segment's food
        seg_slice = slice(a + 1, b + 1)
        heading[seg_slice] = head_raw[seg_slice] + off
        food[seg_slice] = fpos
    heading[0] = heading[1]
    food[0] = food[1]
    return xz, heading, food, eats, stds


def main() -> None:
    torso, frel, ate, yaw_rate, energy = load(EPISODE)
    n = min(TRIM_END, len(torso))
    xz, heading, food, eats, stds = reconstruct(torso, frel, ate, yaw_rate, n)
    energy = energy[:n]

    print(f"[episode] {EPISODE.name}  steps used = {n}  eats = {eats}")
    print("[FOOD VALIDATION] per-segment world-food std (m):",
          [round(s, 2) for s in stds],
          f" mean={np.mean(stds):.2f}  (food distance ~1.0-1.8 m)")

    # frame subsampling
    stride = max(1, n // TARGET_FRAMES)
    frames = list(range(0, n, stride))
    print(f"[frames] {len(frames)} frames  stride={stride}  ~{len(frames)/FPS:.1f}s @ {FPS}fps")

    # plot bounds (equal aspect) from path + food
    allpts = np.vstack([xz, food])
    lo = allpts.min(0) - 1.2
    hi = allpts.max(0) + 1.2
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    cx, cy = (lo + hi) / 2
    x0, x1 = cx - span / 2, cx + span / 2
    y0, y1 = cy - span / 2, cy + span / 2

    fig = plt.figure(figsize=(FIGSIZE, FIGSIZE), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.90])
    ax.set_facecolor(BG)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.5, 0.965, "Sylvan — foraging in a learned world-model",
             ha="center", va="top", color=FG, fontsize=12, weight="bold")
    step_txt = fig.text(0.5, 0.925, "", ha="center", va="top", color="#94a3b8", fontsize=9)

    # energy bar (top-left inside frame)
    bar_bg = plt.Rectangle((0.045, 0.885), 0.24, 0.028, transform=fig.transFigure,
                           facecolor="#1f2733", edgecolor="#334155", lw=0.8, zorder=5)
    bar_fg = plt.Rectangle((0.045, 0.885), 0.0, 0.028, transform=fig.transFigure,
                           facecolor="#22c55e", edgecolor="none", zorder=6)
    fig.patches.extend([bar_bg, bar_fg])
    en_txt = fig.text(0.045, 0.862, "", color="#94a3b8", fontsize=8)

    def eat_flash(step: int) -> float:
        """0..1 flash intensity if an eat happened just before this step."""
        best = 0.0
        for e in eats:
            d = step - e
            if 0 <= d <= 14:
                best = max(best, 1.0 - d / 14.0)
        return best

    writer = PillowWriter(fps=FPS)
    with writer.saving(fig, str(OUT), dpi=DPI):
        for step in frames:
            ax.cla()
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
            ax.axis("off")

            # fading trail
            t0 = max(0, step - TRAIL)
            seg = xz[t0:step + 1]
            for k in range(1, len(seg)):
                a = (k / len(seg)) ** 1.5
                ax.plot(seg[k - 1:k + 1, 0], seg[k - 1:k + 1, 1],
                        color=TRAIL_C, alpha=0.65 * a, lw=2.0, solid_capstyle="round", zorder=2)

            # food star (flash on eat)
            fl = eat_flash(step)
            fx, fy = food[step]
            ax.scatter([fx], [fy], marker="*", s=340 + 900 * fl,
                       c=FOOD_C, edgecolors="white", linewidths=0.6 + fl,
                       alpha=0.9 + 0.1 * fl, zorder=4)
            if fl > 0.05:
                ax.scatter([fx], [fy], marker="o", s=1400 * fl, facecolors="none",
                           edgecolors=FOOD_C, linewidths=2.0, alpha=fl, zorder=3)

            # agent dot + heading arrow
            ax_, ay_ = xz[step]
            h = heading[step]
            ax.add_patch(FancyArrow(ax_, ay_, 0.9 * math.cos(h), 0.9 * math.sin(h),
                                    width=0.06, head_width=0.32, head_length=0.32,
                                    length_includes_head=True, color=ARROW_C, zorder=5))
            ax.scatter([ax_], [ay_], s=130, c=AGENT_C, edgecolors="#0e1117",
                       linewidths=1.2, zorder=6)

            # energy bar update
            e = float(energy[step])
            frac = np.clip(e / 100.0, 0, 1)
            bar_fg.set_width(0.24 * frac)
            r = int(34 + (239 - 34) * (1 - frac))
            g = int(197 + (68 - 197) * (1 - frac))
            b = int(94 + (68 - 94) * (1 - frac))
            bar_fg.set_facecolor((r / 255, g / 255, b / 255))
            en_txt.set_text(f"energy  {e:4.0f}")
            step_txt.set_text(f"step {step:4d} / {n}")

            writer.grab_frame()
    plt.close(fig)

    mb = OUT.stat().st_size / 1e6
    print(f"[out] {OUT}  size={mb:.2f} MB  {int(FIGSIZE*DPI)}x{int(FIGSIZE*DPI)}px  "
          f"frames={len(frames)}  dur={len(frames)/FPS:.1f}s @ {FPS}fps")


if __name__ == "__main__":
    main()
