# Task 4 Report — serve_mode1.py (Mode-1 deterministic server)

## Status: DONE

## What was forked / kept / replaced

**Forked from:** `python/scripts/serve_planner_command.py`

**KEPT (verbatim or structurally identical):**
- TCP server loop (`socketserver.ThreadingTCPServer` + `StreamRequestHandler`)
- Newline-delimited JSON protocol: receives payload, returns `{"action":[18], "command":[vx,ω]}`
- Frozen residual loading via `load_checkpoint` + `GaussianActorCritic` (same config)
- **Residual-obs construction** (serve_planner_command.py lines 360–362, copied verbatim):
  ```python
  vision = [float(vx), float(om)] + [0.0] * (VISION_DIM - 2)
  res_in = torch.tensor(proprio + vision, dtype=torch.float32).unsqueeze(0)
  action = self.residual.mean(res_in)[0]
  ```
  `VISION_DIM = 12` (same constant)
- `replan_every` cadence: command held K ticks, BC policy re-queried every K ticks
- Reset handling (stateless for Mode-1: resets ticks + cmd to default)
- Error fallback: broad `except` → safe response, server never crashes

**REPLACED:**
- WorldModel + CommandPlanner (wm_ckpt, plan(), cfg) → removed entirely
- `planner.plan(...)` → `build_tokens(payload)` + `_pol.act(...)` (DriveSymmetricPolicy)

## Residual checkpoint + obs path reused

- **Checkpoint:** `data/checkpoints/hexapod_v2/policy_best.pt`
  - Loaded via `load_checkpoint(residual_ckpt, self.residual)` (mirroring serve_planner_command.py line 119)
- **Obs construction:** serve_planner_command.py **lines 360–362** (within `predict_full`):
  ```python
  vision = [float(vx), float(om)] + [0.0] * (VISION_DIM - 2)
  res_in = torch.tensor(proprio + vision, dtype=torch.float32).unsqueeze(0)
  action = self.residual.mean(res_in)[0]
  ```
  `VISION_DIM = 12` defined at module level (line 86)

## BC Policy checkpoint

`data/checkpoints/mode1_bc/policy.pt`
Loaded via `torch.load(..., weights_only=False)` → `DriveSymmetricPolicy(proprio_dim=132).load_state_dict(...)`

## Smoke test output (verbatim)

```
Server PID: 26742
[serve-mode1] BC policy = policy.pt (proprio_dim=132, n_drives=?, epochs=?)
[serve-mode1] residual=policy_best.pt | replan_every=10
[serve-mode1] serving on 127.0.0.1:6099 — Ctrl-C to stop
---SMOKE TEST---
action len = 18
command = [0.6224035620689392, 0.44082406163215637]
vx in [0.55,0.75]: True  (vx=0.6224)
omega in [-0.6,0.6]: True  (omega=0.4408)
SMOKE TEST: PASS
Test exit code: 0
Server stopped.
```

## Files changed

- **Created:** `python/scripts/serve_mode1.py` (189 lines)
- **Commit:** `ac28ee2` — "Mode-1 Task4 : serveur deterministe (politique BC -> commande -> residu gele), contrat TCP inchange"
- **Branch:** `mode1-build`

## Concerns

Minor: the checkpoint meta fields `n_drives` and `epochs_trained` are absent from `mode1_bc/policy.pt` (logged as `?`). No functional impact — `proprio_dim=132` is present and sufficient for loading.

Default port is 6052 (overridable via `--port` or `PORT` env var, same pattern as serve_planner_command.py).
