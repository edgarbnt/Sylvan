"""Torch device selection helpers with robust ROCm fallback."""

from __future__ import annotations

import os

import torch

_GPU_SMOKE_TEST_CACHE: tuple[bool, str] | None = None


def _gpu_smoke_test() -> tuple[bool, str]:
    global _GPU_SMOKE_TEST_CACHE
    if _GPU_SMOKE_TEST_CACHE is not None:
        return _GPU_SMOKE_TEST_CACHE
    try:
        x = torch.randn((64, 64), device="cuda")
        y = torch.randn((64, 64), device="cuda")
        z = x @ y
        _ = float(z.mean().item())
        _GPU_SMOKE_TEST_CACHE = (True, "GPU kernel smoke test passed")
    except Exception as exc:  # pragma: no cover - hardware dependent path
        _GPU_SMOKE_TEST_CACHE = (False, f"GPU smoke test failed: {exc}")
    return _GPU_SMOKE_TEST_CACHE


def _resolve_torch_device_raw() -> tuple[torch.device, str]:
    requested = os.environ.get("SYLVAN_TORCH_DEVICE", "auto").strip().lower()

    if requested not in {"auto", "cpu", "cuda", "force_cuda"}:
        requested = "auto"

    if requested == "cpu":
        return torch.device("cpu"), "Using CPU (SYLVAN_TORCH_DEVICE=cpu)"

    if requested == "force_cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "Using GPU (SYLVAN_TORCH_DEVICE=force_cuda, smoke test bypassed)"
        return (
            torch.device("cpu"),
            "force_cuda requested but CUDA/ROCm unavailable. Falling back to CPU.",
        )

    if not torch.cuda.is_available():
        if requested == "cuda":
            return (
                torch.device("cpu"),
                "CUDA/ROCm requested but unavailable. Falling back to CPU.",
            )
        return torch.device("cpu"), "CUDA/ROCm unavailable. Using CPU."

    ok, reason = _gpu_smoke_test()
    if ok:
        return torch.device("cuda"), f"Using GPU ({reason})"

    if requested == "cuda":
        return (
            torch.device("cpu"),
            "CUDA/ROCm requested but unstable. Falling back to CPU. %s" % reason,
        )
    return torch.device("cpu"), "GPU unstable. Using CPU. %s" % reason


def resolve_torch_device() -> tuple[torch.device, str]:
    device, reason = _resolve_torch_device_raw()
    if device.type == "cpu":
        # Optimise le nombre de threads CPU pour éviter les surcoûts
        # de synchronisation sur les petits modèles de Sylvan.
        # Idempotent: safe to call from multiple trainers in the same process.
        try:
            torch.set_num_threads(4)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    return device, reason
