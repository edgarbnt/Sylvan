# ─── Base ─────────────────────────────────────────────────────────────────────
# ROCm 5.7 + PyTorch 2.0.1 pre-installed (AMD GPU support)
FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1

# ─── Environment ──────────────────────────────────────────────────────────────
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    # HSA_ENABLE_SDMA=0 : désactive le System DMA engine ROCm.
    # Correctif connu pour HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (0x29)
    # sur gfx1030 (RX 6000) avec ROCm 5.7.
    HSA_ENABLE_SDMA=0 \
    # GPU_MAX_HW_QUEUES=1 : limite les queues GPU concurrentes.
    # Évite les conflits entre PyTorch et Godot (Vulkan) sur le même GPU.
    GPU_MAX_HW_QUEUES=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ─── Dependencies ─────────────────────────────────────────────────────────────
# Copy only pyproject.toml first to leverage Docker layer cache:
# if dependencies haven't changed, this layer won't be rebuilt.
WORKDIR /workspace
COPY python/pyproject.toml python/pyproject.toml

# NE PAS réinstaller torch : la base image contient déjà PyTorch 2.0.1 (ROCm).
# Réinstaller depuis PyPI téléchargerait la version CUDA (~2.5 Go inutilisable sur AMD).
# On installe uniquement les dépendances légères manquantes.
RUN pip install --upgrade pip && \
    pip install \
        "numpy>=1.21,<2.0" \
        "onnx>=1.15.0"

# ─── Source code ──────────────────────────────────────────────────────────────
# Copy the full python package after deps are cached
COPY python/ python/

# Install the sylvan package in editable mode
RUN pip install --no-deps -e python/

# ─── Entry point ──────────────────────────────────────────────────────────────
# ENTRYPOINT = le script fixe, CMD = les arguments par défaut (overridables via docker run)
ENTRYPOINT ["run-sylvan"]
CMD ["--num-cycles", "1"]
