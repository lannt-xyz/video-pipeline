# ---------------------------------------------------------------------------
# Stage 1 — dependency builder
# ---------------------------------------------------------------------------
# CUDA 12.6 + cuDNN runtime on Ubuntu 24.04 (ships Python 3.12 out of the box)
# NVENC is provided by the host NVIDIA driver through nvidia-container-toolkit;
# FFmpeg from apt already has NVENC headers baked in.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System-level deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        build-essential \
        curl \
        git \
        # Audio processing libs required by piper-tts / soundfile / kokoro
        libsndfile1 \
        libgomp1 \
        # espeak-ng used by kokoro TTS
        espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Pull uv binary from its official image (no PATH pollution)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Create isolated venv
RUN uv venv --python python3.12 .venv
ENV PATH="/app/.venv/bin:$PATH"

# ---------------------------------------------------------------------------
# Pre-install PyTorch with CUDA wheels BEFORE uv sync.
# If uv later resolves torch as a transitive dep it will see the CUDA version
# already installed and skip the PyPI CPU-only wheel.
# cu124 wheels run on CUDA 12.4+ drivers (covers RTX 5060 Ti / CUDA 12.6 driver).
# ---------------------------------------------------------------------------
RUN uv pip install \
        "torch>=2.4" \
        "torchvision" \
        "torchaudio" \
        --index-url https://download.pytorch.org/whl/cu124

# Install remaining project deps (torch already present → uv won't reinstall it)
COPY pyproject.toml .
RUN uv sync --no-dev

# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Tell Python to use the venv copied from the builder stage
    PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

# Runtime system deps only (no build-essential / headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        # Shared library required by torchcodec (f5-tts dep) custom_ops for FFmpeg 6
        libpython3.12-dev \
        # FFmpeg with NVENC support (nvenc codec linked to host driver at runtime)
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        espeak-ng \
        # Required by httpx / TLS
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed venv from builder (avoids re-downloading packages)
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY . .

# ---------------------------------------------------------------------------
# Volumes expected at runtime (bind-mount in docker-compose):
#   /app/data          — episode state, audio, images, clips, output
#   /app/logs          — pipeline logs + debug dumps
#   /app/assets        — BGM music, reference voice (f5tts_ref_audio)
#   /app/models        — F5-TTS & Piper model checkpoints (large binaries)
#   /app/config/settings.yaml — editable at runtime without rebuilding
# ---------------------------------------------------------------------------

ENTRYPOINT ["python", "main.py"]
# Default: run full pipeline from episode 1.
# Override at `docker compose run pipeline --episode 1 --from-phase llm`
