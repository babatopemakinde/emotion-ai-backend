# =============================================================================
#  Multimodal Emotion AI v7.2-azure — Dockerfile
#
#  Build:  docker build -t emotionai:latest .
#  Run:    docker run -p 8000:8000 emotionai:latest
#
#  Azure deployment paths
#  ─────────────────────────────────────────────────────────────────────────
#  A) Azure App Service (Web App for Containers)
#       Push image to Azure Container Registry (ACR), point App Service at it.
#       Azure injects PORT env-var automatically — uvicorn reads it.
#
#  B) Azure Container Apps
#       Same image; set EMOTIONAI_WORKSPACE to an Azure File Share mount
#       path for persistent model/data storage across replicas.
#
#  Model weights (vision_model.keras, audio_model.keras, joint_infer_model.keras)
#  ─────────────────────────────────────────────────────────────────────────
#  Option 1 (baked-in):  COPY them into /app/emotionai_workspace/models/final/
#                         before building — simplest for single-replica deploys.
#  Option 2 (mount):     Leave them out of the image; mount an Azure File Share
#                         at /mnt/emotionai and set:
#                         ENV EMOTIONAI_WORKSPACE=/mnt/emotionai/workspace
#  Option 3 (download):  Set AZURE_STORAGE_CONNECTION_STRING and
#                         AZURE_BLOB_CONTAINER in App Service Application
#                         Settings — ModelRegistry downloads .keras files from
#                         Blob Storage automatically on startup (default).
# =============================================================================

# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim chosen for broad TF wheel availability and smaller size.
# Switch to 3.13-slim once TF 2.20 slim wheels are confirmed stable on ACR runners.
# azure-storage-blob and azure-identity are pure-Python wheels — no extra OS
# packages required beyond what is already installed below.
FROM python:3.11-slim

# ── Build-time metadata ───────────────────────────────────────────────────────
LABEL maintainer="CIoTH Research Group — University of Greater Manchester"
LABEL version="7.2.0-azure"
LABEL description="Multimodal Emotion AI — ASD Support System (FastAPI backend)"

# ── OS dependencies ───────────────────────────────────────────────────────────
# libgl1 / libglib2.0-0 : OpenCV headless runtime
# libsndfile1           : soundfile / librosa audio I/O
# ffmpeg                : audioread backend for MP3 decoding in librosa
# curl                  : Docker health-check probe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        ffmpeg \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (layer-cached separately from source) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY app.py .

# ── Optional: bake model weights into the image (Option 1) ───────────────────
# Uncomment and adjust path if your weights are in the repo / build context.
# COPY models/ /app/emotionai_workspace/models/

# ── Writable workspace (ephemeral by default; override via volume/mount) ──────
ENV EMOTIONAI_WORKSPACE=/tmp/emotionai_workspace
RUN mkdir -p /tmp/emotionai_workspace

# ── Non-root user for Azure security best-practice ───────────────────────────
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser \
    && chown -R appuser:appgroup /app /tmp/emotionai_workspace
USER appuser

# ── Expose port (Azure App Service uses PORT env-var; default 8000) ───────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# $PORT is injected by Azure App Service; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info"]
