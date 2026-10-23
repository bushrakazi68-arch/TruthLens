# TruthLens — single-container image for Hugging Face Spaces.
#
# Spaces expose exactly one port (7860), so both processes run here:
#   - FastAPI  on 127.0.0.1:8000  (internal only)
#   - Express  on 0.0.0.0:7860    (public; proxies /api/* to FastAPI)
#
# The Express adapter already reads BACKEND_URL and already serves the built
# client when NODE_ENV=production, so no application code changes are needed.

FROM node:20-bookworm-slim

# Python 3 plus the build tools chromadb needs for its native wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. UID 1000 is already taken by the base image's "node"
# user, so this one uses 1001; files must be writable by it.
RUN useradd -m -u 1001 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# ── Python dependencies ──────────────────────────────────────
# By default only the LIGHTWEIGHT set is installed (requirements-base.txt, no
# PyTorch). PyTorch adds ~225 MB of resident memory, and free hosting tiers
# (Render: 512 MB) cannot hold it alongside the app. The spread predictor falls
# back to plain arithmetic and returns identical results.
#
# On a host with ~1 GB+ of RAM, build the full image with PyTorch Geometric:
#     docker build --build-arg WITH_TORCH=1 .
ARG WITH_TORCH=0
COPY backend/requirements-base.txt /app/backend/requirements-base.txt
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/backend/requirements-base.txt
# CPU-only torch wheel (~200 MB instead of ~2.5 GB for CUDA); inference runs
# remotely on Groq, so a GPU build would be pointless.
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
      && pip install --no-cache-dir torch-geometric==2.8.0.post1 ; \
    fi

# ── Node dependencies and client build ───────────────────────
COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY . /app

# Vite build + esbuild bundle of the Express adapter (npm run build).
RUN npm run build

# ── Runtime layout ───────────────────────────────────────────
# /data is where Spaces mounts persistent storage when it is enabled. Without
# it the directory is simply container-local and the corpus is re-seeded on
# each cold start by start.sh.
ENV CHROMA_PATH=/data/chroma_db \
    BACKEND_URL=http://127.0.0.1:8000 \
    NODE_ENV=production \
    PORT=7860

RUN mkdir -p /data/chroma_db && chown -R user:user /data /app

USER user
EXPOSE 7860

CMD ["bash", "/app/start.sh"]
