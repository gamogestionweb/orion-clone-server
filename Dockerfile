# ===========================================
# Orion Clone Server — RunPod GPU Deployment
# ===========================================
# Deploy en RunPod: GPU Template → Custom Docker Image
# URL: https://runpod.io
#
# Uso:
#   1. Crear cuenta en RunPod
#   2. Crear GPU Pod (RTX 3090/4090 o A100 recomendado)
#   3. Template: Custom Docker Image
#   4. Image: tu-registry/orion-clone-server:latest
#   5. Env vars: LLM_API_KEY, LLM_PROVIDER, LLM_MODEL
#   6. Puerto: 8765
# ===========================================

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Instalar PyTorch con CUDA primero (capa cacheada)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Instalar resto de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar modelos de Whisper y XTTS al build (evita descarga en runtime)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')" || true
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')" || true

# Copiar código
COPY app/ ./app/

# Crear directorios
RUN mkdir -p storage models

EXPOSE 8765

# Variables de entorno por defecto
ENV HOST=0.0.0.0
ENV PORT=8765
ENV WHISPER_MODEL=medium
ENV STORAGE_PATH=./storage
ENV MAX_USERS=5

# Ejecutar servidor
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765", "--ws-max-size", "16777216"]
