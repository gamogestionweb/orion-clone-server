#!/bin/bash
# ===========================================
# Orion Clone Server — RunPod Startup Script
# ===========================================
# Este script se ejecuta al arrancar el pod.
# Instala dependencias y lanza el servidor.
#
# En RunPod:
#   Docker Image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
#   Docker Command: bash -c "apt-get update && apt-get install -y git wget ffmpeg libsndfile1 libgl1-mesa-glx libglib2.0-0 && cd /workspace && git clone https://github.com/TU_USUARIO/orion-clone-server.git server 2>/dev/null; cd /workspace/server && pip install -r requirements.txt && python -m uvicorn app.main:app --host 0.0.0.0 --port 8765 --ws-max-size 16777216"

echo "=========================================="
echo "  ORION CLONE SERVER — RunPod Init"
echo "=========================================="

set -e

# Instalar dependencias del sistema si no están
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1 || true

cd /workspace

# Si no existe el servidor, clonar o copiar
if [ ! -d "server" ]; then
    echo "[1/3] Preparando código del servidor..."
    mkdir -p server
fi

cd server

# Instalar dependencias Python
echo "[2/3] Instalando dependencias Python..."
pip install --quiet --no-cache-dir \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    websockets>=12.0 \
    python-multipart>=0.0.6 \
    faster-whisper>=0.10.0 \
    TTS>=0.22.0 \
    numpy>=1.24.0 \
    opencv-python-headless>=4.8.0 \
    Pillow>=10.0.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    openai>=1.0.0 \
    anthropic>=0.20.0 \
    aiofiles>=23.0.0

echo "[3/3] Iniciando servidor..."
echo "Puerto: 8765"
echo "Device: $(python -c 'import torch; print("GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"

# Lanzar servidor
python -m uvicorn app.main:app --host 0.0.0.0 --port 8765 --ws-max-size 16777216
