#!/bin/bash
set -e

echo "=========================================="
echo "  ORION CLONE SERVER — RunPod Init"
echo "=========================================="

# Instalar dependencias del sistema
apt-get update -qq
apt-get install -y -qq git ffmpeg libsndfile1 libgl1-mesa-glx libglib2.0-0

# Clonar o actualizar repo
cd /workspace
if [ -d "server" ]; then
    cd server
    git pull
else
    git clone https://github.com/gamogestionweb/orion-clone-server.git server
    cd server
fi

# Instalar dependencias Python
pip install --quiet -r requirements.txt

# Lanzar servidor
echo "=========================================="
echo "  SERVIDOR LISTO — Puerto 8765"
echo "=========================================="
python -m uvicorn app.main:app --host 0.0.0.0 --port 8765 --ws-max-size 16777216
