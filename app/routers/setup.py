import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/setup", tags=["setup"])

# Referencias a los modelos (se inyectan desde main.py)
_tts = None
_lipsync = None
_storage_path = "./storage"


def init_router(tts, lipsync, storage_path: str):
    global _tts, _lipsync, _storage_path
    _tts = tts
    _lipsync = lipsync
    _storage_path = storage_path


# ==========================================
# RESPONSE MODELS
# ==========================================

class SetupResponse(BaseModel):
    status: str
    user_id: str
    message: str = ""


class SetupStatus(BaseModel):
    user_id: str
    has_voice: bool
    has_face: bool
    voice_duration_seconds: int = 0


# ==========================================
# ENDPOINTS
# ==========================================

@router.post("/voice", response_model=SetupResponse)
async def upload_voice_sample(
    user_id: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """
    Sube muestra de voz y crea speaker embedding para clonación.
    Acepta WAV PCM 16-bit mono a 44.1kHz (formato de VoiceEssenceEngine).
    """
    if _tts is None:
        raise HTTPException(500, "TTS no inicializado")

    # Crear directorio del usuario
    user_dir = Path(_storage_path) / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    # Guardar archivo de audio
    audio_path = user_dir / "voice_sample.wav"
    try:
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        file_size = audio_path.stat().st_size
        # Estimar duración (44100Hz, 16-bit mono = 88200 bytes/segundo)
        duration = file_size // 88200

        logger.info(f"Voz recibida: {user_id}, {file_size} bytes, ~{duration}s")

        # Crear speaker embedding
        success = _tts.create_speaker_embedding(user_id, [str(audio_path)])

        if success:
            return SetupResponse(
                status="ok",
                user_id=user_id,
                message=f"Voz clonada exitosamente ({duration}s de audio)"
            )
        else:
            raise HTTPException(500, "Error creando embedding de voz")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando voz de {user_id}: {e}")
        raise HTTPException(500, f"Error procesando audio: {str(e)}")


@router.post("/video", response_model=SetupResponse)
async def upload_face_video(
    user_id: str = Form(...),
    video_file: UploadFile = File(...),
):
    """
    Sube video del rostro (5-10s) para animación con video loop + lip-sync.
    Acepta MP4, WEBM, o MOV.
    """
    if _lipsync is None:
        raise HTTPException(500, "Lip-sync no inicializado")

    user_dir = Path(_storage_path) / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    ext = ".mp4"
    if video_file.content_type:
        if "webm" in video_file.content_type:
            ext = ".webm"
        elif "mov" in video_file.content_type or "quicktime" in video_file.content_type:
            ext = ".mov"

    video_path = user_dir / f"face_video{ext}"
    try:
        with open(video_path, "wb") as f:
            content = await video_file.read()
            f.write(content)

        logger.info(f"Video recibido: {user_id}, {len(content)} bytes")

        success = _lipsync.preprocess_video(user_id, str(video_path))

        if success:
            return SetupResponse(
                status="ok",
                user_id=user_id,
                message="Video procesado exitosamente"
            )
        else:
            return SetupResponse(
                status="warning",
                user_id=user_id,
                message="No se detectó rostro en el video. Intenta de nuevo mirando a cámara."
            )

    except Exception as e:
        logger.error(f"Error procesando video de {user_id}: {e}")
        raise HTTPException(500, f"Error procesando video: {str(e)}")


@router.post("/face", response_model=SetupResponse)
async def upload_face_photo(
    user_id: str = Form(...),
    image_file: UploadFile = File(...),
):
    """
    Sube foto del rostro y preprocesa para lip-sync.
    Acepta JPEG o PNG.
    """
    if _lipsync is None:
        raise HTTPException(500, "Lip-sync no inicializado")

    user_dir = Path(_storage_path) / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    # Determinar extensión
    ext = ".jpg"
    if image_file.content_type and "png" in image_file.content_type:
        ext = ".png"

    image_path = user_dir / f"face_original{ext}"
    try:
        with open(image_path, "wb") as f:
            content = await image_file.read()
            f.write(content)

        logger.info(f"Foto recibida: {user_id}, {len(content)} bytes")

        # Preprocesar cara
        success = _lipsync.preprocess_face(user_id, str(image_path))

        if success:
            return SetupResponse(
                status="ok",
                user_id=user_id,
                message="Rostro procesado exitosamente"
            )
        else:
            return SetupResponse(
                status="warning",
                user_id=user_id,
                message="No se detectó rostro claro. Se usará la imagen completa."
            )

    except Exception as e:
        logger.error(f"Error procesando foto de {user_id}: {e}")
        raise HTTPException(500, f"Error procesando imagen: {str(e)}")


@router.get("/status/{user_id}", response_model=SetupStatus)
async def get_setup_status(user_id: str):
    """Verifica estado de configuración del usuario."""
    has_voice = _tts.has_speaker(user_id) if _tts else False
    has_face = _lipsync.has_face(user_id) if _lipsync else False

    # Calcular duración de audio si existe
    duration = 0
    voice_path = Path(_storage_path) / user_id / "voice_sample.wav"
    if voice_path.exists():
        duration = int(voice_path.stat().st_size / 88200)

    return SetupStatus(
        user_id=user_id,
        has_voice=has_voice,
        has_face=has_face,
        voice_duration_seconds=duration,
    )


@router.delete("/{user_id}")
async def delete_user_data(user_id: str):
    """Elimina todos los datos del usuario."""
    # Limpiar caches de modelos
    if _tts:
        _tts.remove_speaker(user_id)
    if _lipsync:
        _lipsync.remove_face(user_id)

    # Eliminar directorio
    user_dir = Path(_storage_path) / user_id
    if user_dir.exists():
        shutil.rmtree(user_dir)
        logger.info(f"Datos eliminados: {user_id}")
        return {"status": "ok", "message": f"Datos de {user_id} eliminados"}

    return {"status": "ok", "message": "No había datos que eliminar"}
