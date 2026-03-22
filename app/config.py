import os
import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Auto-detecta si hay GPU disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detectada: {gpu_name}")
            return "cuda"
    except ImportError:
        pass
    logger.info("Sin GPU — modo CPU (más lento pero funcional)")
    return "cpu"


@dataclass
class ServerConfig:
    """Configuración central del servidor Orion Clone."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8765

    # Device (auto-detectado)
    device: str = "cpu"

    # Whisper STT
    whisper_model: str = "medium"
    whisper_compute_type: str = "float16"

    # XTTS v2 Voice Cloning
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    # Wav2Lip Face Animation
    wav2lip_checkpoint: str = "models/wav2lip_gan.pth"
    face_detect_model: str = "models/s3fd.pth"

    # LLM Proxy
    llm_provider: str = "openai"      # "openai" | "anthropic"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # Storage
    storage_path: str = "./storage"

    # Audio
    input_sample_rate: int = 16000
    output_sample_rate: int = 16000

    # Face Animation
    face_fps: int = 25
    face_resolution: Tuple[int, int] = (256, 256)
    jpeg_quality: int = 75

    # Limits
    max_concurrent_users: int = 5
    max_voice_duration_seconds: int = 300
    vad_silence_threshold_ms: int = 800

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Carga configuración desde variables de entorno con auto-detección."""
        device = detect_device()

        # Ajustar modelos según dispositivo
        if device == "cpu":
            # CPU: modelos más ligeros para que funcione
            whisper_model = os.getenv("WHISPER_MODEL", "base")  # base es rápido en CPU
            compute_type = "int8"  # Cuantizado para CPU
        else:
            # GPU: modelos completos
            whisper_model = os.getenv("WHISPER_MODEL", "medium")
            compute_type = "float16"

        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8765")),
            device=device,
            whisper_model=whisper_model,
            whisper_compute_type=compute_type,
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            storage_path=os.getenv("STORAGE_PATH", "./storage"),
            max_concurrent_users=int(os.getenv("MAX_USERS", "5")),
        )
