import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class WhisperSTT:
    """
    Speech-to-Text con faster-whisper (CTranslate2).
    ~4x más rápido que openai-whisper original.
    """

    def __init__(self, model_size: str = "medium", device: str = "cuda", compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device if device != "cpu" else "cpu"
        self.model = None
        self._compute_type = compute_type if device != "cpu" else "int8"
        logger.info(f"WhisperSTT configurado: model={model_size}, device={self.device}, compute={self._compute_type}")

    def load(self):
        """Carga el modelo. Llamar durante startup del servidor."""
        from faster_whisper import WhisperModel
        logger.info(f"Cargando Whisper {self.model_size}...")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self._compute_type,
        )
        logger.info("Whisper cargado.")

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio float32 a texto.

        Args:
            audio: numpy float32 array [-1.0, 1.0]
            sample_rate: sample rate del audio (debe ser 16kHz)
            language: código de idioma (ej: "es", "en"). None = autodetect.

        Returns:
            Texto transcrito.
        """
        if self.model is None:
            raise RuntimeError("Whisper no está cargado. Llama a load() primero.")

        if len(audio) == 0:
            return ""

        # faster-whisper espera float32 a 16kHz
        if sample_rate != 16000:
            from app.services.audio_utils import resample_audio
            audio = resample_audio(audio, sample_rate, 16000)

        # Mapear códigos de idioma
        lang_map = {
            "ES": "es", "EN": "en", "FR": "fr", "PT": "pt",
            "DE": "de", "RU": "ru", "CN": "zh", "FA": "fa",
        }
        whisper_lang = lang_map.get(language, language) if language else None

        segments, info = self.model.transcribe(
            audio,
            language=whisper_lang,
            beam_size=5,
            vad_filter=True,  # Filtrar silencio automáticamente
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )

        # Concatenar segmentos
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        result = " ".join(text_parts).strip()
        logger.debug(f"STT [{info.language}]: '{result[:100]}...' ({info.duration:.1f}s audio)")
        return result

    def is_loaded(self) -> bool:
        return self.model is not None
