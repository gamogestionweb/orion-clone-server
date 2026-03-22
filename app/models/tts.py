import logging
import os
import numpy as np
import torch
from pathlib import Path
from typing import Optional, AsyncGenerator

logger = logging.getLogger(__name__)


class XTTSCloner:
    """
    Clonación de voz con Coqui XTTS v2.
    Soporta síntesis multilingüe con voz clonada desde pocos segundos de audio.
    """

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.model_name = model_name
        self.tts = None
        # Cache de embeddings por usuario
        self._speaker_cache: dict[str, dict] = {}
        logger.info(f"XTTSCloner configurado: {model_name}")

    def load(self):
        """Carga el modelo XTTS v2."""
        from TTS.api import TTS
        logger.info(f"Cargando XTTS v2...")
        self.tts = TTS(self.model_name, gpu=torch.cuda.is_available())
        logger.info("XTTS v2 cargado.")

    def create_speaker_embedding(self, user_id: str, audio_paths: list[str]) -> bool:
        """
        Crea y cachea el embedding de voz del usuario.

        Args:
            user_id: ID del usuario
            audio_paths: Lista de paths a archivos WAV con muestras de voz

        Returns:
            True si se creó exitosamente.
        """
        if self.tts is None:
            raise RuntimeError("XTTS no está cargado.")

        try:
            # Verificar que los archivos existen
            valid_paths = [p for p in audio_paths if os.path.exists(p)]
            if not valid_paths:
                logger.error(f"No hay archivos de audio válidos para {user_id}")
                return False

            # XTTS v2 puede computar speaker embeddings desde archivos de referencia
            # Usamos el método interno para obtener gpt_cond_latent y speaker_embedding
            model = self.tts.synthesizer.tts_model

            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=valid_paths,
                gpt_cond_len=30,     # Usar hasta 30s de contexto
                gpt_cond_chunk_len=4,
                max_ref_length=60,   # Máximo 60s de referencia
            )

            self._speaker_cache[user_id] = {
                "gpt_cond_latent": gpt_cond_latent,
                "speaker_embedding": speaker_embedding,
            }

            # Guardar embeddings en disco para persistencia
            cache_dir = Path(f"storage/{user_id}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self._speaker_cache[user_id], cache_dir / "voice_embedding.pt")

            logger.info(f"Speaker embedding creado para {user_id} ({len(valid_paths)} archivos)")
            return True

        except Exception as e:
            logger.error(f"Error creando speaker embedding para {user_id}: {e}")
            return False

    def load_speaker_embedding(self, user_id: str) -> bool:
        """Carga embedding guardado en disco."""
        if user_id in self._speaker_cache:
            return True

        cache_path = Path(f"storage/{user_id}/voice_embedding.pt")
        if not cache_path.exists():
            return False

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            data = torch.load(cache_path, map_location=device, weights_only=False)
            self._speaker_cache[user_id] = data
            logger.info(f"Speaker embedding cargado desde disco: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error cargando embedding de {user_id}: {e}")
            return False

    def has_speaker(self, user_id: str) -> bool:
        """Verifica si existe embedding para el usuario."""
        if user_id in self._speaker_cache:
            return True
        return Path(f"storage/{user_id}/voice_embedding.pt").exists()

    async def synthesize(
        self,
        user_id: str,
        text: str,
        language: str = "es",
    ) -> Optional[np.ndarray]:
        """
        Sintetiza texto con la voz clonada del usuario.

        Args:
            user_id: ID del usuario
            text: Texto a sintetizar
            language: Código de idioma

        Returns:
            Audio float32 numpy array a 24kHz (sample rate nativo de XTTS),
            o None si hay error.
        """
        if self.tts is None:
            raise RuntimeError("XTTS no está cargado.")

        # Cargar embedding si no está en cache
        if not self.load_speaker_embedding(user_id):
            logger.error(f"No hay embedding de voz para {user_id}")
            return None

        cache = self._speaker_cache[user_id]

        # Mapear códigos de idioma a formato XTTS
        lang_map = {
            "ES": "es", "EN": "en", "FR": "fr", "PT": "pt",
            "DE": "de", "RU": "ru", "CN": "zh-cn", "FA": "fa",
        }
        xtts_lang = lang_map.get(language.upper(), "es")

        try:
            model = self.tts.synthesizer.tts_model

            # Generar audio con voz clonada
            result = model.inference(
                text=text,
                language=xtts_lang,
                gpt_cond_latent=cache["gpt_cond_latent"],
                speaker_embedding=cache["speaker_embedding"],
                temperature=0.65,
                length_penalty=1.0,
                repetition_penalty=5.0,
                top_k=50,
                top_p=0.85,
                speed=1.0,
            )

            # result["wav"] es un tensor torch
            audio = result["wav"]
            if isinstance(audio, torch.Tensor):
                audio = audio.squeeze().cpu().numpy()

            logger.debug(f"TTS sintetizado: {len(text)} chars → {len(audio)/24000:.1f}s audio")
            return audio

        except Exception as e:
            logger.error(f"Error sintetizando para {user_id}: {e}")
            return None

    def get_output_sample_rate(self) -> int:
        """XTTS v2 genera audio a 24kHz."""
        return 24000

    def remove_speaker(self, user_id: str):
        """Elimina embedding del usuario."""
        self._speaker_cache.pop(user_id, None)
        cache_path = Path(f"storage/{user_id}/voice_embedding.pt")
        if cache_path.exists():
            cache_path.unlink()

    def is_loaded(self) -> bool:
        return self.tts is not None
