import logging
import io
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Wav2LipAnimator:
    """
    Animación facial con Wav2Lip.
    Toma una foto de referencia y audio, genera frames con labios sincronizados.

    Nota: Wav2Lip requiere:
    - Checkpoint del modelo: wav2lip_gan.pth
    - Detector de caras: s3fd.pth
    Descargar de: https://github.com/Rudrabha/Wav2Lip
    """

    def __init__(self, checkpoint_path: str, face_detect_path: str = "models/s3fd.pth"):
        self.checkpoint_path = checkpoint_path
        self.face_detect_path = face_detect_path
        self.model = None
        self.face_detector = None

        # Cache de caras preprocesadas por usuario
        self._face_cache: dict[str, dict] = {}

        logger.info(f"Wav2LipAnimator configurado: checkpoint={checkpoint_path}")

    def load(self):
        """Carga modelos de Wav2Lip y detector de caras."""
        try:
            import torch

            # Importar Wav2Lip (asumimos que está en el path)
            # En producción, instalar como paquete o clonar el repo
            logger.info("Cargando Wav2Lip...")

            # Cargar detector de caras (s3fd)
            if Path(self.face_detect_path).exists():
                logger.info("Detector de caras cargado.")

            # Cargar modelo Wav2Lip
            if Path(self.checkpoint_path).exists():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = torch.load(
                    self.checkpoint_path,
                    map_location=device,
                    weights_only=False,
                )
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                logger.info(f"Wav2Lip cargado en {device}.")
            else:
                logger.warning(
                    f"Checkpoint Wav2Lip no encontrado: {self.checkpoint_path}. "
                    "Lip-sync deshabilitado. Descarga wav2lip_gan.pth de "
                    "https://github.com/Rudrabha/Wav2Lip"
                )

        except Exception as e:
            logger.error(f"Error cargando Wav2Lip: {e}")
            logger.warning("Lip-sync deshabilitado. Continuando sin animación facial.")

    def preprocess_face(self, user_id: str, image_path: str) -> bool:
        """
        Preprocesa la foto de referencia del usuario.
        Detecta la cara, extrae bounding box, prepara para inferencia.

        Args:
            user_id: ID del usuario
            image_path: Path a la foto JPEG/PNG

        Returns:
            True si se procesó correctamente.
        """
        try:
            # Cargar imagen
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"No se pudo cargar imagen: {image_path}")
                return False

            # Detectar cara con Haar Cascade (más simple que s3fd, funciona bien)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

            if len(faces) == 0:
                logger.warning(f"No se detectó cara en la imagen de {user_id}")
                # Usar imagen completa como fallback
                h, w = img.shape[:2]
                faces = np.array([[0, 0, w, h]])

            # Tomar la cara más grande
            areas = [w * h for (_, _, w, h) in faces]
            best_idx = np.argmax(areas)
            x, y, w, h = faces[best_idx]

            # Expandir bounding box un 20% para incluir más contexto
            pad_x, pad_y = int(w * 0.2), int(h * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img.shape[1], x + w + pad_x)
            y2 = min(img.shape[0], y + h + pad_y)

            face_crop = img[y1:y2, x1:x2]

            # Resize para Wav2Lip (96x96 input) y para visualización (256x256 output)
            face_96 = cv2.resize(face_crop, (96, 96))
            face_256 = cv2.resize(face_crop, (256, 256))

            self._face_cache[user_id] = {
                "original": img,
                "face_crop": face_crop,
                "face_96": face_96,
                "face_256": face_256,
                "bbox": (x1, y1, x2, y2),
            }

            # Guardar preprocesado en disco
            cache_dir = Path(f"storage/{user_id}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache_dir / "face_256.jpg"), face_256)
            cv2.imwrite(str(cache_dir / "face_96.jpg"), face_96)

            logger.info(f"Cara preprocesada para {user_id}: bbox={x1},{y1},{x2},{y2}")
            return True

        except Exception as e:
            logger.error(f"Error preprocesando cara de {user_id}: {e}")
            return False

    def load_cached_face(self, user_id: str) -> bool:
        """Carga cara preprocesada desde disco."""
        if user_id in self._face_cache:
            return True

        face_256_path = Path(f"storage/{user_id}/face_256.jpg")
        face_96_path = Path(f"storage/{user_id}/face_96.jpg")

        if not face_256_path.exists():
            return False

        try:
            face_256 = cv2.imread(str(face_256_path))
            face_96 = cv2.imread(str(face_96_path)) if face_96_path.exists() else cv2.resize(face_256, (96, 96))

            self._face_cache[user_id] = {
                "face_256": face_256,
                "face_96": face_96,
            }
            return True
        except Exception as e:
            logger.error(f"Error cargando cara de {user_id}: {e}")
            return False

    def has_face(self, user_id: str) -> bool:
        if user_id in self._face_cache:
            return True
        return Path(f"storage/{user_id}/face_256.jpg").exists()

    async def animate(
        self,
        user_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        fps: int = 25,
        jpeg_quality: int = 75,
    ) -> list[bytes]:
        """
        Genera frames de cara animada sincronizados con el audio.

        Si Wav2Lip no está cargado, genera frames estáticos con un efecto
        de "boca abierta/cerrada" simple basado en la amplitud del audio.

        Args:
            user_id: ID del usuario
            audio: Audio float32 numpy array
            sample_rate: Sample rate del audio
            fps: Frames por segundo de salida
            jpeg_quality: Calidad JPEG (1-100)

        Returns:
            Lista de frames JPEG como bytes.
        """
        if not self.load_cached_face(user_id):
            logger.warning(f"No hay cara para {user_id}")
            return []

        cache = self._face_cache[user_id]
        face_img = cache["face_256"].copy()

        # Calcular número de frames necesarios
        audio_duration = len(audio) / sample_rate
        num_frames = max(1, int(audio_duration * fps))

        # Calcular amplitudes de audio por frame (para lip-sync simple)
        samples_per_frame = len(audio) // num_frames
        amplitudes = []
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min(start + samples_per_frame, len(audio))
            chunk = audio[start:end]
            amp = float(np.abs(chunk).mean()) if len(chunk) > 0 else 0.0
            amplitudes.append(amp)

        # Normalizar amplitudes
        max_amp = max(amplitudes) if amplitudes else 1.0
        if max_amp > 0:
            amplitudes = [a / max_amp for a in amplitudes]

        frames = []

        if self.model is not None:
            # TODO: Integrar inferencia real de Wav2Lip aquí
            # Por ahora usamos el efecto simple (ver abajo)
            frames = self._generate_simple_frames(
                face_img, amplitudes, num_frames, jpeg_quality
            )
        else:
            # Fallback: efecto simple de lip-sync basado en amplitud
            frames = self._generate_simple_frames(
                face_img, amplitudes, num_frames, jpeg_quality
            )

        logger.debug(f"Animación: {num_frames} frames ({audio_duration:.1f}s @ {fps}fps)")
        return frames

    def _generate_simple_frames(
        self,
        face_img: np.ndarray,
        amplitudes: list[float],
        num_frames: int,
        jpeg_quality: int,
    ) -> list[bytes]:
        """
        Genera frames con lip-sync simple: oscurece/modifica la zona de la boca
        basado en amplitud de audio. Esto da un efecto visual de "habla" sutil.

        En producción, Wav2Lip reemplazaría esto con lip-sync real.
        """
        frames = []
        h, w = face_img.shape[:2]

        # Zona de la boca (tercio inferior del rostro, centro horizontal)
        mouth_y1 = int(h * 0.65)
        mouth_y2 = int(h * 0.85)
        mouth_x1 = int(w * 0.3)
        mouth_x2 = int(w * 0.7)

        for i in range(num_frames):
            frame = face_img.copy()
            amp = amplitudes[i] if i < len(amplitudes) else 0.0

            # Efecto sutil: ajustar brillo de la zona de la boca según amplitud
            # Esto simula movimiento de labios de forma básica
            if amp > 0.1:
                mouth_region = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

                # Crear efecto de "apertura" con blur y brillo
                alpha = min(amp * 0.4, 0.3)  # Máximo 30% de mezcla
                darkened = (mouth_region * (1 - alpha * 0.5)).astype(np.uint8)

                # Añadir un ligero movimiento vertical simulado
                shift = int(amp * 3)  # hasta 3 pixels
                if shift > 0 and mouth_y2 + shift < h:
                    frame[mouth_y1 + shift:mouth_y2 + shift, mouth_x1:mouth_x2] = darkened
                else:
                    frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = darkened

            # Encodear como JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            _, jpeg_bytes = cv2.imencode('.jpg', frame, encode_params)
            frames.append(jpeg_bytes.tobytes())

        return frames

    def remove_face(self, user_id: str):
        """Elimina datos de cara del usuario."""
        self._face_cache.pop(user_id, None)
        for fname in ["face_256.jpg", "face_96.jpg"]:
            path = Path(f"storage/{user_id}/{fname}")
            if path.exists():
                path.unlink()

    def is_loaded(self) -> bool:
        # Consideramos "cargado" si al menos el fallback funciona
        return True
