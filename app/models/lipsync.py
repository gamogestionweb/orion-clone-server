import logging
import io
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional
import random

logger = logging.getLogger(__name__)


class Wav2LipAnimator:
    """
    Animación facial con Video Loop + Lip-sync.

    El usuario graba un video corto (5-10s) mirando a cámara.
    El servidor:
    1. Extrae frames del video como "loop base" (parpadeo natural, micro-movimientos)
    2. Aplica lip-sync con audio generado (modifica zona de la boca según amplitud)
    3. Envía frames JPEG al cliente

    Si wav2lip_gan.pth está disponible, usa Wav2Lip real para lip-sync.
    Si no, usa un fallback basado en amplitud de audio.
    """

    def __init__(self, checkpoint_path: str, face_detect_path: str = "models/s3fd.pth"):
        self.checkpoint_path = checkpoint_path
        self.face_detect_path = face_detect_path
        self.wav2lip_model = None
        self.face_detector = None

        # Cache de video frames por usuario
        self._video_cache: dict[str, dict] = {}
        # Cache de foto estática (fallback)
        self._face_cache: dict[str, dict] = {}

        logger.info(f"Wav2LipAnimator configurado: checkpoint={checkpoint_path}")

    def load(self):
        """Carga modelos de Wav2Lip si están disponibles."""
        try:
            if Path(self.checkpoint_path).exists():
                logger.info(f"Wav2Lip checkpoint encontrado: {self.checkpoint_path}")
                # TODO: Cargar modelo Wav2Lip real aquí cuando se integre
                # Por ahora usamos el sistema de video loop + lip-sync por amplitud
                logger.info("Wav2Lip: usando video loop + lip-sync por amplitud")
            else:
                logger.warning(
                    f"Checkpoint Wav2Lip no encontrado: {self.checkpoint_path}. "
                    "Descarga wav2lip_gan.pth de https://github.com/Rudrabha/Wav2Lip"
                )
        except Exception as e:
            logger.error(f"Error cargando Wav2Lip: {e}")

    # ==========================================
    # VIDEO PROCESSING
    # ==========================================

    def preprocess_video(self, user_id: str, video_path: str, max_frames: int = 150) -> bool:
        """
        Preprocesa el video del usuario para loop de animación.

        Extrae frames, detecta la cara, y prepara para lip-sync.
        El video se loopea durante la conversación para dar vida natural
        (parpadeo, respiración, micro-movimientos).

        Args:
            user_id: ID del usuario
            video_path: Path al video MP4/WEBM
            max_frames: Máximo de frames a extraer (5s * 30fps = 150)

        Returns:
            True si se procesó correctamente.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"No se pudo abrir video: {video_path}")
                return False

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video: {total_frames} frames @ {fps}fps")

            # Extraer frames
            frames = []
            frame_count = 0
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1

            cap.release()

            if len(frames) < 5:
                logger.error(f"Video muy corto: {len(frames)} frames")
                return False

            # Detectar cara en el primer frame
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            if len(faces) == 0:
                # Usar frame completo
                h, w = frames[0].shape[:2]
                bbox = (0, 0, w, h)
                logger.warning("No se detectó cara, usando frame completo")
            else:
                areas = [w * h for (_, _, w, h) in faces]
                best_idx = np.argmax(areas)
                x, y, w, h = faces[best_idx]
                # Expandir bbox 20%
                pad_x, pad_y = int(w * 0.2), int(h * 0.2)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frames[0].shape[1], x + w + pad_x)
                y2 = min(frames[0].shape[0], y + h + pad_y)
                bbox = (x1, y1, x2, y2)

            # Recortar y resize todos los frames a 256x256
            processed_frames = []
            x1, y1, x2, y2 = bbox
            for frame in frames:
                face_crop = frame[y1:y2, x1:x2]
                face_256 = cv2.resize(face_crop, (256, 256))
                processed_frames.append(face_256)

            # Detectar zona de la boca (tercio inferior, centro)
            mouth_bbox = {
                "y1": int(256 * 0.62),
                "y2": int(256 * 0.88),
                "x1": int(256 * 0.25),
                "x2": int(256 * 0.75),
            }

            self._video_cache[user_id] = {
                "frames": processed_frames,
                "fps": fps,
                "bbox": bbox,
                "mouth_bbox": mouth_bbox,
                "frame_count": len(processed_frames),
            }

            # Guardar metadata en disco
            cache_dir = Path(f"storage/{user_id}")
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Guardar frames como numpy para rápida carga futura
            np.savez_compressed(
                str(cache_dir / "video_frames.npz"),
                frames=np.array(processed_frames),
                fps=np.array([fps]),
                mouth_bbox=np.array([mouth_bbox["y1"], mouth_bbox["y2"],
                                     mouth_bbox["x1"], mouth_bbox["x2"]]),
            )

            logger.info(f"Video preprocesado para {user_id}: "
                        f"{len(processed_frames)} frames @ {fps}fps, "
                        f"mouth_bbox={mouth_bbox}")
            return True

        except Exception as e:
            logger.error(f"Error preprocesando video de {user_id}: {e}")
            return False

    def preprocess_face(self, user_id: str, image_path: str) -> bool:
        """Preprocesa foto estática (fallback si no hay video)."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

            if len(faces) == 0:
                h, w = img.shape[:2]
                faces = np.array([[0, 0, w, h]])

            areas = [w * h for (_, _, w, h) in faces]
            best_idx = np.argmax(areas)
            x, y, w, h = faces[best_idx]
            pad_x, pad_y = int(w * 0.2), int(h * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img.shape[1], x + w + pad_x)
            y2 = min(img.shape[0], y + h + pad_y)

            face_crop = img[y1:y2, x1:x2]
            face_256 = cv2.resize(face_crop, (256, 256))

            self._face_cache[user_id] = {
                "face_256": face_256,
                "bbox": (x1, y1, x2, y2),
            }

            cache_dir = Path(f"storage/{user_id}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cache_dir / "face_256.jpg"), face_256)

            logger.info(f"Cara preprocesada para {user_id}: bbox={x1},{y1},{x2},{y2}")
            return True

        except Exception as e:
            logger.error(f"Error preprocesando cara de {user_id}: {e}")
            return False

    def load_cached_data(self, user_id: str) -> bool:
        """Carga video o foto cacheada desde disco."""
        if user_id in self._video_cache:
            return True

        # Intentar cargar video
        npz_path = Path(f"storage/{user_id}/video_frames.npz")
        if npz_path.exists():
            try:
                data = np.load(str(npz_path))
                frames = list(data["frames"])
                fps = float(data["fps"][0])
                mb = data["mouth_bbox"]
                self._video_cache[user_id] = {
                    "frames": frames,
                    "fps": fps,
                    "mouth_bbox": {"y1": int(mb[0]), "y2": int(mb[1]),
                                   "x1": int(mb[2]), "x2": int(mb[3])},
                    "frame_count": len(frames),
                }
                logger.info(f"Video cacheado cargado: {user_id} ({len(frames)} frames)")
                return True
            except Exception as e:
                logger.error(f"Error cargando video cache: {e}")

        # Fallback: cargar foto
        if user_id in self._face_cache:
            return True

        face_path = Path(f"storage/{user_id}/face_256.jpg")
        if face_path.exists():
            face_256 = cv2.imread(str(face_path))
            if face_256 is not None:
                self._face_cache[user_id] = {"face_256": face_256}
                return True

        return False

    def has_face(self, user_id: str) -> bool:
        if user_id in self._video_cache or user_id in self._face_cache:
            return True
        return (Path(f"storage/{user_id}/video_frames.npz").exists() or
                Path(f"storage/{user_id}/face_256.jpg").exists())

    def has_video(self, user_id: str) -> bool:
        return (user_id in self._video_cache or
                Path(f"storage/{user_id}/video_frames.npz").exists())

    # ==========================================
    # ANIMACIÓN
    # ==========================================

    async def animate(
        self,
        user_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        fps: int = 25,
        jpeg_quality: int = 75,
    ) -> list[bytes]:
        """
        Genera frames animados sincronizados con el audio.

        Si hay video: usa video loop como base + lip-sync
        Si solo hay foto: usa foto estática + lip-sync simple
        """
        if not self.load_cached_data(user_id):
            logger.warning(f"No hay datos de cara para {user_id}")
            return []

        # Calcular amplitudes de audio por frame
        audio_duration = len(audio) / sample_rate
        num_frames = max(1, int(audio_duration * fps))
        samples_per_frame = len(audio) // num_frames

        amplitudes = []
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min(start + samples_per_frame, len(audio))
            chunk = audio[start:end]
            amp = float(np.abs(chunk).mean()) if len(chunk) > 0 else 0.0
            amplitudes.append(amp)

        max_amp = max(amplitudes) if amplitudes else 1.0
        if max_amp > 0:
            amplitudes = [a / max_amp for a in amplitudes]

        # Elegir método según datos disponibles
        if user_id in self._video_cache:
            frames = self._animate_video_loop(user_id, amplitudes, num_frames, jpeg_quality)
        else:
            face_img = self._face_cache[user_id]["face_256"]
            frames = self._animate_static_face(face_img, amplitudes, num_frames, jpeg_quality)

        logger.debug(f"Animación: {num_frames} frames ({audio_duration:.1f}s @ {fps}fps) "
                     f"[{'video' if user_id in self._video_cache else 'foto'}]")
        return frames

    def _animate_video_loop(
        self,
        user_id: str,
        amplitudes: list[float],
        num_frames: int,
        jpeg_quality: int,
    ) -> list[bytes]:
        """
        Anima usando el video loop del usuario como base.
        El video se loopea y se le aplica lip-sync basado en amplitud.
        Resultado: cara natural (parpadeo, movimiento) + labios sincronizados.
        """
        cache = self._video_cache[user_id]
        base_frames = cache["frames"]
        mouth = cache["mouth_bbox"]
        total_base = len(base_frames)

        frames = []
        for i in range(num_frames):
            # Loopear el video base (ping-pong para transición suave)
            cycle_len = total_base * 2 - 2  # ida y vuelta
            if cycle_len <= 0:
                cycle_len = 1
            pos = i % cycle_len
            if pos < total_base:
                frame_idx = pos
            else:
                frame_idx = cycle_len - pos  # rebote

            frame = base_frames[frame_idx].copy()
            amp = amplitudes[i] if i < len(amplitudes) else 0.0

            # === LIP-SYNC POR AMPLITUD ===
            if amp > 0.05:
                my1, my2 = mouth["y1"], mouth["y2"]
                mx1, mx2 = mouth["x1"], mouth["x2"]

                mouth_region = frame[my1:my2, mx1:mx2].copy()
                mouth_h = my2 - my1

                # Simular apertura de boca:
                # Comprimir verticalmente la zona inferior de la boca proporcional a amplitud
                open_amount = int(amp * mouth_h * 0.3)  # hasta 30% de apertura

                if open_amount > 1:
                    # Dividir boca en mitad superior e inferior
                    mid_y = mouth_h // 2

                    upper_lip = mouth_region[:mid_y, :]
                    lower_lip = mouth_region[mid_y:, :]

                    # Mover labio inferior hacia abajo (oscurecer espacio = boca abierta)
                    gap = np.zeros((open_amount, mx2 - mx1, 3), dtype=np.uint8)
                    # Color interior de boca (oscuro rojizo)
                    gap[:, :] = [30, 20, 40]

                    # Reconstruir zona de boca
                    new_mouth_h = mid_y + open_amount + (mouth_h - mid_y)
                    if my1 + new_mouth_h < 256:
                        # Comprimir para mantener mismo tamaño
                        combined = np.vstack([upper_lip, gap, lower_lip])
                        resized = cv2.resize(combined, (mx2 - mx1, mouth_h))
                        frame[my1:my2, mx1:mx2] = resized

                    # Suavizar bordes con blur
                    blur_region = frame[max(0, my1 - 2):min(256, my2 + 2),
                                        max(0, mx1 - 2):min(256, mx2 + 2)]
                    frame[max(0, my1 - 2):min(256, my2 + 2),
                          max(0, mx1 - 2):min(256, mx2 + 2)] = cv2.GaussianBlur(
                        blur_region, (3, 3), 0
                    )

            # Encodear como JPEG
            _, jpeg_bytes = cv2.imencode('.jpg', frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            frames.append(jpeg_bytes.tobytes())

        return frames

    def _animate_static_face(
        self,
        face_img: np.ndarray,
        amplitudes: list[float],
        num_frames: int,
        jpeg_quality: int,
    ) -> list[bytes]:
        """Fallback: anima foto estática con lip-sync + parpadeo simulado."""
        frames = []
        h, w = face_img.shape[:2]

        mouth_y1 = int(h * 0.62)
        mouth_y2 = int(h * 0.88)
        mouth_x1 = int(h * 0.25)
        mouth_x2 = int(h * 0.75)

        # Parpadeo aleatorio cada ~3-5 segundos
        blink_frames = set()
        blink_interval = random.randint(75, 125)  # cada 3-5s a 25fps
        for b in range(0, num_frames, blink_interval):
            blink_start = b + random.randint(0, 20)
            for bf in range(blink_start, min(blink_start + 4, num_frames)):
                blink_frames.add(bf)

        eye_y1 = int(h * 0.25)
        eye_y2 = int(h * 0.40)

        for i in range(num_frames):
            frame = face_img.copy()
            amp = amplitudes[i] if i < len(amplitudes) else 0.0

            # Parpadeo
            if i in blink_frames:
                eye_region = frame[eye_y1:eye_y2, :]
                # Oscurecer ojos ligeramente para simular parpadeo
                blink_alpha = 0.4
                frame[eye_y1:eye_y2, :] = (eye_region * (1 - blink_alpha)).astype(np.uint8)

            # Lip-sync
            if amp > 0.05:
                mouth_region = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2].copy()
                mouth_h = mouth_y2 - mouth_y1
                open_amount = int(amp * mouth_h * 0.3)

                if open_amount > 1:
                    mid_y = mouth_h // 2
                    upper = mouth_region[:mid_y, :]
                    lower = mouth_region[mid_y:, :]
                    gap = np.full((open_amount, mouth_x2 - mouth_x1, 3), [30, 20, 40],
                                 dtype=np.uint8)
                    combined = np.vstack([upper, gap, lower])
                    resized = cv2.resize(combined, (mouth_x2 - mouth_x1, mouth_h))
                    frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = resized

            _, jpeg_bytes = cv2.imencode('.jpg', frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            frames.append(jpeg_bytes.tobytes())

        return frames

    def remove_face(self, user_id: str):
        self._video_cache.pop(user_id, None)
        self._face_cache.pop(user_id, None)
        for fname in ["face_256.jpg", "video_frames.npz"]:
            path = Path(f"storage/{user_id}/{fname}")
            if path.exists():
                path.unlink()

    def is_loaded(self) -> bool:
        return True
