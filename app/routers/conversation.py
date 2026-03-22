import asyncio
import base64
import json
import logging
import time
import uuid
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.audio_utils import pcm16_to_float32, float32_to_pcm16, resample_audio, split_into_chunks

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversation"])

# Referencias a modelos y servicios (se inyectan desde main.py)
_stt = None
_tts = None
_lipsync = None
_llm = None
_session_manager = None
_config = None


def init_router(stt, tts, lipsync, llm, session_manager, config):
    global _stt, _tts, _lipsync, _llm, _session_manager, _config
    _stt = stt
    _tts = tts
    _lipsync = lipsync
    _llm = llm
    _session_manager = session_manager
    _config = config


class AudioBuffer:
    """Buffer acumulativo de audio PCM con detección de silencio."""

    def __init__(self, sample_rate: int = 16000, silence_threshold_ms: int = 800):
        self.sample_rate = sample_rate
        self.silence_threshold_ms = silence_threshold_ms
        self._chunks: list[bytes] = []
        self._total_bytes = 0
        self._last_voice_time = time.time()
        self._has_speech = False
        # Umbral de energía para detectar voz (ajustable)
        self._energy_threshold = 0.01

    def append(self, pcm_chunk: bytes):
        """Añade chunk de audio PCM 16-bit."""
        self._chunks.append(pcm_chunk)
        self._total_bytes += len(pcm_chunk)

        # Detectar si hay voz en este chunk
        audio = pcm16_to_float32(pcm_chunk)
        energy = float(np.abs(audio).mean())

        if energy > self._energy_threshold:
            self._last_voice_time = time.time()
            self._has_speech = True

    def is_speech_ended(self) -> bool:
        """Retorna True si hubo habla y ahora hay silencio sostenido."""
        if not self._has_speech:
            return False

        silence_duration = (time.time() - self._last_voice_time) * 1000
        return silence_duration >= self.silence_threshold_ms

    def flush(self) -> bytes:
        """Retorna todo el audio acumulado y limpia el buffer."""
        all_audio = b"".join(self._chunks)
        self._chunks.clear()
        self._total_bytes = 0
        self._has_speech = False
        return all_audio

    def clear(self):
        """Limpia el buffer sin retornar datos."""
        self._chunks.clear()
        self._total_bytes = 0
        self._has_speech = False

    @property
    def duration_seconds(self) -> float:
        return self._total_bytes / (self.sample_rate * 2)  # 16-bit = 2 bytes


@router.websocket("/ws/conversation/{user_id}")
async def conversation_websocket(websocket: WebSocket, user_id: str):
    """
    WebSocket de conversación en tiempo real.

    Protocolo compatible con ElevenLabs para minimizar cambios en el cliente Android.
    Añade eventos 'face_frame' para la animación facial.
    """
    await websocket.accept()

    # Verificar que el usuario tiene voz configurada
    if not _tts or not _tts.has_speaker(user_id):
        await websocket.send_json({
            "type": "error",
            "error": f"Usuario {user_id} no tiene voz configurada"
        })
        await websocket.close(1008, "No voice configured")
        return

    # Cargar embedding de voz
    _tts.load_speaker_embedding(user_id)

    # Cargar cara si existe
    has_face = _lipsync and _lipsync.has_face(user_id)
    if has_face:
        _lipsync.load_cached_face(user_id)

    # Crear sesión
    conversation_id = str(uuid.uuid4())[:8]
    session = _session_manager.create_session(user_id, conversation_id)
    if session is None:
        await websocket.send_json({
            "type": "error",
            "error": "Servidor lleno. Intenta más tarde."
        })
        await websocket.close(1013, "Server full")
        return

    # Buffer de audio
    audio_buffer = AudioBuffer(
        sample_rate=_config.input_sample_rate,
        silence_threshold_ms=_config.vad_silence_threshold_ms,
    )

    system_prompt = ""
    language = "es"
    is_processing = False
    ping_counter = 0

    logger.info(f"Conversación iniciada: {user_id} ({conversation_id})")

    try:
        # Esperar mensaje de inicialización
        init_data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
        init_msg = json.loads(init_data)

        if init_msg.get("type") == "conversation_init":
            system_prompt = init_msg.get("system_prompt", "")
            language = init_msg.get("language", "es")
            session.system_prompt = system_prompt
            session.language = language

        # Enviar metadata de conexión (formato ElevenLabs compatible)
        await websocket.send_json({
            "type": "conversation_initiation_metadata",
            "conversation_initiation_metadata_event": {
                "conversation_id": conversation_id,
                "agent_output_audio_format": f"pcm_{_config.output_sample_rate}",
            }
        })

        # Tarea de ping keep-alive
        async def ping_task():
            nonlocal ping_counter
            while True:
                await asyncio.sleep(15)
                try:
                    ping_counter += 1
                    await websocket.send_json({
                        "type": "ping",
                        "ping_event": {"event_id": ping_counter, "ping_ms": 0}
                    })
                except Exception:
                    break

        ping_job = asyncio.create_task(ping_task())

        # Tarea de detección de fin de habla
        async def vad_check_task():
            nonlocal is_processing
            while True:
                await asyncio.sleep(0.1)  # Check cada 100ms
                if audio_buffer.is_speech_ended() and not is_processing:
                    is_processing = True
                    try:
                        await process_utterance(
                            websocket, user_id, audio_buffer, session,
                            system_prompt, language, has_face,
                        )
                    except Exception as e:
                        logger.error(f"Error procesando utterance: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "error": str(e)
                        })
                    finally:
                        is_processing = False

        vad_job = asyncio.create_task(vad_check_task())

        # Loop principal: recibir audio del cliente
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60)
                msg = json.loads(data)

                if "user_audio_chunk" in msg:
                    pcm_bytes = base64.b64decode(msg["user_audio_chunk"])
                    audio_buffer.append(pcm_bytes)

                elif msg.get("type") == "interrupt":
                    audio_buffer.clear()
                    # No interrumpimos el TTS actualmente en progreso
                    # El cliente puede manejar esto silenciando su AudioTrack

                elif msg.get("type") == "pong":
                    pass  # Keep-alive response

                elif msg.get("type") == "contextual_update":
                    # Añadir contexto adicional al system prompt
                    extra_context = msg.get("text", "")
                    if extra_context:
                        system_prompt += f"\n\n{extra_context}"
                        session.system_prompt = system_prompt

            except asyncio.TimeoutError:
                # Sin audio en 60s - enviar ping
                continue

    except WebSocketDisconnect:
        logger.info(f"Cliente desconectado: {user_id}")
    except Exception as e:
        logger.error(f"Error en conversación {user_id}: {e}")
    finally:
        # Cleanup
        ping_job.cancel()
        vad_job.cancel()
        _session_manager.remove_session(user_id)
        logger.info(f"Conversación terminada: {user_id} ({conversation_id})")


async def process_utterance(
    websocket: WebSocket,
    user_id: str,
    audio_buffer: AudioBuffer,
    session,
    system_prompt: str,
    language: str,
    has_face: bool,
):
    """Procesa una utterance completa: STT → LLM → TTS → Lip-sync."""

    # 1. Obtener audio acumulado
    raw_audio = audio_buffer.flush()
    if len(raw_audio) < 3200:  # Menos de 100ms, ignorar
        return

    # Convertir a float32
    audio_float = pcm16_to_float32(raw_audio)

    # 2. STT con Whisper
    transcript = await _stt.transcribe(audio_float, _config.input_sample_rate, language)

    if not transcript or len(transcript.strip()) < 2:
        return  # Transcripción vacía o muy corta

    logger.info(f"[{user_id}] Usuario: {transcript}")

    # Enviar transcripción al cliente
    await websocket.send_json({
        "type": "user_transcript",
        "user_transcription_event": {"user_transcript": transcript}
    })

    # 3. LLM - generar respuesta por oraciones
    _session_manager.add_message(user_id, "user", transcript)

    full_response = ""

    async for sentence in _llm.generate_streaming(system_prompt, session.messages):
        full_response += sentence + " "

        # Enviar texto de respuesta
        await websocket.send_json({
            "type": "agent_response",
            "agent_response_event": {"agent_response": sentence}
        })

        # 4. TTS - sintetizar esta oración con voz clonada
        tts_audio = await _tts.synthesize(user_id, sentence, language)

        if tts_audio is None:
            continue

        # Resamplear de 24kHz (XTTS nativo) a output_sample_rate
        if _tts.get_output_sample_rate() != _config.output_sample_rate:
            tts_audio = resample_audio(
                tts_audio,
                _tts.get_output_sample_rate(),
                _config.output_sample_rate,
            )

        # Convertir a PCM16 bytes
        tts_pcm = float32_to_pcm16(tts_audio)

        # 5. Enviar audio en chunks de 100ms
        chunks = split_into_chunks(tts_pcm, 100, _config.output_sample_rate)
        for chunk in chunks:
            await websocket.send_json({
                "type": "audio",
                "audio_event": {
                    "audio_base_64": base64.b64encode(chunk).decode()
                }
            })
            # Pequeña pausa para no saturar el WebSocket
            await asyncio.sleep(0.01)

        # 6. Lip-sync - generar frames si hay cara
        if has_face and _lipsync:
            try:
                frames = await _lipsync.animate(
                    user_id,
                    tts_audio,
                    sample_rate=_config.output_sample_rate,
                    fps=_config.face_fps,
                    jpeg_quality=_config.jpeg_quality,
                )

                # Enviar frames con timestamp
                frame_interval_ms = 1000 // _config.face_fps
                for idx, frame_jpeg in enumerate(frames):
                    await websocket.send_json({
                        "type": "face_frame",
                        "frame_event": {
                            "jpeg_base_64": base64.b64encode(frame_jpeg).decode(),
                            "timestamp_ms": idx * frame_interval_ms,
                        }
                    })
                    # No dormir entre frames - enviar lo más rápido posible
                    # El cliente bufferiza y reproduce a la velocidad correcta
            except Exception as e:
                logger.error(f"Error en lip-sync: {e}")
                # Continuar sin cara, el audio ya se envió

    # Guardar respuesta completa en historial
    _session_manager.add_message(user_id, "assistant", full_response.strip())
    logger.info(f"[{user_id}] Respuesta: {full_response[:100]}...")
