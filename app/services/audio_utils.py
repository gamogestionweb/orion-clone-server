import io
import struct
import numpy as np


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convierte PCM 16-bit signed a float32 [-1.0, 1.0]."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convierte float32 [-1.0, 1.0] a PCM 16-bit signed bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767).astype(np.int16)
    return samples.tobytes()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resamplea audio de orig_sr a target_sr."""
    if orig_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def write_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Escribe audio float32 como WAV en memoria."""
    pcm = float32_to_pcm16(audio)
    buf = io.BytesIO()

    num_channels = 1
    sample_width = 2  # 16-bit
    data_size = len(pcm)

    # RIFF header
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')

    # fmt chunk
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))  # chunk size
    buf.write(struct.pack('<H', 1))   # PCM format
    buf.write(struct.pack('<H', num_channels))
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', sample_rate * num_channels * sample_width))
    buf.write(struct.pack('<H', num_channels * sample_width))
    buf.write(struct.pack('<H', sample_width * 8))

    # data chunk
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(pcm)

    return buf.getvalue()


def split_into_chunks(audio_bytes: bytes, chunk_duration_ms: int, sample_rate: int) -> list[bytes]:
    """Divide audio PCM en chunks de duración fija."""
    bytes_per_ms = (sample_rate * 2) // 1000  # 16-bit mono
    chunk_size = bytes_per_ms * chunk_duration_ms

    chunks = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks
