"""
Orion Clone Server — Servidor de clonación de voz y cara animada.

Reemplaza ElevenLabs con modelos open-source:
- STT: faster-whisper
- TTS: Coqui XTTS v2 (clonación de voz)
- Lip-sync: Wav2Lip (animación facial)
- LLM: Proxy a OpenAI/Anthropic

Ejecutar:
    uvicorn app.main:app --host 0.0.0.0 --port 8765
    # o con Docker:
    docker-compose up
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ServerConfig
from app.models.stt import WhisperSTT
from app.models.tts import XTTSCloner
from app.models.lipsync import Wav2LipAnimator
from app.models.llm import LLMProxy
from app.services.session_manager import SessionManager
from app.routers import setup as setup_router
from app.routers import conversation as conversation_router

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orion-clone")

# ==========================================
# CONFIGURACIÓN
# ==========================================

config = ServerConfig.from_env()

# ==========================================
# MODELOS GLOBALES
# ==========================================

stt = WhisperSTT(
    model_size=config.whisper_model,
    device=config.whisper_device,
    compute_type=config.whisper_compute_type,
)

tts = XTTSCloner(model_name=config.xtts_model)

lipsync = Wav2LipAnimator(
    checkpoint_path=config.wav2lip_checkpoint,
    face_detect_path=config.face_detect_model,
)

llm = LLMProxy(
    provider=config.llm_provider,
    api_key=config.llm_api_key,
    model=config.llm_model,
)

session_manager = SessionManager(max_concurrent=config.max_concurrent_users)


# ==========================================
# LIFESPAN (precarga de modelos)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precarga modelos al iniciar el servidor."""
    logger.info("=" * 60)
    logger.info("ORION CLONE SERVER — Iniciando...")
    logger.info("=" * 60)

    # Cargar modelos en orden
    logger.info("[1/4] Cargando Whisper STT...")
    stt.load()

    logger.info("[2/4] Cargando XTTS v2...")
    tts.load()

    logger.info("[3/4] Cargando Wav2Lip...")
    lipsync.load()

    logger.info("[4/4] Inicializando LLM proxy...")
    llm.load()

    # Inyectar dependencias en routers
    setup_router.init_router(tts, lipsync, config.storage_path)
    conversation_router.init_router(stt, tts, lipsync, llm, session_manager, config)

    logger.info("=" * 60)
    logger.info(f"Servidor listo en {config.host}:{config.port}")
    logger.info(f"  STT: Whisper {config.whisper_model}")
    logger.info(f"  TTS: XTTS v2")
    logger.info(f"  LLM: {config.llm_provider}/{config.llm_model}")
    logger.info(f"  Face: {'Wav2Lip activo' if lipsync.is_loaded() else 'Simple fallback'}")
    logger.info(f"  Max users: {config.max_concurrent_users}")
    logger.info("=" * 60)

    yield  # Servidor corriendo

    # Cleanup
    logger.info("Servidor cerrándose...")


# ==========================================
# APP
# ==========================================

app = FastAPI(
    title="Orion Clone Server",
    description="Servidor de clonación de voz y cara animada para Orion",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS para cliente Android
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar routers
app.include_router(setup_router.router)
app.include_router(conversation_router.router)


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
async def health_check():
    """Estado del servidor y sus componentes."""
    return {
        "status": "ok",
        "components": {
            "stt": "loaded" if stt.is_loaded() else "not_loaded",
            "tts": "loaded" if tts.is_loaded() else "not_loaded",
            "lipsync": "loaded" if lipsync.is_loaded() else "not_loaded",
            "llm": "loaded" if llm.is_loaded() else "not_loaded",
        },
        "active_sessions": session_manager.active_count,
        "max_sessions": config.max_concurrent_users,
    }


@app.get("/")
async def root():
    return {"service": "Orion Clone Server", "status": "running"}
