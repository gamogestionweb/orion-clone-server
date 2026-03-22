import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Estado de una sesión de conversación activa."""
    user_id: str
    conversation_id: str
    system_prompt: str = ""
    language: str = "es"
    messages: list = field(default_factory=list)
    is_active: bool = True


class SessionManager:
    """Gestiona sesiones activas de conversación."""

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self._sessions: dict[str, UserSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def can_accept(self) -> bool:
        return self.active_count < self.max_concurrent

    def create_session(self, user_id: str, conversation_id: str) -> Optional[UserSession]:
        """Crea nueva sesión. Retorna None si se alcanzó el límite."""
        if not self.can_accept():
            logger.warning(f"Límite de sesiones alcanzado ({self.max_concurrent})")
            return None

        # Si el usuario ya tiene sesión activa, cerrarla
        if user_id in self._sessions:
            logger.info(f"Cerrando sesión previa de {user_id}")
            self.remove_session(user_id)

        session = UserSession(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        self._sessions[user_id] = session
        self._locks[user_id] = asyncio.Lock()

        logger.info(f"Sesión creada: {user_id} ({self.active_count}/{self.max_concurrent})")
        return session

    def get_session(self, user_id: str) -> Optional[UserSession]:
        return self._sessions.get(user_id)

    def get_lock(self, user_id: str) -> asyncio.Lock:
        """Lock por usuario para serializar inferencia."""
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def remove_session(self, user_id: str):
        self._sessions.pop(user_id, None)
        self._locks.pop(user_id, None)
        logger.info(f"Sesión eliminada: {user_id} ({self.active_count}/{self.max_concurrent})")

    def add_message(self, user_id: str, role: str, content: str):
        session = self._sessions.get(user_id)
        if session:
            session.messages.append({"role": role, "content": content})
            # Mantener historial razonable (últimos 20 turnos)
            if len(session.messages) > 40:
                session.messages = session.messages[-40:]
