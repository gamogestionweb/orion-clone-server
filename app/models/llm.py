import logging
import re
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class LLMProxy:
    """
    Proxy para LLMs cloud (OpenAI/Anthropic).
    Streaming por oraciones para pipeline TTS paralelo.
    """

    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self._client = None
        logger.info(f"LLMProxy configurado: provider={provider}, model={model}")

    def load(self):
        """Inicializa el cliente del LLM."""
        if self.provider == "openai":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Provider no soportado: {self.provider}")
        logger.info(f"LLM client inicializado: {self.provider}")

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> str:
        """Genera respuesta completa (no streaming)."""
        if self.provider == "openai":
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                max_tokens=500,
                temperature=0.8,
            )
            return response.choices[0].message.content or ""

        elif self.provider == "anthropic":
            response = await self._client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=messages,
                max_tokens=500,
                temperature=0.8,
            )
            return response.content[0].text if response.content else ""

        return ""

    async def generate_streaming(
        self,
        system_prompt: str,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        """
        Genera respuesta en streaming, yieldeando oraciones completas.
        Esto permite que el TTS empiece a sintetizar mientras el LLM
        sigue generando texto.
        """
        buffer = ""
        # Patrón para detectar fin de oración
        sentence_end = re.compile(r'[.!?:;]\s*$')

        if self.provider == "openai":
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                max_tokens=500,
                temperature=0.8,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    buffer += delta.content

                    # Buscar fin de oración
                    if sentence_end.search(buffer) and len(buffer) > 10:
                        sentence = buffer.strip()
                        buffer = ""
                        if sentence:
                            yield sentence

            # Flush del buffer restante
            if buffer.strip():
                yield buffer.strip()

        elif self.provider == "anthropic":
            async with self._client.messages.stream(
                model=self.model,
                system=system_prompt,
                messages=messages,
                max_tokens=500,
                temperature=0.8,
            ) as stream:
                async for text in stream.text_stream:
                    buffer += text

                    if sentence_end.search(buffer) and len(buffer) > 10:
                        sentence = buffer.strip()
                        buffer = ""
                        if sentence:
                            yield sentence

            if buffer.strip():
                yield buffer.strip()

    def is_loaded(self) -> bool:
        return self._client is not None
