import json
import logging

from voice_engine.config import VoiceSettings as Settings

_log = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    aioredis = None  # type: ignore[assignment]
    _REDIS_AVAILABLE = False


class RedisSessionStore:
    """Server-side session history backed by Redis.

    Provides graceful degradation: if Redis is unavailable or not installed,
    all operations silently no-op and history falls back to empty per turn.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._enabled = False
        self._client = None
        if not _REDIS_AVAILABLE:
            _log.warning("redis package not installed — session history will not persist across turns")
            return
        if not settings.redis_session_enabled:
            _log.info("Redis session store disabled via config (REDIS_SESSION_ENABLED=false)")
            return
        try:
            self._client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        except Exception:
            _log.exception("Failed to create Redis client — session history disabled")

    async def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.ping()
            self._enabled = True
            _log.info("Redis session store connected: %s", self._settings.redis_url)
            return True
        except Exception:
            _log.warning("Redis ping failed — session history will be in-memory only")
            self._enabled = False
            return False

    async def load(self, session_id: str) -> list[dict]:
        if not self._enabled or not self._client:
            return []
        try:
            raw = await self._client.get(f"session:{session_id}")
            if not raw:
                return []
            data = json.loads(raw)
            if not isinstance(data, list):
                return []
            return [
                item for item in data
                if isinstance(item, dict)
                and item.get("role") in {"user", "assistant"}
                and item.get("content")
            ]
        except Exception:
            _log.warning("Redis load failed for session %s", session_id)
            return []

    async def save(self, session_id: str, history: list[dict], ttl_minutes: int) -> None:
        if not self._enabled or not self._client:
            return
        try:
            trimmed = history[-self._settings.session_max_messages:]
            await self._client.setex(
                f"session:{session_id}",
                ttl_minutes * 60,
                json.dumps(trimmed),
            )
        except Exception:
            _log.warning("Redis save failed for session %s", session_id)

    async def delete(self, session_id: str) -> None:
        if not self._enabled or not self._client:
            return
        try:
            await self._client.delete(f"session:{session_id}")
        except Exception:
            _log.warning("Redis delete failed for session %s", session_id)

    async def set_partial(self, session_id: str, text: str, ttl_sec: int = 30) -> None:
        """Store a partial (in-flight) transcript fragment with short TTL."""
        if not self._enabled or not self._client:
            return
        try:
            await self._client.setex(f"partial:{session_id}", ttl_sec, text)
        except Exception:
            pass  # hot path — silent fail is fine

    async def get_partial(self, session_id: str) -> str:
        """Return accumulated partial transcript, or empty string."""
        if not self._enabled or not self._client:
            return ""
        try:
            raw = await self._client.get(f"partial:{session_id}")
            return raw or ""
        except Exception:
            return ""

    async def clear_partial(self, session_id: str) -> None:
        """Delete partial key once consumed by a turn."""
        if not self._enabled or not self._client:
            return
        try:
            await self._client.delete(f"partial:{session_id}")
        except Exception:
            pass

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass
