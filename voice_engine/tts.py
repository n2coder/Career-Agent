import asyncio
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import time
from uuid import uuid4

from fastapi import HTTPException
from openai import AsyncOpenAI

from voice_engine.config import VoiceSettings as Settings

_log = logging.getLogger(__name__)
_CLEANUP_INTERVAL_SEC = 60.0


class TTSService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self._last_cleanup_at: float = 0.0

    async def synthesize(self, text: str) -> str:
        """Synthesize to disk and return the URL path (for HTTP clients)."""
        if not self.client:
            raise HTTPException(status_code=500, detail="OpenAI API key is not configured for TTS")

        self.settings.audio_path.mkdir(parents=True, exist_ok=True)
        await self._cleanup_audio_cache()
        output_name = f"{uuid4().hex}.mp3"
        output_path = Path(self.settings.audio_path) / output_name
        await self._synthesize_to_path(text=text, output_path=output_path)
        return f"/static/audio/{output_name}"

    async def synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesize and return raw MP3 bytes (no disk I/O — for WebSocket streaming)."""
        if not self.client:
            raise HTTPException(status_code=500, detail="OpenAI API key is not configured for TTS")

        speed = min(2.0, max(0.8, float(self.settings.openai_tts_speed or 1.0)))
        request_payload = {
            "model": self.settings.openai_tts_model,
            "voice": self.settings.openai_tts_voice,
            "input": text,
            "speed": speed,
        }
        instructions = (self.settings.openai_tts_instructions or "").strip()
        if instructions:
            request_payload["instructions"] = instructions

        try:
            response = await self.client.audio.speech.create(**request_payload)
            return response.content
        except TypeError:
            request_payload.pop("instructions", None)
            response = await self.client.audio.speech.create(**request_payload)
            return response.content

    async def ensure_named_audio(self, text: str, output_name: str) -> str | None:
        if not self.client:
            return None
        self.settings.audio_path.mkdir(parents=True, exist_ok=True)
        output_path = Path(self.settings.audio_path) / output_name
        if output_path.exists() and output_path.stat().st_size > 0:
            return f"/static/audio/{output_name}"
        try:
            await self._synthesize_to_path(text=text, output_path=output_path)
            return f"/static/audio/{output_name}"
        except Exception:
            return None

    async def _synthesize_to_path(self, text: str, output_path: Path) -> None:
        if not self.client:
            raise HTTPException(status_code=500, detail="OpenAI API key is not configured for TTS")

        speed = min(2.0, max(0.8, float(self.settings.openai_tts_speed or 1.0)))
        request_payload = {
            "model": self.settings.openai_tts_model,
            "voice": self.settings.openai_tts_voice,
            "input": text,
            "speed": speed,
        }
        instructions = (self.settings.openai_tts_instructions or "").strip()
        if instructions:
            request_payload["instructions"] = instructions

        try:
            async with self.client.audio.speech.with_streaming_response.create(**request_payload) as response:
                await response.stream_to_file(output_path)
        except TypeError:
            # Backward compatibility if SDK/model does not accept `instructions`.
            if "instructions" not in request_payload:
                raise
            request_payload.pop("instructions", None)
            async with self.client.audio.speech.with_streaming_response.create(**request_payload) as response:
                await response.stream_to_file(output_path)

    async def _cleanup_audio_cache(self) -> None:
        # Rate-limit to avoid scanning the audio directory on every TTS synthesis call
        now = time.monotonic()
        if now - self._last_cleanup_at < _CLEANUP_INTERVAL_SEC:
            return
        self._last_cleanup_at = now

        retention = max(10, int(self.settings.tts_retention_minutes))
        max_files = max(50, int(self.settings.tts_max_files))
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=retention)
        audio_dir = Path(self.settings.audio_path)

        def _cleanup() -> None:
            # Single sort: newest first. Delete expired files by time, then trim by count.
            try:
                all_files = sorted(
                    [p for p in audio_dir.glob("*.mp3") if p.is_file()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except OSError:
                return
            kept: list = []
            for p in all_files:
                try:
                    modified = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                    if modified < cutoff:
                        p.unlink(missing_ok=True)
                    else:
                        kept.append(p)
                except OSError:
                    continue
            for p in kept[max_files:]:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    continue

        await asyncio.to_thread(_cleanup)
