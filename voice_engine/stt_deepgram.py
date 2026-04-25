"""Deepgram STT service — batch + streaming (SDK v6).

Two modes:
  1. Batch (transcribe_bytes): drop-in replacement for OpenAI Whisper.

  2. Streaming: opens a Deepgram Live WebSocket per utterance.
     Each 500ms audio_chunk_partial is forwarded in real-time; by the time
     the user stops speaking the transcript is already mostly assembled.
     close_stream() returns the final accumulated transcript.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

try:
    from deepgram import DeepgramClient
    from deepgram.listen.v1 import (
        ListenV1Results,
        ListenV1UtteranceEnd,
    )
    _DG_AVAILABLE = True
except ImportError:
    _DG_AVAILABLE = False
    _log.warning("deepgram-sdk not installed — Deepgram STT unavailable")

from voice_engine.config import VoiceSettings as Settings


_NON_ALLOWED = re.compile(r"[\u1100-\u11FF\u3040-\u30FF\uAC00-\uD7FF\u4E00-\u9FFF]")


# ---------------------------------------------------------------------------
# Streaming session state
# ---------------------------------------------------------------------------

@dataclass
class _StreamSession:
    finals: list[str] = field(default_factory=list)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    chunks: asyncio.Queue = field(default_factory=asyncio.Queue)
    lang_hint: str = "en"
    task: "asyncio.Task | None" = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class DeepgramSTTService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        self._sessions: dict[str, _StreamSession] = {}

        if not _DG_AVAILABLE:
            return
        if not settings.deepgram_api_key:
            _log.warning("DEEPGRAM_API_KEY not set — Deepgram STT disabled")
            return
        try:
            self._client = DeepgramClient(api_key=settings.deepgram_api_key)
            _log.info("DeepgramSTTService ready (model=%s)", settings.deepgram_model)
        except Exception:
            _log.exception("Failed to initialise Deepgram client")

    # ------------------------------------------------------------------
    # Batch transcription — drop-in for STTService.transcribe_bytes
    # ------------------------------------------------------------------

    async def transcribe_bytes(
        self,
        content: bytes,
        filename: str = "audio.webm",
        lang_hint: str | None = None,
    ) -> str:
        if not self._client:
            raise RuntimeError("Deepgram client not initialised")
        if not content or len(content) < 600:
            raise ValueError("Audio too short")

        lang = self._resolve_lang(lang_hint)
        mime = _mime_from_filename(filename)

        def _sync_transcribe():
            try:
                response = self._client.listen.v1.media.transcribe_file(
                    {"content": content, "mimetype": mime},
                    model=self._settings.deepgram_model,
                    language=lang,
                    smart_format=True,
                    punctuate=True,
                )
                return response.results.channels[0].alternatives[0].transcript or ""
            except Exception as e:
                _log.exception("Deepgram batch transcription failed")
                raise

        text = await asyncio.get_event_loop().run_in_executor(None, _sync_transcribe)
        text = (text or "").strip()
        if _NON_ALLOWED.search(text):
            return ""
        return text

    # ------------------------------------------------------------------
    # Streaming session management
    # ------------------------------------------------------------------

    async def open_stream(self, session_id: str, lang_hint: str | None = None) -> None:
        """Open a Deepgram Live WebSocket for this utterance."""
        if not self._client:
            return
        if session_id in self._sessions:
            await self.close_stream(session_id)

        lang = self._resolve_lang(lang_hint)
        session = _StreamSession(lang_hint=lang)
        self._sessions[session_id] = session

        async def _stream_worker():
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: self._run_sync_stream(session))
            except Exception:
                _log.exception("Deepgram stream worker failed for %s", session_id)
            finally:
                session.done.set()

        session.task = asyncio.create_task(_stream_worker())
        _log.debug("Deepgram stream opened for session %s (lang=%s)", session_id, lang)

    def _run_sync_stream(self, session: _StreamSession) -> None:
        """Run the synchronous Deepgram WebSocket in a thread."""
        loop = asyncio.new_event_loop()
        try:
            with self._client.listen.v1.connect(
                model=self._settings.deepgram_model,
                language=session.lang_hint,
                smart_format=True,
                punctuate=True,
                interim_results=False,
                endpointing=300,
            ) as ws:
                def _listen():
                    for msg in ws:
                        if isinstance(msg, ListenV1Results):
                            try:
                                text = msg.channel.alternatives[0].transcript or ""
                                if text.strip() and msg.is_final:
                                    session.finals.append(text.strip())
                            except Exception:
                                pass
                        elif isinstance(msg, ListenV1UtteranceEnd):
                            break

                listen_thread = threading.Thread(target=_listen, daemon=True)
                listen_thread.start()

                # Feed chunks from the queue until sentinel (None) arrives
                while True:
                    try:
                        chunk = loop.run_until_complete(
                            asyncio.wait_for(session.chunks.get(), timeout=0.1)
                        )
                    except asyncio.TimeoutError:
                        continue
                    if chunk is None:
                        break
                    ws.send_media(chunk)

                ws.send_finalize()
                listen_thread.join(timeout=2.0)
        except Exception:
            _log.exception("Deepgram sync stream error")
        finally:
            loop.close()

    async def send_chunk(self, session_id: str, audio: bytes) -> None:
        """Forward a raw audio chunk into the open stream."""
        session = self._sessions.get(session_id)
        if not session or not audio:
            return
        await session.chunks.put(audio)

    async def close_stream(self, session_id: str) -> str:
        """Finish the stream and return the accumulated final transcript."""
        session = self._sessions.pop(session_id, None)
        if not session:
            return ""
        # Signal the worker to finish
        await session.chunks.put(None)
        try:
            await asyncio.wait_for(session.done.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            _log.warning("Deepgram stream close timed out for %s", session_id)
        if session.task:
            session.task.cancel()

        transcript = " ".join(session.finals).strip()
        if _NON_ALLOWED.search(transcript):
            return ""
        _log.debug("Deepgram stream transcript for %s: %r", session_id, transcript[:120])
        return transcript

    def has_open_stream(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def close_all(self) -> None:
        for sid in list(self._sessions.keys()):
            await self.close_stream(sid)

    def _resolve_lang(self, lang_hint: str | None) -> str:
        h = (lang_hint or "").strip().lower()
        if h == "hi":
            return "hi"
        return "en-IN"


def _mime_from_filename(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".mp4"):
        return "audio/mp4"
    if fn.endswith(".wav"):
        return "audio/wav"
    if fn.endswith(".ogg"):
        return "audio/ogg"
    return "audio/webm"
