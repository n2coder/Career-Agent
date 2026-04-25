"""Generic real-time voice pipeline for FastAPI.

HOW TO USE IN A NEW APP
=======================

1. Copy the entire voice_engine/ folder into your project.
2. Mount the static files and create a WebSocket route:

    from fastapi import FastAPI, WebSocket
    from voice_engine.pipeline import VoicePipeline, VoiceServices
    from voice_engine.config import VoiceSettings

    app = FastAPI()
    settings = VoiceSettings()

    # Build services once at startup
    services = VoiceServices.build(settings)

    # Define your app-specific query handler
    async def my_handler(query: str, history: list, lang: str) -> str:
        # Call your LLM, retrieval, etc. here
        return f"You said: {query}"

    @app.websocket("/ws/voice")
    async def voice_ws(ws: WebSocket):
        await ws.accept()
        pipeline = VoicePipeline(services, on_query=my_handler)
        await pipeline.run(ws)

3. Mount audio static files:

    from fastapi.staticfiles import StaticFiles
    app.mount("/static/audio", StaticFiles(directory=settings.audio_dir), name="audio")

4. Copy voice_engine/static/vad-worklet.js to your static folder.
   Include voice_engine/static/voice-client.js in your HTML.

QUERY HANDLER SIGNATURE
=======================

    async def on_query(
        query: str,           # transcribed user speech
        history: list[dict],  # conversation history [{role, content}, ...]
        lang: str,            # detected language: "en" or "hi"
    ) -> AsyncIterator[str] | str:
        # Return a plain string OR an async generator that yields text chunks
        # for streaming (lower latency).
        ...
"""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from fastapi import WebSocket

from voice_engine.config import VoiceSettings
from voice_engine.session_store import RedisSessionStore
from voice_engine.session_token import SessionTokenManager
from voice_engine.stt_deepgram import DeepgramSTTService
from voice_engine.stt_openai import STTService
from voice_engine.tts import TTSService

_log = logging.getLogger(__name__)

SENTENCE_BOUNDARY = re.compile(r"[.!?\u0964](?:['\"\)\]]*)\s*")


# ---------------------------------------------------------------------------
# Services container
# ---------------------------------------------------------------------------

@dataclass
class VoiceServices:
    settings: VoiceSettings
    stt: STTService
    tts: TTSService
    session_tokens: SessionTokenManager
    redis_store: RedisSessionStore
    deepgram_stt: DeepgramSTTService | None = field(default=None)

    @classmethod
    def build(cls, settings: VoiceSettings) -> "VoiceServices":
        redis_store = RedisSessionStore(settings)
        deepgram = (
            DeepgramSTTService(settings)
            if settings.stt_provider == "deepgram"
            else None
        )
        return cls(
            settings=settings,
            stt=STTService(settings),
            tts=TTSService(settings),
            session_tokens=SessionTokenManager(settings),
            redis_store=redis_store,
            deepgram_stt=deepgram,
        )

    async def startup(self) -> None:
        await self.redis_store.ping()

    async def shutdown(self) -> None:
        if self.deepgram_stt:
            await self.deepgram_stt.close_all()
        await self.redis_store.close()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class VoicePipeline:
    """
    Handles one WebSocket connection end-to-end.

    Parameters
    ----------
    services : VoiceServices
    on_query : async callable(query, history, lang) -> str | AsyncIterator[str]
        Your application logic — LLM call, retrieval, etc.
    greeting : str, optional
        First message spoken when a new session starts.
    """

    def __init__(
        self,
        services: VoiceServices,
        on_query: Callable,
        greeting: str = "Hi! I'm your voice assistant. How can I help?",
    ) -> None:
        self._svc = services
        self._on_query = on_query
        self._greeting = greeting

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, websocket: WebSocket) -> None:
        send_lock = asyncio.Lock()
        session_token = ""
        cancel_event: asyncio.Event | None = None
        active_task: asyncio.Task | None = None
        rate_bucket: deque = deque()

        async def cancel_active():
            nonlocal active_task, cancel_event
            if cancel_event:
                cancel_event.set()
            if active_task and not active_task.done():
                active_task.cancel()
                try:
                    await active_task
                except (asyncio.CancelledError, Exception):
                    pass
            active_task = None
            cancel_event = None

        try:
            async for raw in websocket.iter_text():
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue

                msg_type = str(payload.get("type", "")).strip()

                # ---- session init ----------------------------------------
                if msg_type == "start_session":
                    tok = str(payload.get("session_token", "")).strip()
                    if tok:
                        try:
                            self._svc.session_tokens.decode(tok)
                            session_token = tok
                        except Exception:
                            pass
                    if not session_token:
                        sid, session_token, _ = self._svc.session_tokens.start_session()
                    await self._send(websocket, send_lock, {
                        "type": "session_started",
                        "session_token": session_token,
                    })
                    # Play greeting
                    await cancel_active()
                    cancel_event = asyncio.Event()
                    tok_snap = session_token
                    _ce = cancel_event

                    async def _greet(_tok=tok_snap, _ce=_ce):
                        nonlocal session_token
                        session_token = await self._run_turn(
                            websocket, send_lock, _tok, self._greeting, _ce, "en"
                        )
                    active_task = asyncio.create_task(_greet())

                # ---- text query (for testing / browser speech API) --------
                elif msg_type == "user_query":
                    if not session_token:
                        continue
                    query = str(payload.get("query", "")).strip()
                    lang = str(payload.get("lang_hint", "en")).strip().lower()
                    if not query:
                        continue
                    await cancel_active()
                    cancel_event = asyncio.Event()
                    tok_snap, _ce = session_token, cancel_event

                    async def _text_turn(_q=query, _l=lang, _tok=tok_snap, _ce=_ce):
                        nonlocal session_token
                        session_token = await self._run_turn(
                            websocket, send_lock, _tok, _q, _ce, _l
                        )
                    active_task = asyncio.create_task(_text_turn())

                # ---- audio blob (MediaRecorder path) ----------------------
                elif msg_type == "user_audio":
                    if not session_token:
                        continue
                    # Rate limit
                    now = time.time()
                    s = self._svc.settings
                    while rate_bucket and now - rate_bucket[0] >= s.rate_limit_window_sec:
                        rate_bucket.popleft()
                    if len(rate_bucket) >= s.rate_limit_voice_per_window:
                        await self._send(websocket, send_lock, {"type": "error", "detail": "rate limit exceeded"})
                        continue
                    rate_bucket.append(now)

                    audio_b64 = str(payload.get("audio_b64", "")).strip()
                    lang = str(payload.get("lang_hint", "en")).strip().lower()
                    tok_snap = str(payload.get("session_token", session_token)).strip() or session_token
                    session_token = tok_snap
                    mime = str(payload.get("mime_type", "audio/webm")).strip()
                    ext = ".webm" if "webm" in mime else ".mp4" if "mp4" in mime else ".webm"
                    audio_bytes = self._decode_b64(audio_b64)
                    if not audio_bytes:
                        continue

                    await cancel_active()
                    cancel_event = asyncio.Event()
                    _ce = cancel_event

                    async def _audio_turn(_b=audio_bytes, _l=lang, _ext=ext, _tok=tok_snap, _ce=_ce):
                        nonlocal session_token
                        # Try Deepgram stream → fallback Whisper
                        query = await self._transcribe(_b, f"audio{_ext}", _l, _tok)
                        if not query:
                            await self._send(websocket, send_lock, {"type": "error", "detail": "could not transcribe audio"})
                            return
                        session_token = await self._run_turn(
                            websocket, send_lock, _tok, query, _ce, _l
                        )
                    active_task = asyncio.create_task(_audio_turn())

                # ---- partial STT chunks -----------------------------------
                elif msg_type == "audio_chunk_partial":
                    if not self._svc.settings.partial_stt_enabled or not session_token:
                        continue
                    chunk = self._decode_b64(str(payload.get("audio_b64", "")))
                    lang = str(payload.get("lang_hint", "en")).strip().lower()
                    if len(chunk) < 600:
                        continue
                    dg = self._svc.deepgram_stt
                    if dg and dg._client:
                        try:
                            sid = self._svc.session_tokens.decode(session_token).session_id
                        except Exception:
                            sid = ""
                        if sid:
                            if not dg.has_open_stream(sid):
                                asyncio.create_task(dg.open_stream(sid, lang))
                            else:
                                asyncio.create_task(dg.send_chunk(sid, chunk))

                # ---- barge-in --------------------------------------------
                elif msg_type == "barge_in":
                    await cancel_active()
                    await self._send(websocket, send_lock, {"type": "barge_in_ack"})

                # ---- ping ------------------------------------------------
                elif msg_type == "ping":
                    await self._send(websocket, send_lock, {"type": "pong"})

        except Exception:
            _log.exception("VoicePipeline error")

    # ------------------------------------------------------------------
    # Turn: LLM → TTS → stream audio
    # ------------------------------------------------------------------

    async def _run_turn(
        self,
        websocket: WebSocket,
        send_lock: asyncio.Lock,
        session_token: str,
        query: str,
        cancel_event: asyncio.Event,
        lang: str,
    ) -> str:
        svc = self._svc

        try:
            state = svc.session_tokens.decode(session_token)
        except Exception:
            _, session_token, _ = svc.session_tokens.start_session()
            state = svc.session_tokens.decode(session_token)

        # Load history from Redis
        if svc.redis_store._enabled:
            state.history = await svc.redis_store.load(state.session_id)

        await self._send(websocket, send_lock, {"type": "turn_started", "query": query})

        # Call the app's query handler
        result = self._on_query(query, state.history, lang)
        if inspect.isawaitable(result):
            result = await result

        tts_futures: list[tuple[str, asyncio.Task]] = []
        llm_done = asyncio.Event()
        buffer = ""
        full_text = ""

        def _dispatch(sentence: str) -> None:
            text = sentence.strip()
            if not text:
                return
            task = asyncio.create_task(self._tts_to_b64(text, lang))
            tts_futures.append((text, task))

        async def _drain():
            idx = sent = 0
            while True:
                if cancel_event.is_set():
                    break
                if sent < len(tts_futures):
                    sentence, task = tts_futures[sent]
                    try:
                        audio_b64 = await task
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        audio_b64 = None
                    sent += 1
                    if audio_b64 and not cancel_event.is_set():
                        await self._send(websocket, send_lock, {
                            "type": "audio_chunk",
                            "index": idx,
                            "sentence": sentence,
                            "audio_b64": audio_b64,
                        })
                        idx += 1
                elif llm_done.is_set():
                    break
                else:
                    await asyncio.sleep(0.02)

        drain_task = asyncio.create_task(_drain())

        try:
            # Handle both streaming (AsyncIterator) and plain string results
            if hasattr(result, "__aiter__"):
                async for delta in result:
                    if cancel_event.is_set():
                        break
                    full_text += delta
                    await self._send(websocket, send_lock, {"type": "text_delta", "delta": delta})
                    buffer += delta
                    ready, buffer = self._split_sentences(buffer)
                    for s in ready:
                        _dispatch(s)
            else:
                full_text = str(result)
                await self._send(websocket, send_lock, {"type": "text_delta", "delta": full_text})
                ready, buffer = self._split_sentences(full_text)
                for s in ready:
                    _dispatch(s)

            if buffer.strip():
                _dispatch(buffer)
                buffer = ""
        finally:
            llm_done.set()
            await drain_task

        await self._send(websocket, send_lock, {"type": "turn_end", "text": full_text})

        # Save history to Redis
        updated = state.history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": full_text},
        ]
        if svc.redis_store._enabled:
            await svc.redis_store.save(state.session_id, updated, svc.settings.session_ttl_minutes)

        # Refresh JWT
        return svc.session_tokens.encode(state.session_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _tts_to_b64(self, text: str, lang: str) -> str | None:
        try:
            audio_bytes = await self._svc.tts.synthesize_to_bytes(text)
            return base64.b64encode(audio_bytes).decode()
        except Exception:
            _log.warning("TTS failed for: %r", text[:60])
            return None

    async def _transcribe(self, audio: bytes, filename: str, lang: str, token: str) -> str:
        svc = self._svc
        dg = svc.deepgram_stt
        query = ""
        if dg and dg._client:
            try:
                sid = svc.session_tokens.decode(token).session_id
            except Exception:
                sid = ""
            if sid and dg.has_open_stream(sid):
                await dg.send_chunk(sid, audio)
                query = await dg.close_stream(sid)
        if not query:
            try:
                query = await svc.stt.transcribe_bytes(audio, filename=filename, lang_hint=lang)
            except Exception:
                pass
        return (query or "").strip()

    @staticmethod
    def _decode_b64(b64: str) -> bytes:
        try:
            return base64.b64decode(b64 + "==")
        except Exception:
            return b""

    @staticmethod
    def _split_sentences(text: str) -> tuple[list[str], str]:
        parts = SENTENCE_BOUNDARY.split(text)
        if len(parts) <= 1:
            return [], text
        complete = []
        for m in SENTENCE_BOUNDARY.finditer(text):
            end = m.end()
            complete.append(text[:end].strip())
            text = text[end:]
        return complete, text

    @staticmethod
    async def _send(ws: WebSocket, lock: asyncio.Lock, data: dict) -> None:
        async with lock:
            try:
                await ws.send_text(json.dumps(data, ensure_ascii=False))
            except Exception:
                pass
