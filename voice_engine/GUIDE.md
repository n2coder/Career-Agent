# Voice Engine — Integration Guide

A portable, production-ready real-time voice AI pipeline for any FastAPI application.
Plug in your own LLM/logic via a single callback and get a full voice agent in minutes.

---

## What's Included

| File | Purpose |
|---|---|
| `config.py` | All settings (STT, TTS, WebRTC, Redis, session) |
| `pipeline.py` | Core WebSocket handler + turn logic |
| `stt_openai.py` | OpenAI Whisper STT (batch) |
| `stt_deepgram.py` | Deepgram nova-3 STT (real-time streaming) |
| `tts.py` | OpenAI TTS with audio file caching |
| `session_token.py` | JWT-based session management |
| `session_store.py` | Redis conversation history (graceful degradation) |
| `webrtc.py` | WebRTC peer/track management |
| `static/vad-worklet.js` | Browser AudioWorklet — FFT spectral VAD |
| `static/voice-client.js` | Drop-in browser client (VAD, recorder, playback) |

---

## Architecture Overview

```
Browser                          Server
──────                          ──────
Mic → AudioWorklet (VAD)
    → MediaRecorder (500ms chunks) ──► audio_chunk_partial ──► Deepgram stream
    → final blob ─────────────────► user_audio ──────────────► close stream → transcript
                                                                     ↓
WebSocket (control) ◄────────────────────────────────────── your on_query(text, history, lang)
                                                                     ↓
                                                               LLM response
                                                                     ↓
Audio (base64) ◄──────────────────────────────────────── TTS sentence-by-sentence
Speaker ◄── WebRTC DataChannel (or base64 audio)
```

---

## Step-by-Step Integration

### Step 1 — Copy the package

Copy the entire `voice_engine/` folder into the root of your project:

```
your_project/
├── voice_engine/        ← paste here
├── app/
│   └── main.py
├── .env
└── requirements.txt
```

---

### Step 2 — Install dependencies

Add these to your `requirements.txt`:

```
fastapi
uvicorn
openai>=1.0.0
deepgram-sdk>=6.0.0
redis[asyncio]>=5.0.0
PyJWT>=2.0.0
aiortc>=1.9.0
pydantic-settings>=2.0.0
```

Then install:

```bash
pip install -r requirements.txt
```

---

### Step 3 — Configure `.env`

Create a `.env` file in your project root:

```env
# Required
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=your_deepgram_key

# STT provider: "deepgram" (recommended) or "openai"
STT_PROVIDER=deepgram
DEEPGRAM_MODEL=nova-3

# TTS voice and speed
OPENAI_TTS_VOICE=coral
OPENAI_TTS_SPEED=1.3
OPENAI_TTS_INSTRUCTIONS=Speak naturally and warmly like chatting with a friend.

# Session security (change this in production!)
SESSION_JWT_SECRET=your-strong-secret-32-chars-minimum

# Redis for conversation history (optional — app works without it)
REDIS_URL=redis://localhost:6379/0
REDIS_SESSION_ENABLED=true

# WebRTC uplink (mic audio over WebRTC instead of WebSocket)
ENABLE_WEBRTC_UPLINK=true

# Audio cache directory
AUDIO_DIR=static/audio
```

---

### Step 4 — Write your query handler

This is the **only app-specific code** you need to write.
It receives the user's transcribed speech and returns the agent's reply.

```python
# Simple string response
async def my_handler(query: str, history: list, lang: str) -> str:
    return f"You said: {query}"
```

**With OpenAI LLM:**

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-...")

async def my_handler(query: str, history: list, lang: str) -> str:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in history[-4:]:
        messages.append(msg)
    messages.append({"role": "user", "content": query})

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content
```

**With streaming (lower latency):**

```python
async def my_handler(query: str, history: list, lang: str):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in history[-4:]:
        messages.append(msg)
    messages.append({"role": "user", "content": query})

    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta  # yields text chunks → TTS starts on first sentence
```

---

### Step 5 — Wire into FastAPI

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from voice_engine.pipeline import VoicePipeline, VoiceServices
from voice_engine.config import VoiceSettings

settings = VoiceSettings()
services = None

@asynccontextmanager
async def lifespan(app):
    global services
    services = VoiceServices.build(settings)
    await services.startup()          # connects Redis, initialises Deepgram
    settings.audio_path.mkdir(parents=True, exist_ok=True)
    yield
    await services.shutdown()         # closes Redis, Deepgram streams

app = FastAPI(lifespan=lifespan)

# Serve audio files generated by TTS
app.mount("/static/audio", StaticFiles(directory=settings.audio_dir), name="audio")

# Your query handler (defined in Step 4)
async def my_handler(query: str, history: list, lang: str) -> str:
    return f"Echo: {query}"

@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    pipeline = VoicePipeline(
        services=services,
        on_query=my_handler,
        greeting="Hi! How can I help you today?",
    )
    await pipeline.run(websocket)
```

---

### Step 6 — Add the browser client to your HTML

Copy `voice_engine/static/vad-worklet.js` and `voice_engine/static/voice-client.js`
to your project's static folder, then add to your HTML:

```html
<script src="/static/voice-client.js"></script>
<script>
  const voice = new VoiceClient({
    wsUrl: "ws://localhost:8000/ws/voice",
    vadWorkletUrl: "/static/vad-worklet.js",

    onStatusChange: (status) => {
      document.getElementById("status").textContent = status;
    },
    onTextDelta: (delta) => {
      document.getElementById("response").textContent += delta;
    },
    onAudioChunk: (b64, sentence) => {
      console.log("Playing:", sentence);
    },
    onTurnEnd: (text) => {
      document.getElementById("response").textContent = text;
    },
    onError: (msg) => {
      console.error("Voice error:", msg);
    },
    onBackchannel: () => {
      // User made a short sound while agent was speaking
      document.getElementById("status").textContent = "Go on...";
    },
  });

  // Start on button click
  document.getElementById("startBtn").onclick = async () => {
    await voice.start();
  };

  // Stop
  document.getElementById("stopBtn").onclick = () => voice.stop();

  // Mute/unmute
  document.getElementById("muteBtn").onclick = () => {
    voice.setMuted(!voice._isMuted);
  };

  // Send a text query programmatically
  document.getElementById("sendBtn").onclick = () => {
    voice.sendQuery(document.getElementById("textInput").value);
  };
</script>
```

---

### Step 7 — Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at `http://localhost:8000` and start talking.

---

## VoiceClient API Reference

| Method | Description |
|---|---|
| `await voice.start()` | Request mic permission, connect WebSocket, start VAD |
| `voice.stop()` | Disconnect everything |
| `voice.setMuted(true/false)` | Mute/unmute microphone |
| `voice.sendQuery("text")` | Send a text query without speaking |
| `voice.stopAssistant()` | Interrupt current agent speech (barge-in) |

| Callback | When it fires |
|---|---|
| `onStatusChange(status)` | `"calibrating"`, `"listening"`, `"recording"`, `"processing"`, `"speaking"` |
| `onTextDelta(delta)` | Each token from the LLM stream |
| `onAudioChunk(b64, sentence)` | Each audio chunk ready to play |
| `onTurnEnd(fullText)` | Agent finished speaking |
| `onError(message)` | Any error |
| `onBackchannel()` | User made a short sound during agent speech |

---

## VoiceSettings Reference

| Setting | Default | Description |
|---|---|---|
| `STT_PROVIDER` | `deepgram` | `deepgram` or `openai` |
| `DEEPGRAM_API_KEY` | — | Your Deepgram key |
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram model |
| `OPENAI_API_KEY` | — | Your OpenAI key |
| `OPENAI_TTS_VOICE` | `coral` | TTS voice |
| `OPENAI_TTS_SPEED` | `1.3` | Playback speed (1.0 = normal) |
| `OPENAI_TTS_INSTRUCTIONS` | — | Natural language style instructions for TTS |
| `SESSION_JWT_SECRET` | `change-me` | JWT signing secret |
| `SESSION_TTL_MINUTES` | `60` | Session expiry |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `REDIS_SESSION_ENABLED` | `true` | Enable Redis history |
| `ENABLE_WEBRTC_UPLINK` | `true` | Mic audio over WebRTC |
| `PARTIAL_STT_ENABLED` | `true` | Stream 500ms chunks to Deepgram |
| `AUDIO_DIR` | `static/audio` | TTS cache directory |
| `TTS_RETENTION_MINUTES` | `120` | How long to keep cached audio |

---

## Latency Profile

| Stage | Time |
|---|---|
| VAD silence detection | ~400ms |
| Deepgram STT (streaming) | ~50–100ms |
| LLM first token | ~300–600ms |
| TTS first sentence | ~300–500ms |
| **Total first audio** | **~1.0–1.5s** |

Deepgram streams audio while the user is still speaking — by the time speech ends, the transcript is already ~80% assembled.

---

## Language Support

The pipeline detects language automatically from the transcript.
Pass `lang` from `on_query` into your LLM system prompt to control response language:

```python
async def my_handler(query: str, history: list, lang: str) -> str:
    lang_rule = "Reply in Hindi using Devanagari script." if lang == "hi" else "Reply in English."
    # add lang_rule to your system prompt
    ...
```

---

## Troubleshooting

**No audio in browser**
→ Make sure `/static/audio` is mounted and `AUDIO_DIR` matches the mounted path.

**Deepgram not loading**
→ If using a user-level pip install on Windows, set:
```bash
PYTHONPATH=C:/Users/<user>/AppData/Roaming/Python/Python311/site-packages
```
Or use `bash start.sh` which sets this automatically.

**VAD triggering on background noise**
→ Increase `voiceRatioThreshold` in `voice-client.js` `_VAD_PROFILES.desktop` from `0.40` to `0.50`.

**Voice too fast / too slow**
→ Adjust `OPENAI_TTS_SPEED` in `.env`. Range: `0.8` (slow) to `1.5` (fast). `1.3` is natural conversational pace.

**Redis unavailable**
→ The app works without Redis — history just resets each turn. Set `REDIS_SESSION_ENABLED=false` to silence the warning.
